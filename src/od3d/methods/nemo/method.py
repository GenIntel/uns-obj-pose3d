import time
from typing import List
import od3d.io
from od3d.cv.statistics.standard_devation import mean_avg_std_with_id
from od3d.methods.method import OD3D_Method
from od3d.datasets.dataset import OD3D_Dataset
from od3d.benchmark.results import OD3D_Results
from omegaconf import DictConfig
import pytorch3d.transforms
import pandas as pd
import numpy as np
from torch.utils.data import RandomSampler
import logging

from od3d.cv.metric.pose import get_pose_diff_in_rad
logger = logging.getLogger(__name__)
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from od3d.cv.geometry.transform import se3_exp_map
from od3d.cv.visual.show import imgs_to_img
from od3d.cv.geometry.mesh import Meshes
from pathlib import Path
from od3d.cv.geometry.transform import transf4x4_from_spherical, tform4x4, rot3x3, inv_tform4x4, tform4x4_broadcast
from od3d.cv.visual.show import show_img
import torchvision
from od3d.cv.visual.blend import blend_rgb
from od3d.cv.visual.sample import sample_pxl2d_pts
from tqdm import tqdm
from od3d.cv.geometry.mesh import MESH_RENDER_MODALITIES
# note: math is actually used by config
import math
from od3d.datasets.co3d import CO3D

from od3d.cv.io import image_as_wandb_image
from od3d.cv.visual.resize import resize
from od3d.models.model import OD3D_Model

from od3d.cv.geometry.grid import get_pxl2d_like
from od3d.cv.geometry.fit3d2d import batchwise_fit_se3_to_corresp_3d_2d_and_masks
from od3d.cv.transforms.transform import OD3D_Transform
from od3d.cv.transforms.sequential import SequentialTransform

from typing import Dict
from od3d.data.ext_enum import ExtEnum

import matplotlib.pyplot as plt

plt.switch_backend('Agg')
from od3d.cv.visual.show import get_img_from_plot
from od3d.cv.visual.draw import draw_text_in_rgb

class VISUAL_MODALITIES(str, ExtEnum):
    PRED_VERTS_NCDS_IN_RGB = 'pred_verts_ncds_in_rgb'
    GT_VERTS_NCDS_IN_RGB = 'gt_verts_ncds_in_rgb'
    PRED_VS_GT_VERTS_NCDS_IN_RGB = 'pred_vs_gt_verts_ncds_in_rgb'
    NET_FEATS_NEAREST_VERTS = 'net_feats_nearest_verts'
    SIM_PXL = 'sim_pxl'
    SAMPLES = 'samples'

class SIM_FEATS_MESH_WITH_IMAGE(str, ExtEnum):
    VERTS2D = 'verts2d'
    RENDERED = 'rendered'

class NeMo(OD3D_Method):
    def setup(self):
        pass


    def __init__(
            self,
            config: DictConfig,
            logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)

        self.device = 'cuda:0'

        # init Network
        self.net = OD3D_Model(config.model)

        self.transform_train = SequentialTransform([
            OD3D_Transform.subclasses[config.train.transform.class_name].create_from_config(config=config.train.transform),
            self.net.transform,
        ])
        self.transform_test = SequentialTransform([
            OD3D_Transform.subclasses[config.test.transform.class_name].create_from_config(config=config.test.transform),
            self.net.transform
        ])

        # if config.train.transform.random_color:
        #     self.transform_train = SequentialTransform([
        #         RandomCenterZoom3D.create_from_config(config.train.transform.random_center_zoom3d),
        #         RGB_Random(),
        #         self.net.transform
        #     ])
        # else:
        #     self.transform_train = SequentialTransform([
        #         RandomCenterZoom3D.create_from_config(config.train.transform.random_center_zoom3d),
        #         self.net.transform
        #     ])
        #
        # self.transform_test = SequentialTransform([
        #         CenterZoom3D.create_from_config(config.test.transform),
        #         self.net.transform
        # ])

        # init Meshes / Features
        self.total_params = sum(p.numel() for p in self.net.parameters())
        # self.path_shapenemo = Path(config.path_shapenemo)
        # self.fpaths_meshes_shapenemo = [self.path_shapenemo.joinpath(cls, '01.off') for cls in config.categories]
        self.fpaths_meshes = [self.config.fpaths_meshes[cls] for cls in config.categories]
        fpaths_meshes_tform_obj = self.config.get('fpaths_meshes_tform_obj', None)
        if fpaths_meshes_tform_obj is not None:
            self.fpaths_meshes_tform_obj = [fpaths_meshes_tform_obj[cls] for cls in config.categories]
        else:
            self.fpaths_meshes_tform_obj = [None for _ in config.categories]

        self.meshes = Meshes.load_from_files(fpaths_meshes=self.fpaths_meshes, fpaths_meshes_tforms=self.fpaths_meshes_tform_obj)
        self.meshes.gaussian_splat_enabled = self.config.meshes_gaussian_splat_enabled
        self.meshes.gaussian_splat_opacity = self.config.meshes_gaussian_splat_opacity
        self.meshes.geodesic_prob_sigma = self.config.train.geodesic_prob_sigma
        self.meshes.pt3d_raster_perspective_correct = self.config.meshes_pt3d_raster_perspective_correct
        self.meshes.gaussian_splat_pts3d_size_rel_to_neighbor_dist  = self.config.meshes_gaussian_splat_pts3d_size_rel_to_neighbor_dist
        self.meshes_ranges = self.meshes.get_ranges().detach().cuda()
        self.refine_update_max = torch.Tensor(self.config.inference.refine.dims_grad_max).cuda()[None,].expand(self.meshes_ranges.shape[0], 6).clone()
        self.refine_update_max[:, :3] = self.refine_update_max[:, :3] * self.meshes_ranges

        #self.meshes.rgb = (self.meshes.geodesic_prob[3, :, None].repeat(1, 3)).clamp(0, 1)
        # self.meshes.show()

        logger.info(f'loading meshes from following fpaths: {self.fpaths_meshes}...')
        # self.meshes.show()
        self.verts_count_max = self.meshes.verts_counts_max
        self.mem_verts_feats_count = len(config.categories) * self.verts_count_max
        self.mem_clutter_feats_count = config.num_noise * config.max_group
        self.mem_count = self.mem_verts_feats_count + self.mem_clutter_feats_count

        self.feats_bank_count = self.verts_count_max * len(self.meshes) + 1

        self.clutter_feats = torch.nn.Parameter(torch.randn(size=(1, self.net.out_dim), device=self.device),
                                                requires_grad=True)
        self.meshes.set_feats_cat_with_pad(torch.nn.Parameter(
            torch.randn(size=(self.verts_count_max * len(self.meshes), self.net.out_dim), device=self.device),
            requires_grad=True))

        # self.meshes.set_feats_cat_with_pad(torch.nn.Parameter(torch.randn(size=(self.verts_count_max * len(self.meshes), self.net.feat_dim), device=self.device), requires_grad=True))

        # dict to save estimated tforms, sequence : tform,
        self.seq_obj_tform4x4_est_obj = {}
        self.seq_obj_tform4x4_est_obj_sim = {}

        self.normalize_feats()

        if self.config.train.loss == 'cross_entropy':
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        elif self.config.train.loss == 'cross_entropy_smooth_geo':
            from od3d.cv.metric.cross_entropy_smooth import CrossEntropyLabelsSmoothed
            labels_smoothed = self.meshes.get_geodesic_prob_with_noise().to(device=self.device)
            self.criterion = CrossEntropyLabelsSmoothed(labels_smoothed=labels_smoothed)
        elif self.config.train.loss == 'nll_softmax':
            self.softmax = torch.nn.LogSoftmax(dim=1)
            self.criterion = torch.nn.NLLLoss().cuda()
        elif self.config.train.loss == 'nll_clip':
            self.criterion = torch.nn.NLLLoss().cuda()
        elif self.config.train.loss == 'nll_affine_to_prob':
            self.criterion = torch.nn.NLLLoss().cuda()
        elif self.config.train.loss == 'l2':
            self.criterion = torch.nn.MSELoss().cuda()
        elif self.config.train.loss == 'l2_squared':
            self.criterion = torch.nn.MSELoss().cuda()


        # self.net = torch.nn.DataParallel(self.net).cuda()
        self.net.cuda()
        self.meshes.cuda()
        self.net.eval()

        self.optim = od3d.io.get_obj_from_config(config=self.config.train.optimizer, params=list(self.net.parameters()) + [self.meshes.feats] + [self.clutter_feats])
        self.scheduler = od3d.io.get_obj_from_config(self.optim, config=self.config.train.scheduler)

        # load checkpoint
        if config.get("checkpoint", None) is not None:
            self.load_checkpoint(Path(config.checkpoint))
        elif config.get("checkpoint_old", None) is not None:
            self.load_checkpoint_old(Path(config.checkpoint_old))
        # load_mesh(config.path_shapenemo)

        # self.meshes.show()

        # self.verts_feats = checkpoint["memory"][:self.mem_verts_feats_count].clone().detach().cpu()
        # note: somehow vertices are stored in wrong order of classes (starting with last class tvmonitor until first class aeroplane
        # self.verts_feats = self.verts_feats.reshape(len(self.meshes), self.verts_count_max, -1).flip(dims=(0,)).reshape(len(self.meshes) * self.verts_count_max, -1)
        self.down_sample_rate = self.net.downsample_rate

    def normalize_feats(self):
        if self.config.bank_feats_normalize:
            self.clutter_feats.data = self.clutter_feats.detach() / self.clutter_feats.detach().norm(dim=-1, keepdim=True)
            self.meshes.feats.data = self.meshes.feats.detach() / self.meshes.feats.detach().norm(dim=-1, keepdim=True)
            # self.meshes.set_feats_cat(self.meshes.feats.detach() / self.meshes.feats.detach().norm(dim=-1, keepdim=True))
            # logger.info(self.clutter_feats[:1])
            # logger.info(self.meshes.feats[:1])
    def calc_sim(self, comb, featsA, featsB):
        """
        Expand and permute a tensor based on the einsum equation.

        Parameters:
            tensor (torch.Tensor): Input tensor.
            equation (str): Einsum equation specifying the dimensions.

        Returns:
            torch.Tensor: Expanded and permuted tensor.
        """
        if self.config.bank_feats_distribution == 'von-mises-fisher':
            return torch.einsum(comb, featsA, featsB)
        elif self.config.bank_feats_distribution == 'gaussian':
            from od3d.cv.geometry.dist import einsum_cdist
            return -einsum_cdist(comb, featsA, featsB)
        else:
            msg = f'Unknown distribution {self.config.distribution}'
            raise NotImplementedError(msg)

    def load_checkpoint_old(self, path_checkpoint):
        fpaths_meshes_old = list(self.config.fpaths_meshes.values())
        meshes_old = Meshes.load_from_files(fpaths_meshes=fpaths_meshes_old)
        verts_count_max = meshes_old.verts_counts_max
        mem_verts_feats_count = len(fpaths_meshes_old) * verts_count_max
        checkpoint = torch.load(path_checkpoint, map_location="cuda:0")
        self.net.backbone.net = torch.nn.DataParallel(self.net.backbone.net).cuda()
        self.net.backbone.net.load_state_dict(checkpoint["state"], strict=False)
        self.net.backbone.net = self.net.backbone.net.module
        self.clutter_feats = checkpoint["memory"][mem_verts_feats_count:].clone().detach().cpu()
        # self.clutter_feats = self.clutter_feats.mean(dim=0, keepdim=True)
        self.clutter_feats = torch.nn.Parameter(self.clutter_feats.to(device=self.device), requires_grad=True)

        verts_feats = []
        map_mesh_id_to_old_id = [fpaths_meshes_old.index(fpath_mesh) for fpath_mesh in self.fpaths_meshes]
        for i in range(len(self.fpaths_meshes)):
            mesh_old_id = map_mesh_id_to_old_id[i]
            verts_feats.append(checkpoint["memory"][mesh_old_id * verts_count_max: (mesh_old_id + 1) * verts_count_max].clone().detach().cpu())
        self.meshes.set_feats_cat_with_pad(torch.cat(verts_feats, dim=0))

    def save_checkpoint(self, path_checkpoint: Path):
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'meshes_feats': self.meshes.feats,
            'clutter_feats': self.clutter_feats
        }, path_checkpoint)

    def load_checkpoint(self, path_checkpoint):
        checkpoint = torch.load(path_checkpoint)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.meshes.set_feats_cat(checkpoint['meshes_feats'])
        self.clutter_feats = checkpoint['clutter_feats']

    @property
    def fpath_checkpoint(self):
        return self.logging_dir.joinpath(self.rfpath_checkpoint)

    @property
    def rfpath_checkpoint(self):
        return Path('nemo.ckpt')

    def train(self, datasets_train: Dict[str, OD3D_Dataset], datasets_val: Dict[str, OD3D_Dataset]):
        score_metric_name = 'pose/acc_pi18'  # 'pose/acc_pi18' 'pose/acc_pi6'
        score_ckpt_val = 0.
        score_latest = 0.
        self.save_checkpoint(path_checkpoint=self.fpath_checkpoint)

        if 'main' in datasets_val.keys():
            dataset_train_sub = datasets_train['labeled']
        else:
            dataset_train_sub, dataset_val_sub = datasets_train['labeled'].get_split(fraction1=1. - self.config.train.val_fraction,
                                                                                     fraction2=self.config.train.val_fraction,
                                                                                     split=self.config.train.split)
            datasets_val['main'] = dataset_val_sub

        for epoch in range(self.config.train.epochs):
            if self.config.train.val and self.config.train.epochs_to_next_test > 0 and epoch % self.config.train.epochs_to_next_test == 0:
                for dataset_val_key, dataset_val in datasets_val.items():
                    results_val = self.test(dataset_val, val=True)
                    results_val.log_with_prefix(prefix=f'val/{dataset_val.name}')
                    if dataset_val_key == 'main':
                        score_latest = results_val[score_metric_name]

                if not self.config.train.early_stopping or score_latest > score_ckpt_val:
                    score_ckpt_val = score_latest
                    self.save_checkpoint(path_checkpoint=self.fpath_checkpoint)

            results_epoch = self.train_epoch(dataset=dataset_train_sub)
            results_epoch.log_with_prefix('train')
        self.load_checkpoint(path_checkpoint=self.fpath_checkpoint)

    def test(self, dataset: OD3D_Dataset, val=False):
        # note: ensure that checkpoint is saved for checkpointed runs
        if not self.fpath_checkpoint.exists():
            self.save_checkpoint(path_checkpoint=self.fpath_checkpoint)

        logger.info(f'test dataset {dataset.name}')
        self.net.eval()
        self.meshes.feats.requires_grad = False
        clutter_feats = self.clutter_feats.detach()
        dataset.transform = self.transform_test
        if not isinstance(dataset, CO3D):
            dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.test.dataloader.batch_size,
                                                     shuffle=False,
                                                     collate_fn=dataset.collate_fn,
                                                     num_workers=self.config.test.dataloader.num_workers,
                                                     pin_memory=self.config.test.dataloader.pin_memory)
            logger.info(f"Dataset contains {len(dataset)} frames.")

        else:
            dict_category_sequences = {category: list(sequence_dict.keys()) for category, sequence_dict in dataset.dict_nested_frames.items()}
            dataset_sub = dataset.get_subset_by_sequences(dict_category_sequences=dict_category_sequences,
                                                          frames_count_max_per_sequence=self.config.multiview.batch_size)


            dataloader = torch.utils.data.DataLoader(dataset=dataset_sub, batch_size=self.config.multiview.batch_size,
                                                     shuffle=False,
                                                     collate_fn=dataset_sub.collate_fn,
                                                     num_workers=self.config.test.dataloader.num_workers,
                                                     pin_memory=self.config.test.dataloader.pin_memory)
            logger.info(f"Dataset contains {len(dataset_sub)} frames.")

        results_epoch = OD3D_Results(logging_dir=self.logging_dir)
        for i, batch in tqdm(enumerate(iter(dataloader))):
            batch.to(device=self.device)

            if not isinstance(dataset, CO3D):
                results_batch = self.inference_batch_single_view(batch=batch)
            else:
                results_batch = self.inference_batch_multiview(batch=batch, return_samples_with_sim=True)
            results_epoch += results_batch

            if not val and self.config.test.save_results:
                results_visual_batch = self.get_results_visual_batch(batch=batch, results_batch=results_batch,
                                                                         config_visualize=self.config.test.visualize)
                results_visual_batch.save_visual(prefix=f'test/{dataset.name}')

        count_pred_frames = len(results_epoch['item_id'])
        logger.info(f'Predicted {count_pred_frames} frames.')
        if not val and self.config.test.save_results:
            results_epoch.save_with_dataset(prefix='test', dataset=dataset)

        if not isinstance(dataset, CO3D):
            results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset,
                                                     config_visualize=self.config.test.visualize)
        else:
            results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset_sub,
                                                     config_visualize=self.config.test.visualize)


        results_epoch = results_epoch.mean()
        results_epoch += results_visual
        return results_epoch


    def train_epoch(self, dataset: OD3D_Dataset) -> OD3D_Results:
        self.net.train()
        self.meshes.del_pre_rendered()
        self.meshes.feats.requires_grad = True
        dataset.transform = self.transform_train
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=self.config.train.dataloader.batch_size,
                                                       shuffle=True,
                                                       collate_fn=dataset.collate_fn,
                                                       num_workers=self.config.train.dataloader.num_workers,
                                                       pin_memory=self.config.train.dataloader.pin_memory)

        results_epoch = OD3D_Results(logging_dir=self.logging_dir)
        accumulate_steps = 0
        for i, batch in enumerate(iter(dataloader_train)):
            results_batch: OD3D_Results = self.train_batch(batch=batch)
            results_batch.log_with_prefix('train')
            accumulate_steps += 1
            if accumulate_steps % self.config.train.batch_accumulate_to_next_step == 0:
                self.optim.step()
                self.normalize_feats()
                self.optim.zero_grad()

            results_epoch += results_batch

        self.scheduler.step()
        self.optim.zero_grad()

        # results_epoch.log_dict_to_dir(name=f'train_frames/{dataset.name}')

        results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset,
                                                 config_visualize=self.config.train.visualize)
        results_epoch = results_epoch.mean()
        results_epoch += results_visual
        return results_epoch


    def train_batch(self, batch) -> OD3D_Results:
        results_batch = OD3D_Results(logging_dir=self.logging_dir)

        batch.to(device=self.device)

        batch.cam_tform4x4_obj = batch.cam_tform4x4_obj.detach()

        logger.info(f'batch.category_id {batch.category_id}')
        logger.info(f'batch.size {batch.size}')

        # B x x N x 2
        vts2d, vts2d_mask = self.meshes.verts2d(cams_intr4x4=batch.cam_intr4x4,
                                                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                                                imgs_sizes=batch.size, mesh_ids=batch.category_id,
                                                down_sample_rate=self.down_sample_rate)

        N = vts2d.shape[1]
        # B x F+N x C
        logger.info(f'batch.size {batch.size}')
        feats2d_net = self.net(batch.rgb)

        logger.info(f'batch.size {batch.size}')
        feats2d_net_mask = torch.ones(size=(feats2d_net.shape[0], 1, feats2d_net.shape[2], feats2d_net.shape[3])).to(device=self.device)

        if self.config.train.use_mask_rgb:
            feats2d_net_mask = 1. * resize(batch.rgb_mask, H_out=feats2d_net.shape[2], W_out=feats2d_net.shape[3])

        if self.config.train.use_mask_object:
            feats2d_net_mask = feats2d_net_mask * 1. * resize(batch.mask, H_out=feats2d_net.shape[2],
                                                              W_out=feats2d_net.shape[3])
        if self.config.train.use_mask_rendered_object:
            logger.info(f'batch.size {batch.size}')
            feats2d_net_mask = feats2d_net_mask * self.meshes.render_feats(
                cams_intr4x4=batch.cam_intr4x4,
                cams_tform4x4_obj=batch.cam_tform4x4_obj,
                imgs_sizes=batch.size, meshes_ids=batch.category_id,
                down_sample_rate=self.down_sample_rate,
                modality=MESH_RENDER_MODALITIES.MASK)

        H, W = feats2d_net.shape[-2:]
        xy = torch.stack(
            torch.meshgrid(torch.arange(W, device=self.device), torch.arange(H, device=self.device),
                           indexing='xy'), dim=0)  # HxW
        prob_noise = (1. - 1. * feats2d_net_mask).clamp(0, 1).flatten(1)
        prob_noise[prob_noise.sum(dim=-1) <= 0.] = 1.
        noise2d = xy.flatten(1)[:, torch.multinomial(prob_noise, self.config.num_noise, replacement=True)].permute(1, 2, 0)

        # note: visual for debug
        # from od3d.cv.visual.show import show_imgs
        # from od3d.cv.visual.draw import draw_pixels
        # prob_noise_pxls = prob_noise.reshape(-1, 1, H, W).clone().repeat(1, 3, 1, 1)
        # for b in range(len(prob_noise_pxls)):
        #     prob_noise_pxls[b] = draw_pixels(prob_noise_pxls[b], noise2d[b])
        # show_imgs(prob_noise_pxls)

        vts2d_feats2d_net_mask = sample_pxl2d_pts(feats2d_net_mask, pxl2d=torch.cat([vts2d], dim=1))
        vts2d_mask = vts2d_mask * (vts2d_feats2d_net_mask[:, :, 0] > 0.5)
        net_feats = sample_pxl2d_pts(feats2d_net, pxl2d=torch.cat([vts2d, noise2d], dim=1))

        C = net_feats.shape[2]
        # args: X: Bx3xHxW, keypoint_positions: BxNx2, obj_mask: BxHxW ensures that noise is sampled outside of object mask
        # returns: BxF+NxC


        # net_feats = net_feats[:, :].reshape(-1, net_feats.shape[-1])
        logger.info(batch.category_id)
        batch_vts_ids = self.meshes.get_verts_and_noise_ids_stacked(batch.category_id.tolist(),
                                                                    count_noise_ids=self.config.num_noise)

        # weighting with similarity score
        # net_feats = net_feats * (batch.cam_tform4x4_obj_sim[:, None, None] ** 4)

        # sim_weight = batch.cam_tform4x4_obj_sim[:, None].expand(*net_feats.shape[:2])
        # sim_weight = torch.cat([sim_weight[:, :N][mask_vts2d_vsbl], sim_weight[:, N:].reshape(-1)], dim=0)

        batch_vts_ids = torch.cat([batch_vts_ids[:, :N][vts2d_mask], batch_vts_ids[:, N:].reshape(-1)],
                                  dim=0)
        net_feats = torch.cat([net_feats[:, :N][vts2d_mask], net_feats[:, N:].reshape(-1, C)], dim=0)

        # batch_vts_ids = self.meshes.get_feats_ids_stacked(batch.category_id.tolist())

        bank_feats = torch.cat([self.meshes.feats, self.clutter_feats], dim=0)


        if self.config.train.bank_feats_update == 'loss_gradient':
            sim = self.calc_sim('nc,vc->nv', net_feats, bank_feats)
        elif self.config.train.bank_feats_update == 'normalize_loss_gradient':
            sim = self.calc_sim('nc,vc->nv', net_feats, torch.nn.functional.normalize(bank_feats, dim=1))
        elif self.config.train.bank_feats_update == 'moving_average':
            sim = self.calc_sim('nc,vc->nv', net_feats, bank_feats.detach())
            bank_feats_new = self.config.train.alpha * bank_feats[batch_vts_ids].detach() + (1. - self.config.train.alpha) * net_feats.detach()
            batch_vts_ids_unique, batch_vts_ids_unique_inverse, batch_vts_ids_unique_counts = batch_vts_ids.unique(return_inverse=True, return_counts=True)
            bank_feats_new = torch.einsum('nk,nc->kc', torch.nn.functional.one_hot(batch_vts_ids_unique_inverse).to(dtype= bank_feats_new.dtype, device= bank_feats_new.device), bank_feats_new) / batch_vts_ids_unique_counts[:, None]
            bank_feats[batch_vts_ids_unique].data = bank_feats_new
            self.normalize_feats()
        else:
            logger.error(f'unknown bank_feats_update: {self.config.train.bank_feats_update}')
            sim = None

        sim_batchwise_borders = torch.cat([torch.LongTensor([0]).to(device=vts2d_mask.device), vts2d_mask.sum(dim=1).cumsum(dim=0)], dim=0)
        sim_batchwise = torch.stack([sim[sim_batchwise_borders[b]:sim_batchwise_borders[b+1]].max(dim=-1)[0].mean() for b in range(len(sim_batchwise_borders)-1)], dim=0)
        # in case there are 0 vertices inside one image
        sim_batchwise[sim_batchwise.isnan()] = 0.
        results_batch['sim'] = sim_batchwise

        # loss: cross_entropy  # cross_entropy, nll_softmax, nll_clip, nll_affine_to_prob
        # bank_feats_update: loss_gradient  # loss_gradient, normalize_loss_gradient, moving_average, loss
        loss = self.criterion(sim / self.config.train.T, batch_vts_ids)

        loss.backward()
        logger.info(f'loss {loss.item()}')
        results_batch['noise2d'] = noise2d
        results_batch['loss'] = loss[None,]
        results_batch['item_id'] = batch.item_id
        results_batch['name_unique'] = batch.name_unique
        results_batch['gt_cam_tform4x4_obj'] = batch.cam_tform4x4_obj

        return results_batch

    """
    def get_sim_cam_tform4x4_obj(self, batch, cam_intr4x4, cam_tform4x4_obj, broadcast_batch_and_cams=False):
        with torch.no_grad():
            net_feats2d = self.net(batch.rgb)
            mesh_feats2d_rendered = self.meshes.render_feats(cams_tform4x4_obj=cam_tform4x4_obj,
                                                             cams_intr4x4=cam_intr4x4,
                                                             imgs_sizes=batch.size, meshes_ids=batch.category_id,
                                                             down_sample_rate=self.down_sample_rate,
                                                             broadcast_batch_and_cams=broadcast_batch_and_cams)

            sim = self.get_sim_feats2d_net_and_rendered(feats2d_net=net_feats2d, feats2d_rendered=mesh_feats2d_rendered)
        return sim
    """


    def inference_batch_single_view(self, batch, return_samples_with_sim=True):
        results = OD3D_Results(logging_dir=self.logging_dir)
        B = len(batch)

        """
        # these parameters are used in prev. version
        batch.cam_tform4x4_obj[:, 2, 3] = 5. * 6. # 5. * 6.
        batch.cam_tform4x4_obj[:, 0, 3] = 0.
        batch.cam_tform4x4_obj[:, 1, 3] = 0.
        batch.cam_intr4x4[:, 0, 0] = 3000.
        batch.cam_intr4x4[:, 1, 1] = 3000.
        batch.cam_intr4x4[:, 0, 2] = batch.size[1] / 2.
        batch.cam_intr4x4[:, 1, 2] = batch.size[0] / 2.
        """


        time_loaded = time.time()
        with torch.no_grad():
            feats2d_net = self.net(batch.rgb)
            feats2d_net_mask = resize(batch.rgb_mask, H_out=feats2d_net.shape[2], W_out=feats2d_net.shape[3])
            if self.config.inference.use_mask_object:
                feats2d_net_mask = feats2d_net_mask * 1. * resize(batch.mask, H_out=feats2d_net.shape[2], W_out=feats2d_net.shape[3])

            time_pred_net_feats2d = time.time()
            # logger.info(
            #    f"predicted net feats2d, took {(time_pred_net_feats2d - time_loaded):.3f}")
            results['time_feats2d'] = torch.Tensor([time_pred_net_feats2d - time_loaded,]) / B

            meshes_scores = []
            for mesh_id in range(len(self.meshes)):
                # logger.info(f'calc score for mesh {self.config.categories[mesh_id]}')
                bank_feats = torch.cat([self.meshes.get_feats_with_mesh_id(mesh_id), self.clutter_feats.detach()],
                                       dim=0)
                # inner_feats2d_net_bank_vts_max_vals = torch.sum(net_feats2d[:, None] * bank_feats[None, :, :, None, None], dim=2, keepdim=True).max(dim=1).values
                out_shape = feats2d_net.shape[:1] + torch.Size([1]) + feats2d_net.shape[2:]
                inner_feats2d_net_bank_vts_max_vals = self.calc_sim('bchw,kc->bkhw', feats2d_net, bank_feats).max(dim=1,
                                                                                                                 keepdim=True).values
                # inner_feats2d_net_bank_vts_max_vals, inner_feats2d_net_bank_vts_max_ids = inner_feats2d.max(dim=1)
                # show_img(self.meshes.get_verts_with_mesh_id[mesh_id][inner_feats2d_net_bank_vts_max_ids[0, 0]].permute(2, 0, 1), normalize=True)
                # show_img(inner_feats2d_net_bank_vts_max_vals[0])
                mesh_score = inner_feats2d_net_bank_vts_max_vals.flatten(1).mean(dim=1)
                # clutter_score = inner_feats2d[:, -clutter_feats.shape[0]:].mean(dim=1).flatten(1).mean(dim=1)
                # mesh_score -= clutter_score
                meshes_scores.append(mesh_score)
            meshes_scores = torch.stack(meshes_scores, dim=-1)
            pred_class_scores, pred_class_ids = meshes_scores.max(dim=1)

            # logger.info(f'pred class ids {pred_class_ids}')
            time_pred_class = time.time()
            # logger.info(f"predicted class: {self.config.categories[int(pred_class_ids[0])]}, took {(time_pred_class - time_pred_net_feats2d):.3f}")

            results['time_class'] = torch.Tensor([time_pred_class - time_pred_net_feats2d,]) / B

            b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4 = self.get_samples(config_sample=self.config.inference.sample,
                                                                                       cam_intr4x4=batch.cam_intr4x4,
                                                                                       cam_tform4x4_obj=batch.cam_tform4x4_obj,
                                                                                       feats2d_net=feats2d_net,
                                                                                       categories_ids=batch.category_id,
                                                                                       feats2d_net_mask=feats2d_net_mask)

            # if self.config.inference.live:
            #     from od3d.cv.visual.show import show_imgs
            #     show_imgs(
            #         blend_rgb(batch.rgb[:1], (self.meshes.render_feats(cams_tform4x4_obj=b_cams_multiview_tform4x4_obj[0],
            #                                                           cams_intr4x4=b_cams_multiview_intr4x4[0],
            #                                                           imgs_sizes=batch.size,
            #                                                           meshes_ids=batch.category_id[:1],
            #                                                           modality=MESH_RENDER_MODALITIES.VERTS_NCDS,
            #                                                           broadcast_batch_and_cams=True)[0]).to(dtype=batch.rgb.dtype)), duration=-1)

            #  OPTION A: Use 2d gradient of rendered features
            sim = self.get_sim_feats2d_net_with_cams(
                feats2d_net=feats2d_net,
                feats2d_net_mask=feats2d_net_mask,
                cam_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                cam_intr4x4=b_cams_multiview_intr4x4,
                categories_ids=batch.category_id,
                broadcast_batch_and_cams=True,
                only_use_rendered_inliers=self.config.inference.only_use_rendered_inliers,
                allow_clutter=self.config.inference.allow_clutter,
                use_sigmoid=self.config.inference.use_sigmoid
            )

            if return_samples_with_sim:
                results['samples_cam_tform4x4_obj'] = b_cams_multiview_tform4x4_obj
                results['samples_cam_intr4x4'] = b_cams_multiview_intr4x4
                results['samples_sim'] = sim

            mesh_multiple_cams_loss = -sim


            mesh_cam_loss_min_val, mesh_cam_loss_min_id = mesh_multiple_cams_loss.min(dim=1)

            cam_tform4x4_obj = b_cams_multiview_tform4x4_obj[:, mesh_cam_loss_min_id].permute(2, 3, 0, 1).diagonal(
                dim1=-2, dim2=-1).permute(2, 0, 1)

        if self.config.inference.refine.enabled:
            obj_tform6_tmp = torch.nn.Parameter(torch.zeros(size=(B, 6)).to(device=cam_tform4x4_obj.device),
                                                requires_grad=True)
            # transl: 0, 1, 2 rot: 3, 4, 5
            optim_inference = torch.optim.Adam(
                params=[obj_tform6_tmp],
                lr=self.config.inference.optimizer.lr,
                betas=(self.config.inference.optimizer.beta0, self.config.inference.optimizer.beta1),
            )

            time_before_pose_iterative = time.time()
            cam_tform4x4_obj = tform4x4(cam_tform4x4_obj.detach(), se3_exp_map(obj_tform6_tmp))

            refine_update_max = self.refine_update_max[batch.category_id].clone()
            for epoch in range(self.config.inference.optimizer.epochs):

                cam_tform4x4_obj = tform4x4(cam_tform4x4_obj.detach(), se3_exp_map(obj_tform6_tmp.detach()))
                obj_tform6_tmp.data[:, :] = 0.
                cam_tform4x4_obj = tform4x4(cam_tform4x4_obj.detach(), se3_exp_map(obj_tform6_tmp))

                sim, sim_pxl = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                                  feats2d_net_mask=feats2d_net_mask,
                                                                  cam_tform4x4_obj=cam_tform4x4_obj,
                                                                  cam_intr4x4=batch.cam_intr4x4,
                                                                  categories_ids=batch.category_id, return_sim_pxl=True,
                                                                  broadcast_batch_and_cams=False,
                                                                  pre_rendered=False,
                                                                  only_use_rendered_inliers=self.config.inference.only_use_rendered_inliers,
                                                                  allow_clutter=self.config.inference.allow_clutter,
                                                                  use_sigmoid=self.config.inference.use_sigmoid)
                mesh_cam_loss = -sim

                if self.config.inference.live:
                    show_img(
                        blend_rgb(batch.rgb[0], (self.meshes.render_feats(cams_tform4x4_obj=cam_tform4x4_obj[0:0 + 1],
                                                                          cams_intr4x4=batch.cam_intr4x4[0:0 + 1],
                                                                          imgs_sizes=batch.size,
                                                                          meshes_ids=batch.category_id[0:0 + 1],
                                                                          modality=MESH_RENDER_MODALITIES.VERTS_NCDS)[
                            0]).to(dtype=batch.rgb.dtype)), duration=1)

                loss = mesh_cam_loss.sum()
                loss.backward()
                optim_inference.step()
                optim_inference.zero_grad()

                # detach update
                obj_tform6_tmp.data[:, self.config.inference.refine.dims_detached] = 0.
                # clip update
                refine_update_mask = obj_tform6_tmp.data.abs() > refine_update_max
                obj_tform6_tmp.data[refine_update_mask] = obj_tform6_tmp.data[refine_update_mask].sign() * refine_update_max[refine_update_mask]


            cam_tform4x4_obj = tform4x4(cam_tform4x4_obj.detach(), se3_exp_map(obj_tform6_tmp.detach()))

            results['time_pose_iterative'] = torch.Tensor([time.time() - time_before_pose_iterative,]) / B

        cam_tform4x4_obj = cam_tform4x4_obj.clone().detach()

        results['time_pose'] = torch.Tensor([time.time() - time_pred_class,]) / B
        batch_rot_diff_rad = get_pose_diff_in_rad(pred_tform4x4=cam_tform4x4_obj, gt_tform4x4=batch.cam_tform4x4_obj)
        results['rot_diff_rad'] = batch_rot_diff_rad
        for cat_id, cat in enumerate(self.config.categories):
            results[f'{cat}_rot_diff_rad'] = batch_rot_diff_rad[batch.category_id == cat_id]
        results['label_gt'] = batch.category_id
        results['label_pred'] = pred_class_ids
        results['label_names'] = self.config.categories
        results['sim'] = sim
        results['cam_tform4x4_obj'] = cam_tform4x4_obj
        results['item_id'] = batch.item_id
        results['name_unique'] = batch.name_unique

        return results


    def inference_batch_multiview(self, batch, return_samples_with_sim=True):
        results = OD3D_Results(logging_dir=self.logging_dir)
        B = len(batch)

        """
        # these parameters are used in prev. version
        batch.cam_tform4x4_obj[:, 2, 3] = 5. * 6. # 5. * 6.
        batch.cam_tform4x4_obj[:, 0, 3] = 0.
        batch.cam_tform4x4_obj[:, 1, 3] = 0.
        batch.cam_intr4x4[:, 0, 0] = 3000.
        batch.cam_intr4x4[:, 1, 1] = 3000.
        batch.cam_intr4x4[:, 0, 2] = batch.size[1] / 2.
        batch.cam_intr4x4[:, 1, 2] = batch.size[0] / 2.
        """

        time_loaded = time.time()
        with torch.no_grad():
            feats2d_net = self.net(batch.rgb)
            feats2d_net_mask = resize(batch.rgb_mask, H_out=feats2d_net.shape[2], W_out=feats2d_net.shape[3])
            if self.config.inference.use_mask_object:
                feats2d_net_mask = feats2d_net_mask * 1. * resize(batch.mask, H_out=feats2d_net.shape[2],
                                                                  W_out=feats2d_net.shape[3])

            time_pred_net_feats2d = time.time()
            # logger.info(
            #    f"predicted net feats2d, took {(time_pred_net_feats2d - time_loaded):.3f}")
            results['time_feats2d'] = torch.Tensor([time_pred_net_feats2d - time_loaded,]) / B

            meshes_scores = []
            for mesh_id in range(len(self.meshes)):
                # logger.info(f'calc score for mesh {self.config.categories[mesh_id]}')
                bank_feats = torch.cat([self.meshes.get_feats_with_mesh_id(mesh_id), self.clutter_feats.detach()],
                                       dim=0)
                # inner_feats2d_net_bank_vts_max_vals = torch.sum(net_feats2d[:, None] * bank_feats[None, :, :, None, None], dim=2, keepdim=True).max(dim=1).values
                out_shape = feats2d_net.shape[:1] + torch.Size([1]) + feats2d_net.shape[2:]
                inner_feats2d_net_bank_vts_max_vals = self.calc_sim('bchw,kc->bkhw', feats2d_net, bank_feats).max(dim=1,
                                                                                                                 keepdim=True).values
                # inner_feats2d_net_bank_vts_max_vals, inner_feats2d_net_bank_vts_max_ids = inner_feats2d.max(dim=1)
                # show_img(self.meshes.get_verts_with_mesh_id[mesh_id][inner_feats2d_net_bank_vts_max_ids[0, 0]].permute(2, 0, 1), normalize=True)
                # show_img(inner_feats2d_net_bank_vts_max_vals[0])
                mesh_score = inner_feats2d_net_bank_vts_max_vals.flatten(1).mean(dim=1)
                # clutter_score = inner_feats2d[:, -clutter_feats.shape[0]:].mean(dim=1).flatten(1).mean(dim=1)
                # mesh_score -= clutter_score
                meshes_scores.append(mesh_score)
            meshes_scores = torch.stack(meshes_scores, dim=-1)
            pred_class_scores, pred_class_ids = meshes_scores.max(dim=1)

            # logger.info(f'pred class ids {pred_class_ids}')
            time_pred_class = time.time()
            # logger.info(f"predicted class: {self.config.categories[int(pred_class_ids[0])]}, took {(time_pred_class - time_pred_net_feats2d):.3f}")

            results['time_class'] = torch.Tensor([time_pred_class - time_pred_net_feats2d,]) / B

            b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4 = self.get_samples(config_sample=self.config.inference.sample,
                                                                                       cam_intr4x4=batch.cam_intr4x4, #[:1],
                                                                                       cam_tform4x4_obj=batch.cam_tform4x4_obj, #[:1],
                                                                                       feats2d_net=feats2d_net, #[:1],
                                                                                       categories_ids=batch.category_id, #[:1],
                                                                                       feats2d_net_mask=feats2d_net_mask, #[:1],
                                                                                       multiview=True)


            #  OPTION A: Use 2d gradient of rendered features
            sim = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                     cam_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                                                     cam_intr4x4=b_cams_multiview_intr4x4,
                                                     categories_ids=batch.category_id,
                                                     broadcast_batch_and_cams=True,
                                                     feats2d_net_mask=feats2d_net_mask,
                                                     pre_rendered=False,
                                                     only_use_rendered_inliers=self.config.inference.only_use_rendered_inliers,
                                                     allow_clutter=self.config.inference.allow_clutter,
                                                     use_sigmoid=self.config.inference.use_sigmoid)

            sim = sim.mean(dim=0, keepdim=True).expand(*sim.shape)

            if return_samples_with_sim:
                results['samples_cam_tform4x4_obj'] = b_cams_multiview_tform4x4_obj
                results['samples_cam_intr4x4'] = b_cams_multiview_intr4x4
                results['samples_sim'] = sim

            mesh_multiple_cams_loss = -sim


            mesh_cam_loss_min_val, mesh_cam_loss_min_id = mesh_multiple_cams_loss.min(dim=1)

            objs_multiview_tform4x4_cuboid_front = tform4x4_broadcast(inv_tform4x4(batch.cam_tform4x4_obj[:1])[:, None], b_cams_multiview_tform4x4_obj)
            obj_tform4x4_cuboid_front = objs_multiview_tform4x4_cuboid_front[0, mesh_cam_loss_min_id[0]]

            #cam_tform4x4_obj = b_cams_multiview_tform4x4_obj[:, mesh_cam_loss_min_id].permute(2, 3, 0, 1).diagonal(
            #    dim1=-2, dim2=-1).permute(2, 0, 1)

        if self.config.inference.refine.enabled:
            obj_tform6_tmp = torch.nn.Parameter(torch.zeros(size=(1, 6)).to(device=obj_tform4x4_cuboid_front.device),
                                                requires_grad=True)

            optim_inference = torch.optim.Adam(
                params=[obj_tform6_tmp],
                lr=self.config.inference.optimizer.lr,
                betas=(self.config.inference.optimizer.beta0, self.config.inference.optimizer.beta1),
            )

            time_before_pose_iterative = time.time()
            obj_tform4x4_cuboid_front = tform4x4(obj_tform4x4_cuboid_front.detach(), se3_exp_map(obj_tform6_tmp))

            refine_update_max = self.refine_update_max[batch.category_id[:1]].clone()

            for epoch in range(self.config.inference.optimizer.epochs):
                obj_tform4x4_cuboid_front = tform4x4(obj_tform4x4_cuboid_front.detach(), se3_exp_map(obj_tform6_tmp.detach()))
                obj_tform6_tmp.data[:, :] = 0.
                obj_tform4x4_cuboid_front = tform4x4(obj_tform4x4_cuboid_front.detach(), se3_exp_map(obj_tform6_tmp))

                sim, sim_pxl = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                                  cam_tform4x4_obj=tform4x4_broadcast(batch.cam_tform4x4_obj, obj_tform4x4_cuboid_front),
                                                                  cam_intr4x4=batch.cam_intr4x4,
                                                                  categories_ids=batch.category_id, return_sim_pxl=True,
                                                                  broadcast_batch_and_cams=False,
                                                                  feats2d_net_mask=feats2d_net_mask, pre_rendered=False,
                                                                  only_use_rendered_inliers=self.config.inference.only_use_rendered_inliers,
                                                                  allow_clutter=self.config.inference.allow_clutter,
                                                                  use_sigmoid=self.config.inference.use_sigmoid)
                mesh_cam_loss = -sim

                if self.config.inference.live:
                    show_img(
                        blend_rgb(batch.rgb[0], (self.meshes.render_feats(cams_tform4x4_obj=tform4x4_broadcast(batch.cam_tform4x4_obj, obj_tform4x4_cuboid_front)[0:0 + 1],
                                                                          cams_intr4x4=batch.cam_intr4x4[0:0 + 1],
                                                                          imgs_sizes=batch.size,
                                                                          meshes_ids=batch.category_id[0:0 + 1],
                                                                          modality=MESH_RENDER_MODALITIES.VERTS_NCDS)[
                            0]).to(dtype=batch.rgb.dtype)), duration=1)

                loss = mesh_cam_loss.mean()
                loss.backward()
                optim_inference.step()
                optim_inference.zero_grad()

                # detach update
                obj_tform6_tmp.data[:, self.config.inference.refine.dims_detached] = 0.
                # clip update
                refine_update_mask = obj_tform6_tmp.data.abs() > refine_update_max
                obj_tform6_tmp.data[refine_update_mask] = obj_tform6_tmp.data[refine_update_mask].sign() * refine_update_max[refine_update_mask]


            obj_tform4x4_cuboid_front = tform4x4(obj_tform4x4_cuboid_front.detach(), se3_exp_map(obj_tform6_tmp.detach()))

            results['time_pose_iterative'] = torch.Tensor([time.time() - time_before_pose_iterative,]) / B

        obj_tform4x4_cuboid_front = obj_tform4x4_cuboid_front.clone().detach()
        cam_tform4x4_obj = tform4x4_broadcast(batch.cam_tform4x4_obj, obj_tform4x4_cuboid_front)

        results['time_pose'] = torch.Tensor([time.time() - time_pred_class,]) / B
        batch_rot_diff_rad = get_pose_diff_in_rad(pred_tform4x4=cam_tform4x4_obj, gt_tform4x4=batch.cam_tform4x4_obj)
        results['rot_diff_rad'] = batch_rot_diff_rad
        for cat_id, cat in enumerate(self.config.categories):
            results[f'{cat}_rot_diff_rad'] = batch_rot_diff_rad[batch.category_id == cat_id]
        results['label_gt'] = batch.category_id
        results['label_pred'] = pred_class_ids
        results['sim'] = sim.mean(dim=0, keepdim=True).expand(*sim.shape)
        results['cam_tform4x4_obj'] = cam_tform4x4_obj
        results['obj_tform4x4_cuboid_front'] = obj_tform4x4_cuboid_front
        results['item_id'] = batch.item_id
        results['name_unique'] = batch.name_unique

        return results

    def get_results_visual_batch(self, batch, results_batch: OD3D_Results, config_visualize: DictConfig,
                                 dict_name_unique_to_sel_name = None, dict_name_unique_to_result_id=None,
                                 caption_metrics = ['sim', 'rot_diff_rad']):
        results_batch_visual = OD3D_Results(logging_dir=self.logging_dir)
        modalities = config_visualize.modalities
        if len(modalities) == 0:
            return results_batch_visual

        down_sample_rate = config_visualize.down_sample_rate
        samples_sorted = config_visualize.samples_sorted
        samples_scores = config_visualize.samples_scores
        samples_scores = config_visualize.samples_scores
        live = config_visualize.live


        with torch.no_grad():
            batch.to(device=self.device)
            B = len(batch)
            if dict_name_unique_to_result_id is not None:
                batch_result_ids = torch.LongTensor(
                    [dict_name_unique_to_result_id[batch.name_unique[b]] for b in range(B)]).to(device=self.device)
            else:
                batch_result_ids = torch.LongTensor(range(B)).to(device=self.device)

            if 'gt_cam_tform4x4_obj' in results_batch.keys():
                batch.cam_tform4x4_obj = results_batch['gt_cam_tform4x4_obj'].to(device=self.device)[batch_result_ids]
            if 'noise2d' in results_batch.keys():
                batch.noise2d = results_batch['noise2d'].to(device=self.device)[batch_result_ids]

            if dict_name_unique_to_sel_name is not None:
                batch_sel_names = [dict_name_unique_to_sel_name[batch.name_unique[b]] for b in range(B)]
            else:
                batch_sel_names = [batch.name_unique[b] for b in range(B)]

            batch_names = [batch.name_unique[b] for b in range(B)]
            batch_sel_scores = []
            for b in range(B):
                batch_sel_scores.append('\n'.join([f'{metric}={results_batch[metric].to(device=self.device)[batch_result_ids[b]].cpu().detach().item():.3f}'
                                                   for metric in caption_metrics if metric in results_batch.keys()]))

            feats2d_net = self.net(batch.rgb)
            # feats2d_net = resize(feats2d_net,
            #                     scale_factor=self.down_sample_rate / config_visualize.down_sample_rate)

            if VISUAL_MODALITIES.NET_FEATS_NEAREST_VERTS in modalities:

                logger.info('create net_feats_nearest_verts ...')
                verts3d = self.get_nearest_verts3d_to_feats2d_net(feats2d_net=feats2d_net,
                                                                  categories_ids=batch.category_id,
                                                                  zero_if_sim_clutter_larger=True)
                verts3d = resize(verts3d, scale_factor=self.down_sample_rate / down_sample_rate)
                for b in range(len(batch)):
                    img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / down_sample_rate),
                                    verts3d[b])
                    results_batch_visual[
                        f'visual/{VISUAL_MODALITIES.NET_FEATS_NEAREST_VERTS}/{batch_sel_names[b]}'] = image_as_wandb_image(
                        img, caption=f'{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}')
                    if live:
                        show_img(img)

            if VISUAL_MODALITIES.SAMPLES in modalities:
                logger.info('create samples ...')
                s_cam_tform4x4_obj = results_batch['samples_cam_tform4x4_obj'].to(device=self.device)[batch_result_ids]
                s_cam_intr4x4 = results_batch['samples_cam_intr4x4'].to(device=self.device)[batch_result_ids]
                sim = results_batch['samples_sim'].to(device=self.device)[batch_result_ids]

                """        
                s_cam_tform4x4_obj, s_cam_intr4x4 = self.get_samples(config_sample=self.config.inference.sample,
                                                                                           cam_intr4x4=batch.cam_intr4x4,
                                                                                           cam_tform4x4_obj=batch.cam_tform4x4_obj,
                                                                                           feats2d_net=feats2d_net,
                                                                                           categories_ids=batch.category_id)
                sim = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                         cam_tform4x4_obj=s_cam_tform4x4_obj,
                                                         cam_intr4x4=s_cam_intr4x4,
                                                         categories_ids=batch.category_id,
                                                         broadcast_batch_and_cams=True)
                """

                ncds = self.get_ncds_with_cam(cam_tform4x4_obj=s_cam_tform4x4_obj,
                                              cam_intr4x4=s_cam_intr4x4, size=batch.size,
                                              categories_ids=batch.category_id,
                                              down_sample_rate=down_sample_rate,
                                              broadcast_batch_and_cams=True, pre_rendered=False)

                for b in range(len(batch)):
                    imgs = ncds[b]
                    imgs_sim = sim[b][:].expand(*sim[b].shape)  # , *mesh_feats2d_rendered.shape[-2:]
                    if self.config.inference.sample.method == 'uniform':
                        imgs = imgs.reshape(self.config.inference.sample.uniform.azim.steps,
                                            self.config.inference.sample.uniform.elev.steps,
                                            self.config.inference.sample.uniform.theta.steps,
                                            *imgs.shape[-3:])[:, :, :]
                        imgs_sim = imgs_sim.reshape(self.config.inference.sample.uniform.azim.steps,
                                                    self.config.inference.sample.uniform.elev.steps,
                                                    self.config.inference.sample.uniform.theta.steps)[:, :, :]
                    imgs = blend_rgb(resize(batch.rgb[b], scale_factor=1. / down_sample_rate), imgs)

                    if samples_sorted:
                        imgs_sim = imgs_sim.flatten(0)
                        imgs = imgs.reshape(-1, *imgs.shape[-3:])
                        imgs_sim_ids = imgs_sim.sort(descending=True)[1]
                        imgs_sim_ids = imgs_sim_ids[:49]
                        imgs = imgs[imgs_sim_ids]
                        imgs_sim = imgs_sim[imgs_sim_ids]

                    if samples_scores:
                        logger.info('create plot samples scores...')

                        plt.ioff()
                        fig, ax = plt.subplots()
                        ax.plot(imgs_sim.detach().cpu().numpy(), label='sim')  # density=False would make counts
                        # ax.set_ylim(0., 1.)
                        # ax.ylabel('sim')
                        # ax.xlabel('samples')

                        img = get_img_from_plot(ax=ax, fig=fig)
                        plt.close(fig)
                        img = resize(img, H_out=imgs.shape[-2], W_out=imgs.shape[-1])
                        img = draw_text_in_rgb(img, fontScale=0.4, lineThickness=2, fontColor=(0, 0, 0),
                                               text=f'{batch_sel_scores[b]}\nmin={imgs_sim.min().item():.3f}\nmax={imgs_sim.max().item():.3f}')
                        imgs = torch.cat([imgs, img[None,].to(device=imgs.device)], dim=0)
                        # resize(img, )
                        """
                        samples_score_size = imgs.shape[-1] // 5
                        imgs[..., -samples_score_size:, -samples_score_size:] = (
                                255 * imgs_sim.reshape(*imgs_sim.shape, 1, 1, 1).expand(*imgs.shape[:-2],
                                                                                        samples_score_size,
                                                                                        samples_score_size)).to(
                            torch.uint8)
                        """

                    img = imgs_to_img(imgs)
                    results_batch_visual[f'visual/{VISUAL_MODALITIES.SAMPLES}/{batch_sel_names[b]}'] = image_as_wandb_image(img,
                                                                                                               caption=f'{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}, min={imgs_sim.min().item():.3f}, max={imgs_sim.max().item():.3f}')
                    if live:
                        show_img(img)

            if VISUAL_MODALITIES.SIM_PXL in modalities:
                logger.info('create sim pxl...')
                batch_pred_label = results_batch['label_pred'].to(device=self.device)[batch_result_ids]
                batch_pred_cam_tform4x4 = results_batch['cam_tform4x4_obj'].to(device=self.device)[batch_result_ids]
                sim, sim_pxl = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                                  cam_intr4x4=batch.cam_intr4x4,
                                                                  cam_tform4x4_obj=batch_pred_cam_tform4x4,
                                                                  categories_ids=batch.category_id, return_sim_pxl=True,
                                                                  broadcast_batch_and_cams=False,
                                                                  sim_feats_mesh_with_image=SIM_FEATS_MESH_WITH_IMAGE.RENDERED,
                                                                  pre_rendered=False,
                                                                  only_use_rendered_inliers=self.config.inference.only_use_rendered_inliers,
                                                                  allow_clutter=self.config.inference.allow_clutter,
                                                                  use_sigmoid=self.config.inference.use_sigmoid)

                sim_pxl = resize(sim_pxl, scale_factor=self.down_sample_rate / down_sample_rate)
                for b in range(len(batch)):
                    img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / down_sample_rate),
                                    sim_pxl[b])
                    results_batch_visual[f'visual/{VISUAL_MODALITIES.SIM_PXL}/{batch_sel_names[b]}'] = image_as_wandb_image(img,
                                                                                                               caption=f'{batch_sel_names[b]}, {batch_names[b]}, mean sim={sim[b].item()}')
                    if live:
                        show_img(img)

            if VISUAL_MODALITIES.PRED_VERTS_NCDS_IN_RGB in modalities or VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB in modalities:
                logger.info('create pred verts ncds...')
                batch_pred_label = results_batch['label_pred'].to(device=self.device)[batch_result_ids]
                batch_pred_cam_tform4x4 = results_batch['cam_tform4x4_obj'].to(device=self.device)[batch_result_ids]
                pred_verts_ncds = self.get_ncds_with_cam(cam_intr4x4=batch.cam_intr4x4,
                                                         cam_tform4x4_obj=batch_pred_cam_tform4x4,
                                                         categories_ids=batch.category_id, size=batch.size,
                                                         down_sample_rate=down_sample_rate,
                                                         pre_rendered=False)
                if VISUAL_MODALITIES.PRED_VERTS_NCDS_IN_RGB in modalities:
                    for b in range(len(batch)):
                        img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / down_sample_rate),
                                        pred_verts_ncds[b])
                        results_batch_visual[
                            f'visual/{VISUAL_MODALITIES.PRED_VERTS_NCDS_IN_RGB}/{batch_sel_names[b]}'] = image_as_wandb_image(
                            img, caption=f'{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}')
                        if live:
                            show_img(img)

            if VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB in modalities or VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB in modalities:
                logger.info('create gt verts ncds...')
                gt_verts_ncds = self.get_ncds_with_cam(cam_intr4x4=batch.cam_intr4x4,
                                                       cam_tform4x4_obj=batch.cam_tform4x4_obj,
                                                       categories_ids=batch.category_id, size=batch.size,
                                                       down_sample_rate=down_sample_rate,
                                                       pre_rendered=False)
                if VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB in modalities:
                    for b in range(len(batch)):
                        img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / down_sample_rate),
                                        gt_verts_ncds[b])
                        if 'noise2d' in results_batch.keys():
                            from od3d.cv.visual.draw import draw_pixels
                            img = draw_pixels(img, batch.noise2d[b] * (
                                        self.down_sample_rate / down_sample_rate))

                        results_batch_visual[
                            f'visual/{VISUAL_MODALITIES.GT_VERTS_NCDS_IN_RGB}/{batch_sel_names[b]}'] = image_as_wandb_image(
                            img, caption=f'{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}')
                        if live:
                            show_img(img)
            if VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB in modalities:
                logger.info('create pred vs gt verts ncds...')
                for b in range(len(batch)):
                    # if 'noise2d' in results
                    img1 = blend_rgb(resize(batch.rgb[b], scale_factor=1. / down_sample_rate),
                                     pred_verts_ncds[b])
                    img2 = blend_rgb(resize(batch.rgb[b], scale_factor=1. / down_sample_rate),
                                     gt_verts_ncds[b])
                    img = imgs_to_img(torch.stack([img1, img2], dim=0)[None,])

                    results_batch_visual[
                        f'visual/{VISUAL_MODALITIES.PRED_VS_GT_VERTS_NCDS_IN_RGB}/{batch_sel_names[b]}'] = image_as_wandb_image(
                        img, caption=f'{batch_sel_names[b]}, {batch_names[b]}, {batch_sel_scores[b]}')
                    if live:
                        show_img(img)
        return results_batch_visual

    def get_results_visual(self, results_epoch, dataset: OD3D_Dataset, config_visualize: DictConfig,
                           filter_name_unique=True, caption_metrics=['sim', 'rot_diff_rad']):
        results = OD3D_Results(logging_dir=self.logging_dir)
        count_best = config_visualize.count_best
        count_worst = config_visualize.count_worst
        count_rand = config_visualize.count_rand
        modalities = config_visualize.modalities
        if len(modalities) == 0:
            return results

        if 'rot_diff_rad' in results_epoch.keys():
            rank_metric_name = 'rot_diff_rad'
            # sorts values ascending
            epoch_ranked_ids = results_epoch[rank_metric_name].sort(dim=0)[1]
        elif 'sim' in results_epoch.keys():
            rank_metric_name = 'sim'
            # sorts values descending
            epoch_ranked_ids = results_epoch[rank_metric_name].sort(dim=0, descending=True)[1]
        else:
            logger.warning(f'Could not find a suitable rank metric in results {results_epoch.keys()}')
            return results

        if filter_name_unique and 'name_unique' in results_epoch.keys() and len(results_epoch['name_unique']) > 0:
            # this only groups the ranked elements depending on their category / sequence etc.
            # https://stackoverflow.com/questions/51408344/pandas-dataframe-interleaved-reordering
            group_names = list(set(['/'.join(name_unique.split('/')[:-1]) for name_unique in results_epoch['name_unique']]))
            group_ids = [group_id for result_id in range(len(results_epoch['name_unique'])) for group_id, group_name in enumerate(group_names) if results_epoch['name_unique'][epoch_ranked_ids[result_id]].startswith(group_name)]
            df = pd.DataFrame(np.stack([epoch_ranked_ids, np.array(group_ids)], axis=-1), columns=['rank', 'group'])
            epoch_ranked_ids = torch.from_numpy(df.loc[df.groupby("group").cumcount().sort_values(kind='mergesort').index]['rank'].values)

            df = df[::-1]
            epoch_ranked_ids_worst = torch.from_numpy(df.loc[df.groupby("group").cumcount().sort_values(kind='mergesort').index]['rank'].values)
        else:
            epoch_ranked_ids_worst = epoch_ranked_ids.flip(dims=(0,))

        epoch_best_ids = epoch_ranked_ids[:count_best]
        epoch_best_names = [f'best/{i+1}' for i in range(len(epoch_best_ids))]
        epoch_worst_ids = epoch_ranked_ids_worst[:count_worst]
        epoch_worst_names = [f'worst/{len(epoch_worst_ids) - i}' for i in range(len(epoch_worst_ids))]
        epoch_rand_ids = epoch_ranked_ids[torch.randperm(len(epoch_ranked_ids))[:count_rand]]
        epoch_rand_names = [f'rand/{i+1}' for i in range(len(epoch_rand_ids))]

        sel_rank_ids = torch.cat([epoch_best_ids, epoch_worst_ids, epoch_rand_ids], dim=0)
        sel_item_ids = results_epoch['item_id'][sel_rank_ids]
        sel_names = epoch_best_names + epoch_worst_names + epoch_rand_names
        sel_name_unique = [results_epoch['name_unique'][id] for id in sel_rank_ids]
        dict_name_unique_to_result_id = dict(zip(sel_name_unique, sel_rank_ids))
        dict_name_unique_to_sel_name = dict(zip(sel_name_unique, sel_names))
        logger.info('create dataset ...')
        dataset_visualize = dataset.get_subset_with_item_ids(item_ids=sel_item_ids)

        logger.info('create dataloader ...')
        dataloader = torch.utils.data.DataLoader(dataset=dataset_visualize, batch_size=self.config.test.dataloader.batch_size,
                                                 shuffle=False,
                                                 collate_fn=dataset.collate_fn,
                                                 num_workers=self.config.test.dataloader.num_workers,
                                                 pin_memory=self.config.test.dataloader.pin_memory)
        for i, batch in tqdm(enumerate(iter(dataloader))):
            results += self.get_results_visual_batch(batch, results_epoch, config_visualize=config_visualize,
                                                     dict_name_unique_to_sel_name=dict_name_unique_to_sel_name,
                                                     caption_metrics=caption_metrics,
                                                     dict_name_unique_to_result_id=dict_name_unique_to_result_id)
        return results


    def get_nearest_verts3d_to_feats2d_net(self, feats2d_net, categories_ids, zero_if_sim_clutter_larger=False,
                                           return_sim_texture_and_clutter=False):
        B = len(feats2d_net)
        sim_nearest_texture_vals, sim_nearest_texture_ids = self.calc_sim('bchw,bvc->bvhw', feats2d_net,
                                                                         self.meshes.
                                                                         get_feats_stacked_with_mesh_ids(categories_ids)
                                                                         .detach()).max(dim=1, keepdim=True)

        sim_nearest_texture_verts3d = torch.stack([self.meshes.get_verts_stacked_with_mesh_ids(
            categories_ids[b: b + 1])[0, sim_nearest_texture_ids[b, 0]] for b in range(B)], dim=0)

        if zero_if_sim_clutter_larger or return_sim_texture_and_clutter:
            sim_clutter = \
                self.calc_sim('bchw,nc->bnhw', feats2d_net, self.clutter_feats.detach()).max(dim=1, keepdim=True)[0]
            # sim_clutter = self.calc_sim('bchw,nc->bnhw', net_feats2d, self.clutter_feats.detach()).mean(dim=1, keepdim=True)
            if zero_if_sim_clutter_larger:
                sim_nearest_texture_verts3d[sim_clutter[:, 0] > sim_nearest_texture_vals[:, 0]] = 0.

        sim_nearest_texture_verts3d = sim_nearest_texture_verts3d.permute(0, 3, 1, 2)
        if return_sim_texture_and_clutter:
            return sim_nearest_texture_verts3d, sim_nearest_texture_vals, sim_clutter

        return sim_nearest_texture_verts3d

    def get_ncds_with_cam(self, cam_intr4x4: torch.Tensor, cam_tform4x4_obj: torch.Tensor, size: torch.Tensor, categories_ids: torch.Tensor, down_sample_rate=1., broadcast_batch_and_cams=False, pre_rendered: bool= None):
        if pre_rendered is None:
            pre_rendered = self.config.inference.pre_rendered

        if pre_rendered:
            return self.meshes.get_pre_rendered_feats(cams_tform4x4_obj=cam_tform4x4_obj,
                                                      cams_intr4x4=cam_intr4x4,
                                                      imgs_sizes=size, meshes_ids=categories_ids,
                                                      modality=MESH_RENDER_MODALITIES.VERTS_NCDS,
                                                      down_sample_rate=down_sample_rate,
                                                      broadcast_batch_and_cams=broadcast_batch_and_cams
                                                      )
        else:
            return self.meshes.render_feats(cams_tform4x4_obj=cam_tform4x4_obj,
                                        cams_intr4x4=cam_intr4x4,
                                        imgs_sizes=size, meshes_ids=categories_ids,
                                        modality=MESH_RENDER_MODALITIES.VERTS_NCDS,
                                        down_sample_rate=down_sample_rate,
                                        broadcast_batch_and_cams=broadcast_batch_and_cams)

    def get_sim_feats2d_net_with_cams(self, feats2d_net, cam_tform4x4_obj, cam_intr4x4, categories_ids, return_sim_pxl=False,
                                      broadcast_batch_and_cams=False, feats2d_net_mask=None, sim_feats_mesh_with_image: SIM_FEATS_MESH_WITH_IMAGE=None, pre_rendered: bool=None, only_use_rendered_inliers=False, allow_clutter=True, use_sigmoid=False):
        if sim_feats_mesh_with_image is None:
            sim_feats_mesh_with_image = self.config.inference.sim_feats_mesh_with_image
        if pre_rendered is None:
            pre_rendered = self.config.inference.pre_rendered
        size = torch.Tensor([feats2d_net.shape[2] * self.down_sample_rate, feats2d_net.shape[3] * self.down_sample_rate]).to(device=feats2d_net.device)
        if sim_feats_mesh_with_image == SIM_FEATS_MESH_WITH_IMAGE.RENDERED:
            if pre_rendered:
                mesh_feats2d_rendered = self.meshes.get_pre_rendered_feats(cams_tform4x4_obj=cam_tform4x4_obj,
                                                                           cams_intr4x4=cam_intr4x4,
                                                                           imgs_sizes=size, meshes_ids=categories_ids,
                                                                           modality=MESH_RENDER_MODALITIES.FEATS,
                                                                           down_sample_rate=self.down_sample_rate,
                                                                           broadcast_batch_and_cams=broadcast_batch_and_cams
                                                                           )
            else:
                mesh_feats2d_rendered = self.meshes.render_feats(cams_tform4x4_obj=cam_tform4x4_obj,
                                                                 cams_intr4x4=cam_intr4x4,
                                                                 imgs_sizes=size, meshes_ids=categories_ids,
                                                                 down_sample_rate=self.down_sample_rate,
                                                                 broadcast_batch_and_cams=broadcast_batch_and_cams)

            return self.get_sim_feats2d_net_and_rendered(feats2d_net=feats2d_net, feats2d_rendered=mesh_feats2d_rendered, return_sim_pxl=return_sim_pxl, feats2d_net_mask=feats2d_net_mask, only_use_rendered_inliers=only_use_rendered_inliers, allow_clutter=allow_clutter, use_sigmoid=use_sigmoid)

        elif sim_feats_mesh_with_image == SIM_FEATS_MESH_WITH_IMAGE.VERTS2D:
            verts2d_mesh, verts2d_mesh_mask = self.meshes.verts2d(cams_intr4x4=cam_intr4x4,
                                                                  cams_tform4x4_obj=cam_tform4x4_obj,
                                                                  imgs_sizes=size, mesh_ids=categories_ids,
                                                                  down_sample_rate=self.down_sample_rate,
                                                                  broadcast_batch_and_cams=broadcast_batch_and_cams)
            return self.get_sim_feats2d_net_and_verts(categories_ids=categories_ids, feats2d_net=feats2d_net, verts2d_mesh=verts2d_mesh, verts2d_mesh_mask=verts2d_mesh_mask, return_sim_pxl=return_sim_pxl, feats2d_net_mask=feats2d_net_mask)


    def get_nearest_corresp2d3d(self, feats2d_net, meshes_ids, feats2d_net_mask: torch.Tensor=None):
        nearest_verts3d, sim_texture, sim_clutter = self.get_nearest_verts3d_to_feats2d_net(feats2d_net, meshes_ids, return_sim_texture_and_clutter=True)
        nearest_verts2d = get_pxl2d_like(nearest_verts3d.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # H=sim_clutter.shape[1], W=sim_clutter.shape[2], dtype=sim_nearest_texture_verts.dtype, device=sim_nearest_texture_verts.device)[None,].expand()
        # prob_corresp2d3d = ((sim_texture + 1) / 2) * (1-((sim_clutter+1)/ 2)) #  (sim_clutter < sim_texture) * sim_texture
        prob_corresp2d3d = torch.exp(sim_texture) / (torch.exp(sim_texture) + torch.exp(sim_clutter))

        prob_corresp2d3d *= feats2d_net_mask

        return nearest_verts3d, nearest_verts2d, prob_corresp2d3d

    def get_sim_feats2d_net_and_verts(self, categories_ids, feats2d_net, verts2d_mesh, verts2d_mesh_mask, return_sim_pxl=False, feats2d_net_mask=None):
        """
        Args:
            categories_ids (torch.Tensor): B, ids of categories=meshes to which the similarty should be calculated
            feats2d_net (torch.Tensor): BxCxHxW
            verts2d_mesh (torch.Tensor): Bx(T)xVx2
            verts2d_mesh_mask (torch.Tensor): Bx(T)xV
            return_sim_pxl (bool): Indicates whether pixelwise similarity should be returned or not.
            feats2d_net_mask (torch.Tensor): BxCxHxW

        Returns:
            sim (torch.Tensor): BxV, or Bx1 if rendered features is 4-dimensional.
            sim_pxl (torch.Tensor, optional): BxTxHxW, or Bx1xHxW if rendered features is 4-dimensional.

        """
        B, C, H, W = feats2d_net.shape
        V = verts2d_mesh.shape[-2]

        verts2d_mesh_ids = self.meshes.get_verts_and_noise_ids_stacked(categories_ids, count_noise_ids=0)  # B x V
        feats2d_mesh = self.meshes.feats[verts2d_mesh_ids]  # B x V x C

        if verts2d_mesh.dim() == 4:
            T = verts2d_mesh.shape[1]
            feats2d_net_sampled = sample_pxl2d_pts(feats2d_net, pxl2d=verts2d_mesh.reshape(B, -1, 2)).reshape(B, -1, V, C)
            if feats2d_net_mask is not None:
                feats2d_net_mask_sampled = sample_pxl2d_pts(feats2d_net_mask, pxl2d=verts2d_mesh.reshape(B, -1, 2)).reshape(B, -1, V)
        elif verts2d_mesh.dim() == 3:
            # B x T x V x C
            T = 1
            feats2d_net_sampled = sample_pxl2d_pts(feats2d_net, pxl2d=verts2d_mesh)[:, None]
            if feats2d_net_mask is not None:
                # B x T x V
                feats2d_net_mask_sampled = sample_pxl2d_pts(feats2d_net_mask, pxl2d=verts2d_mesh).reshape(B, -1, V)

        else:
            T = 1
            feats2d_net_mask_sampled = None
            feats2d_net_sampled = None
            logger.error(f'Unexpected dimension of verts2d_mesh {verts2d_mesh.shape}')
        # [verts2d_mesh_mask]
        sim_clutter = self.calc_sim('btvc,nc->bntv', feats2d_net_sampled, self.clutter_feats.detach()).max(dim=1, keepdim=False)[0]
        sim_texture_multiple_cams = self.calc_sim('btvc,bvc->btv', feats2d_net_sampled, feats2d_mesh)
        sim_pxl = torch.max(sim_texture_multiple_cams, sim_clutter)

        if verts2d_mesh_mask.dim() == 3:
            sim_pxl_mask = verts2d_mesh_mask
        elif verts2d_mesh_mask.dim() == 2:
            sim_pxl_mask = verts2d_mesh_mask[:, None]
        else:
            sim_pxl_mask = None
            logger.error(f'Unexpected dimension of verts2d_mesh_mask {verts2d_mesh_mask.shape}')

        if feats2d_net_mask is not None:
            sim_pxl_mask = sim_pxl_mask * feats2d_net_mask_sampled

        sim_pxl *= sim_pxl_mask
        sim = sim_pxl.flatten(2).sum(dim=-1) / (sim_pxl_mask.flatten(2).sum(dim=-1) + 1e-10)

        sim_pxl2d = torch.zeros(size=(B, T, H, W), dtype=feats2d_net.dtype, device=feats2d_net.device)
        #torch.gather(input=sim_pxl2d, dim=-1, index=verts2d_mesh[:, :, :, 0][..., None, :].long())
        #sim_pxl2d_x =
        #sim_pxl2d_y = verts2d_mesh.reshape(-1, V, 2)[:, :, 1].long()

        #sim_pxl2d[]
        #sim_pxl2d_sampled = sample_pxl2d_pts(sim_pxl2d.reshape(-1, 1, H, W), pxl2d=verts2d_mesh.reshape(-1, V, 2)).reshape(B, T, V)
        #sim_pxl2d_sampled[:, :] = sim_pxl
        sim_pxl = sim_pxl2d

        if return_sim_pxl:
            return sim, sim_pxl
        else:
            return sim

    @staticmethod
    def batched_index_select(input, dim, index):
        for ii in range(1, len(input.shape)):
            if ii != dim:
                index = index.unsqueeze(ii)
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.expand(expanse)
        return torch.gather(input, dim, index)

    def get_sim_feats2d_net_and_rendered(self, feats2d_net, feats2d_rendered, return_sim_pxl=False, feats2d_net_mask=None, only_use_rendered_inliers=False, allow_clutter=True, use_sigmoid=False):
        """

        Args:
            feats2d_net (torch.Tensor): BxCxHxW
            feats2d_rendered (torch.Tensor): Bx(T)xCxHxW
            return_sim_pxl (bool): Indicates whether pixelwise similarity should be returned or not.
            feats2d_net_mask (torch.Tensor): Bx1xHxW

        Returns:
            sim (torch.Tensor): BxT, or Bx1 if rendered features is 4-dimensional.
            sim_pxl (torch.Tensor, optional): BxTxHxW, or Bx1xHxW if rendered features is 4-dimensional.
        """

        sim_clutter = self.calc_sim('bchw,nc->bnhw', feats2d_net, self.clutter_feats.detach()).max(dim=1, keepdim=True)[
            0]

        # sim_clutter = torch.einsum('bchw,nc->bnhw', feats2d_net, self.clutter_feats.detach()).mean(dim=1, keepdim=True)

        if feats2d_rendered.dim() == 5:
            feats2d_rendered_clutter_mask = feats2d_rendered.norm(dim=2) == 0.
            sim_texture_multiple_cams = self.calc_sim('bchw,bvchw->bvhw', feats2d_net, feats2d_rendered)
        else:
            feats2d_rendered_clutter_mask = (feats2d_rendered.norm(dim=1) == 0.)[:, None]
            sim_texture_multiple_cams = self.calc_sim('bchw,bchw->bhw', feats2d_net, feats2d_rendered)[:, None]

        # note only for occlusions: either
        # given feats2d_net_mask:
        #   a) only use feats2d_net_mask inliers,
        #   b) use feats2d_net_mask inliers plus clutter for outliers
        # given non mask: use either
        #   a) only inliers of rendered mask with maximum clutter/texture
        #   b) inliers and outliers of rendered mask, both with maximum clutter/texture

        # depcrecated?
        #       sim_pxl = torch.max(sim_texture_multiple_cams, sim_clutter)
        sim_pxl = sim_texture_multiple_cams

        if feats2d_net_mask is not None:
            feats2d_net_mask_clutter_bin = (feats2d_net_mask < 0.5).expand(*sim_pxl.shape)
            sim_pxl[feats2d_net_mask_clutter_bin] = sim_clutter.expand(*sim_pxl.shape)[feats2d_net_mask_clutter_bin]
            sim_pxl[(~feats2d_net_mask_clutter_bin) * feats2d_rendered_clutter_mask] = sim_clutter.expand(*sim_pxl.shape)[(~feats2d_net_mask_clutter_bin) * feats2d_rendered_clutter_mask]
            if allow_clutter:
                sim_pxl[(~feats2d_net_mask_clutter_bin) * (~feats2d_rendered_clutter_mask)] = \
                    torch.max(sim_texture_multiple_cams[(~feats2d_net_mask_clutter_bin) * (~feats2d_rendered_clutter_mask)],
                              sim_clutter.expand(*sim_pxl.shape)[(~feats2d_net_mask_clutter_bin) * (~feats2d_rendered_clutter_mask)])
            else:
                sim_pxl[(~feats2d_net_mask_clutter_bin) * (~feats2d_rendered_clutter_mask)] = \
                    sim_texture_multiple_cams[(~feats2d_net_mask_clutter_bin) * (~feats2d_rendered_clutter_mask)]
        else:
            sim_pxl[feats2d_rendered_clutter_mask] = sim_clutter.expand(*sim_pxl.shape)[feats2d_rendered_clutter_mask]
            if allow_clutter:
                sim_pxl[~feats2d_rendered_clutter_mask] = torch.max(
                    sim_texture_multiple_cams[~feats2d_rendered_clutter_mask],
                    sim_clutter.expand(*sim_pxl.shape)[~feats2d_rendered_clutter_mask])
            else:
                sim_pxl[~feats2d_rendered_clutter_mask] = sim_texture_multiple_cams[~feats2d_rendered_clutter_mask]

        # depcrecated?
        # if feats2d_net_mask is None:
        #     sim = sim_pxl.flatten(2).mean(dim=-1)
        # else:
        #     sim_pxl *= feats2d_net_mask
        #     sim = sim_pxl.flatten(2).sum(dim=-1) / (feats2d_net_mask.flatten(2).sum(dim=-1) + 1e-10)

        # use sigmoid
        # sim_pxl = torch.nn.functional.softmax(sim_pxl, dim=1)

        if use_sigmoid:
            sim_pxl = torch.sigmoid(sim_pxl)

        # only use rendered map inliers
        if only_use_rendered_inliers:
            sim = (sim_pxl * (~feats2d_rendered_clutter_mask)).sum(dim=(2, 3)) / (~feats2d_rendered_clutter_mask).sum(dim=(2, 3)).clamp(min=1e-10)
        # use also outliers
        else:
            sim = sim_pxl.flatten(2).mean(dim=-1)

        if return_sim_pxl:
            return sim, sim_pxl
        else:
            return sim

    def get_samples(self, config_sample: DictConfig, cam_intr4x4: torch.Tensor, cam_tform4x4_obj: torch.Tensor,
                    feats2d_net: torch.Tensor, categories_ids: torch.LongTensor, feats2d_net_mask: torch.Tensor=None, multiview=False):

        if multiview:
            b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4 = self.get_samples(config_sample=config_sample,
                                                                                       cam_intr4x4=cam_intr4x4[:1],
                                                                                       cam_tform4x4_obj=cam_tform4x4_obj[
                                                                                                        :1],
                                                                                       feats2d_net=feats2d_net[:1],
                                                                                       categories_ids=categories_ids[
                                                                                                      :1],
                                                                                       feats2d_net_mask=feats2d_net_mask[
                                                                                                        :1])
            scale = b_cams_multiview_tform4x4_obj[..., 2, 3] / cam_tform4x4_obj[:, None, 2, 3]
            cam_tform4x4_obj_scaled = cam_tform4x4_obj[:, None].clone().expand(cam_tform4x4_obj.shape[0], *b_cams_multiview_tform4x4_obj.shape[1:]).clone()

            # use rot. scaling
            # cam_tform4x4_obj_scaled[:, :, :3, :3] = cam_tform4x4_obj_scaled[:, :, :3, :3] * scale[:, :, None, None]
            # use transl. scaling
            #cam_tform4x4_obj_scaled[:, :, :3, 3] = cam_tform4x4_obj_scaled[:, :, :3, 3] * scale[:, :, None]
            # use depth scaling
            cam_tform4x4_obj_scaled[:, :, 2, 3] = cam_tform4x4_obj_scaled[:, :, 2, 3] * scale[:, :]

            objs_multiview_tform4x4_cuboid_front = tform4x4_broadcast(inv_tform4x4(cam_tform4x4_obj_scaled[:1]),
                                                                      b_cams_multiview_tform4x4_obj)
            b_cams_multiview_tform4x4_obj = tform4x4_broadcast(cam_tform4x4_obj_scaled,
                                                               objs_multiview_tform4x4_cuboid_front)
            b_cams_multiview_intr4x4 = b_cams_multiview_intr4x4.expand(*b_cams_multiview_tform4x4_obj.shape)

        else:

            B = len(feats2d_net)
            if config_sample.method == 'uniform':
                azim = torch.linspace(start=eval(config_sample.uniform.azim.min), end=eval(config_sample.uniform.azim.max), steps=config_sample.uniform.azim.steps).to(
                    device=self.device)  # 12
                elev = torch.linspace(start=eval(config_sample.uniform.elev.min), end=eval(config_sample.uniform.elev.max), steps=config_sample.uniform.elev.steps).to(
                    device=self.device)  # start=-torch.pi / 6, end=torch.pi / 3, steps=4
                theta = torch.linspace(start=eval(config_sample.uniform.theta.min), end=eval(config_sample.uniform.theta.max),
                                       steps=config_sample.uniform.theta.steps).to(
                    device=self.device)  # -torch.pi / 6, end=torch.pi / 6, steps=3

                #dist = torch.linspace(start=eval(config_sample.uniform.dist.min), end=eval(config_sample.uniform.dist.max), steps=config_sample.uniform.dist.steps).to(
                #    device=self.device)
                dist = torch.linspace(start=1., end=1., steps=1).to(device=self.device)

                azim_shape = azim.shape
                elev_shape = elev.shape
                theta_shape = theta.shape
                dist_shape = dist.shape
                in_shape = azim_shape + elev_shape + theta_shape + dist_shape
                azim = azim[:, None, None, None].expand(in_shape).reshape(-1)
                elev = elev[None, :, None, None].expand(in_shape).reshape(-1)
                theta = theta[None, None, :, None].expand(in_shape).reshape(-1)
                dist = dist[None, None, None, :].expand(in_shape).reshape(-1)
                cams_multiview_tform4x4_cuboid = transf4x4_from_spherical(azim=azim, elev=elev, theta=theta, dist=dist)

                C = len(cams_multiview_tform4x4_cuboid)

                b_cams_multiview_tform4x4_obj = cams_multiview_tform4x4_cuboid[None,].repeat(B, 1, 1, 1)

                # assumption 1: distance translation to object is known
                #b_cams_multiview_tform4x4_obj[:, :, 2, 3] = cam_tform4x4_obj[:, None].repeat(1, C, 1, 1)[:, :, 2, 3]
                # logger.info(f'dist {batch.cam_tform4x4_obj[:, 2, 3]}')
                # assumption 2: translation to object is known
                b_cams_multiview_tform4x4_obj[:, :, :3, 3] = cam_tform4x4_obj[:, None].repeat(1, C, 1, 1)[:, :, :3, 3]

                b_cams_multiview_intr4x4 = cam_intr4x4[:, None].repeat(1, C, 1, 1)

            elif config_sample.method == 'epnp3d2d':

                nearest_verts3d, nearest_verts2d, prob_well_corresp = self.get_nearest_corresp2d3d(feats2d_net=feats2d_net,
                                                                                                   meshes_ids=categories_ids,
                                                                                                   feats2d_net_mask=feats2d_net_mask)
                H, W = feats2d_net.shape[2:]
                prob_well_corresp = prob_well_corresp.flatten(1)
                if (prob_well_corresp.sum(dim=-1) == 0).any():
                    logger.warning(f"No texture similarity is larger than the clutter similarity")
                prob_well_corresp[prob_well_corresp.sum(dim=-1) == 0] = 1.

                K = config_sample.epnp3d2d.count_cams
                N = config_sample.epnp3d2d.count_pts

                masks_in_ids = torch.multinomial(prob_well_corresp, num_samples=K * N).reshape(-1, K, N)

                masks_in = torch.zeros(size=(B, K, H * W),
                                       device=self.device, dtype=torch.bool)
                for b in range(B):
                    for k in range(K):
                        masks_in[b, k, masks_in_ids[b, k]] = True
                masks_in = masks_in.reshape(B, K, H, W)
                b_cams_multiview_tform4x4_obj = batchwise_fit_se3_to_corresp_3d_2d_and_masks(masks_in=masks_in,
                                                                                             pts1=nearest_verts3d,
                                                                                             pxl2=nearest_verts2d,
                                                                                             proj_mat=cam_intr4x4[
                                                                                                      :, :2,
                                                                                                      :3] / self.down_sample_rate,
                                                                                             method="cpu-epnp")
                b_cams_multiview_intr4x4 = cam_intr4x4[:, None].repeat(1, K, 1, 1)
                b_cams_multiview_tform4x4_obj[b_cams_multiview_tform4x4_obj.flatten(2).isinf().any(dim=2), :,
                :] = torch.eye(4, device=b_cams_multiview_tform4x4_obj.device)
                b_cams_multiview_tform4x4_obj[(b_cams_multiview_tform4x4_obj[:, :, 3, :3] != 0.).any(dim=-1), :,
                :] = torch.eye(4, device=b_cams_multiview_tform4x4_obj.device)

            else:
                raise NotImplementedError

        if config_sample.depth_from_box:
            from od3d.cv.geometry.fit.depth_from_mesh_and_box import depth_from_mesh_and_box
            b_cams_multiview_tform4x4_obj[..., 2, 3] = depth_from_mesh_and_box(
                b_cams_multiview_intr4x4=cam_intr4x4, b_cams_multiview_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                meshes=self.meshes, labels=categories_ids, mask=feats2d_net_mask, downsample_rate=self.down_sample_rate,
                multiview=multiview)
        return b_cams_multiview_tform4x4_obj, b_cams_multiview_intr4x4


    """
    
            if self.config.train.visualize.verts_ncds_in_rgb:
                for b in range(len(batch)):
                    if batch.name_unique[b] in visual_names_unique:
                        verts_ncds_in_rgb = blend_rgb(batch.rgb[b], (
                            self.meshes.render_feats(cams_tform4x4_obj=batch.cam_tform4x4_obj[b:b + 1],
                                                     cams_intr4x4=batch.cam_intr4x4[b:b + 1],
                                                     imgs_sizes=batch.size, meshes_ids=batch.category_id[b:b + 1],
                                                     modality=MESH_RENDER_MODALITIES.VERTS_NCDS)[0]).to(
                            dtype=batch.rgb.dtype))
                        from od3d.cv.geometry.transform import proj3d2d_origin
                        verts_ncds_in_rgb = draw_pixels(verts_ncds_in_rgb, pxls=proj3d2d_origin(
                            torch.bmm(batch.cam_intr4x4, batch.cam_tform4x4_obj)[b:b + 1]), colors=[1., 0., 0.])
                        verts_ncds_in_rgb = draw_pixels(verts_ncds_in_rgb,
                                                        pxls=proj3d2d_origin(batch.cam_proj4x4_obj[b:b + 1]),
                                                        colors=[1., 0., 0.])
                        img = draw_pixels(verts_ncds_in_rgb, self.down_sample_rate * vts2d[b, mask_vts2d_vsbl[b]],
                                          colors=self.meshes.get_verts_ncds_with_mesh_id(batch.category_id[b])[
                                              mask_vts2d_vsbl[b]])
                        results_train['verts_ncds_in_rgb'] = image_as_wandb_image(img,
                                                                                  caption=f'Frame Name {batch.name_unique[b]}')
                        if self.config.train.visualize.live:
                            show_img(img)
    
            if self.config.train.visualize.net_feats_nearest_verts:
                for b in range(len(batch)):
                    if batch.name_unique[b] in visual_names_unique:
                        clutter_sim, clutter_sim_ids = self.calc_sim('bcn,vc->bnv', net_feats2d[b:b + 1].flatten(-2),
                                                                    self.clutter_feats).max(dim=-1)
                        net_mesh_nearest_feats_sim, net_mesh_nearest_feats_ids = self.calc_sim('bcn,vc->bnv',
                                                                                              net_feats2d[
                                                                                              b:b + 1].flatten(-2),
                                                                                              self.meshes.get_feats_with_mesh_id(
                                                                                                  batch.category_id[
                                                                                                      b])).max(
                            dim=-1)
                        net_mesh_nearest_feats_verts_ncds = self.meshes.get_verts_ncds_with_mesh_id(batch.category_id[b])[
                            net_mesh_nearest_feats_ids]
                        net_mesh_nearest_feats_verts_ncds[clutter_sim > net_mesh_nearest_feats_sim] = 0.
                        net_mesh_nearest_feats_verts_ncds = net_mesh_nearest_feats_verts_ncds.reshape(-1,
                                                                                                      *net_feats2d.shape[
                                                                                                       -2:],
                                                                                                      3).permute(0,
                                                                                                                 3,
                                                                                                                 1,
                                                                                                                 2)
                        img = blend_rgb(resize(batch.rgb[b], scale_factor=1. / self.down_sample_rate),
                                        net_mesh_nearest_feats_verts_ncds[0])
                        results_train['net_feats_nearest_verts'] = image_as_wandb_image(img,
                                                                                        caption=f'Frame Name {batch.name[b]}')
                        if self.config.train.visualize.live:
                            show_img(img)
    """

    # inference()

    # train()
    # 1. net_feats2d = get_net_feats2d()
    # 1. net_feats2d = get_net_feats2d()
    # 2. corresp2d3d = get_corresp2d3d()
    #   -> visualize NET_FEATS_NEAREST_VERTS
    # 2. samples = get_samples(corresp2d3d, method)
    #   -> visualize SAMPLES
    # 3. sample = select_sample()
    # 4. sample = pose_iterative(sample)
    #   -> visualize SIM_PXL, VERTS_NCDS_IN_RGB

    def visualize_test(self):
        pass