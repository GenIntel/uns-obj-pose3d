import time
from typing import List
from od3d.methods.method import OD3D_Method
from od3d.datasets.dataset import OD3D_Dataset, OD3D_Frame, OD3D_Frames
from od3d.datasets.co3d.dataset import CO3D
from omegaconf import DictConfig
import pytorch3d.transforms
from od3d.cv.geometry.transform import se3_log_map

from torch.utils.data import RandomSampler
import logging
logger = logging.getLogger(__name__)
import torch
import numpy as np
import wandb
import math
from od3d.cv.visual.draw import draw_pixels
from od3d.cv.geometry.transform import se3_exp_map
from od3d.cv.visual.show import imgs_to_img
from od3d.cv.geometry.mesh import Meshes
from pathlib import Path
from od3d.cv.geometry.transform import transf4x4_from_spherical, tform4x4_broadcast, tform4x4, rot3x3

from od3d.cv.geometry.transform import inv_tform4x4
from od3d.methods.nemo import NeMo
from od3d.benchmark.results import OD3D_Results
from dataclasses import dataclass
from typing import Dict
import random
from od3d.cv.visual.resize import resize

from tqdm import tqdm

from od3d.cv.visual.show import show_img
import torchvision
from od3d.cv.visual.blend import blend_rgb
from od3d.cv.visual.sample import sample_pxl2d_pts
from tqdm import tqdm
from od3d.cv.geometry.mesh import MESH_RENDER_MODALITIES

from od3d.datasets.meta import OD3D_Meta


@dataclass
class SequencePseudoLabel():
    obj_tform4x4_cuboid_front: torch.Tensor
    sim: float

from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    def __init__(self, datasets: List[OD3D_Dataset], datasets_mix: List[float], dataset_len: int, datasets_pseudo_labels: List):
        self.datasets = datasets
        self.modalities = datasets[0].modalities
        self.datasets_lens = torch.Tensor([len(dataset) for dataset in self.datasets])
        self.datasets_mix = datasets_mix
        self.dataset_len = dataset_len
        self.item_id_to_dataset_id = torch.multinomial(torch.Tensor(datasets_mix), num_samples=dataset_len, replacement=True)
        self.item_id_to_dataset_item_id = (torch.rand(dataset_len) * self.datasets_lens[self.item_id_to_dataset_id]).to(int)
        self.datasets_pseudo_labels = datasets_pseudo_labels

        # self.train_dict_category_sequences_pseudo_labels[batch.category[b]][batch.sequence_name[b]].obj_tform4x4_cuboid_front.to(device=batch.cam_tform4x4_obj.device))
    def __len__(self):
        return self.dataset_len  # length of the data
    def __getitem__(self, idx):
        dataset_id = self.item_id_to_dataset_id[idx]
        item_id = self.item_id_to_dataset_item_id[idx]

        frame: OD3D_Frame = self.datasets[dataset_id][item_id]
        if self.datasets_pseudo_labels[dataset_id] is not None:
            device = frame.cam_tform4x4_obj.device
            frame.cam_tform4x4_obj = tform4x4(frame.cam_tform4x4_obj, self.datasets_pseudo_labels[dataset_id][frame.category][frame.sequence.name].obj_tform4x4_cuboid_front.to(device=device))

        frame.item_id = idx
        return frame

    def collate_fn(self, frames: List[OD3D_Frame], device='cpu', dtype=torch.float32):
        frames = OD3D_Frames.get_frames_from_list(frames, modalities=self.modalities, dtype=dtype, device=device)
        return frames

    def get_subset_with_item_ids(self, item_ids):
        dataset = SiameseDataset(datasets=self.datasets, datasets_mix=self.datasets_mix, dataset_len=len(item_ids), datasets_pseudo_labels=self.datasets_pseudo_labels)
        dataset.item_id_to_dataset_id = self.item_id_to_dataset_id[item_ids]
        dataset.item_id_to_dataset_item_id = self.item_id_to_dataset_item_id[item_ids]
        return dataset

class NeMo_Incremental(NeMo):
    def __init__(
        self,
        config: DictConfig,
        logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)

    def update_pseudo_labels(self, dataset_train: CO3D):
        self.net.eval()
        self.meshes.feats.requires_grad = False

        self.train_dict_category_sequences_pseudo_labeled = []
        train_dict_category_sequences_pseudo_labels: Dict[str, Dict[str, SequencePseudoLabel]] = {}
        train_dict_category_sequences_pseudo_labeled: Dict[str, List[str]] = {}
        train_dict_category_sequences_unlabeled_unrolled = OD3D_Meta.unroll_nested_metas(self.train_dict_category_sequences_unlabeled)
        if self.config.train.incremental.pseudo_labels_update.count_new_labels_proposed >= len(train_dict_category_sequences_unlabeled_unrolled):
            train_dict_category_sequences_pseudo_labeled_proposed = OD3D_Meta.rollup_flattened_frames(train_dict_category_sequences_unlabeled_unrolled)
        else:
            train_dict_category_sequences_pseudo_labeled_proposed = OD3D_Meta.rollup_flattened_frames(random.sample(train_dict_category_sequences_unlabeled_unrolled,
                                                                                                                    k=self.config.train.incremental.pseudo_labels_update.count_new_labels_proposed))

        dataset_update = dataset_train.get_subset_by_sequences(dict_category_sequences=train_dict_category_sequences_pseudo_labeled_proposed,
                                                               frames_count_max_per_sequence=self.config.train.incremental.pseudo_labels_update.multiview_count)
        dataset_update.transform = self.transform_test

        dataloader = torch.utils.data.DataLoader(dataset=dataset_update, batch_size=self.config.train.incremental.pseudo_labels_update.multiview_count, #self.config.test.dataloader.batch_size,
                                                 shuffle=False,
                                                 collate_fn=dataset_update.collate_fn,
                                                 num_workers=self.config.test.dataloader.num_workers,
                                                 pin_memory=self.config.test.dataloader.pin_memory)

        results = OD3D_Results()
        for i, batch in tqdm(enumerate(iter(dataloader))):

            batch.to(device=self.device)

            if self.config.train.incremental.pseudo_labels_update.use_ground_truth:
                if batch.category[0] not in train_dict_category_sequences_pseudo_labels.keys():
                    train_dict_category_sequences_pseudo_labels[batch.category[0]] = {}
                if batch.category[0] not in train_dict_category_sequences_pseudo_labeled.keys():
                    train_dict_category_sequences_pseudo_labeled[batch.category[0]] = []
                obj_tform4x4_cuboid_front = torch.eye(4).to(device=self.device)
                sim = 1.
                train_dict_category_sequences_pseudo_labels[batch.category[0]][
                    batch.sequence_name[0]] = SequencePseudoLabel(obj_tform4x4_cuboid_front=obj_tform4x4_cuboid_front,
                                                                  sim=sim)
                train_dict_category_sequences_pseudo_labeled[batch.category[0]].append(batch.sequence_name[0])
            else:
                results_batch = self.inference_batch_multiview(batch=batch)
                results += results_batch
                sim = results_batch["sim"].mean()
                obj_tform4x4_cuboid_front = results_batch["obj_tform4x4_cuboid_front"]
                if sim > self.config.train.incremental.pseudo_labels_update.sim_threshold:
                    if batch.category[0] not in train_dict_category_sequences_pseudo_labels.keys():
                        train_dict_category_sequences_pseudo_labels[batch.category[0]] = {}
                    if batch.category[0] not in train_dict_category_sequences_pseudo_labeled.keys():
                        train_dict_category_sequences_pseudo_labeled[batch.category[0]] = []
                    train_dict_category_sequences_pseudo_labels[batch.category[0]][batch.sequence_name[0]] = SequencePseudoLabel(obj_tform4x4_cuboid_front=obj_tform4x4_cuboid_front.detach().cpu(), sim=sim)
                    train_dict_category_sequences_pseudo_labeled[batch.category[0]].append(batch.sequence_name[0])

        train_dict_category_sequences_pseudo_labeled = OD3D_Meta.unroll_nested_metas(train_dict_category_sequences_pseudo_labeled)

        logger.info(f'labeled {len(train_dict_category_sequences_pseudo_labeled)} sequences.')
        train_dict_category_sequences_pseudo_labeled = OD3D_Meta.rollup_flattened_frames(sorted(train_dict_category_sequences_pseudo_labeled,
                                                                                                key=lambda cat_seq: train_dict_category_sequences_pseudo_labels[cat_seq.split('/')[0]][cat_seq.split('/')[1]].sim, reverse=True)[:self.config.train.incremental.pseudo_labels_update.count_new_labels_selected_max])

        self.train_dict_category_sequences_pseudo_labeled = train_dict_category_sequences_pseudo_labeled
        self.train_dict_category_sequences_pseudo_labels = train_dict_category_sequences_pseudo_labels

        results_visual = self.get_results_visual(results_epoch=results, dataset=dataset_update,
                                                 config_visualize=self.config.test.visualize, filter_name_unique=False)
        results = results.mean()
        results += results_visual

        results['count'] = len(OD3D_Meta.unroll_nested_metas(self.train_dict_category_sequences_pseudo_labeled))
        logger.info(f'using {results["count"]} sequences.')

        results.log_with_prefix(prefix=f'pseudo_labels/{dataset_update.name}')


    def train(self, datasets_train: Dict[str, CO3D], datasets_val: Dict[str, OD3D_Dataset]):
        score_metric_name = 'pose/acc_pi6'
        score_ckpt_val = 0.
        score_latest = 0.
        self.save_checkpoint(path_checkpoint=self.fpath_checkpoint)

        if 'main' in datasets_val.keys():
            dataset_labeled = datasets_train['labeled']
        else:
            dataset_labeled, dataset_val_sub = datasets_train['labeled'].get_split(fraction1=1.-self.config.train.val_fraction,
                                                                                     fraction2=self.config.train.val_fraction,
                                                                                     split=self.config.train.split)
            datasets_val['main'] = dataset_val_sub

        dataset_unlabeled = datasets_train['unlabeled']
        dataset_labeled.transform = self.transform_train
        dataset_unlabeled.transform = self.transform_train

        #self.train_dict_category_sequences_labeled: Dict[str, List[str]] = {}
        self.train_dict_category_sequences_unlabeled: Dict[str, List[str]] = {}

        self.train_dict_category_sequences_pseudo_labeled: Dict[str, List[str]] = {}
        self.train_dict_category_sequences_pseudo_labels: Dict[str, Dict[str, SequencePseudoLabel]] = {}
        #for category in datasets_train['labeled'].dict_nested_frames.keys():
        #    self.train_dict_category_sequences_labeled[category] = list(dataset_train_sub.dict_nested_frames[category].keys())

        for category in datasets_train['unlabeled'].dict_nested_frames.keys():
            self.train_dict_category_sequences_unlabeled[category] = list(datasets_train['unlabeled'].dict_nested_frames[category].keys())
            self.train_dict_category_sequences_pseudo_labels[category] = {}
            self.train_dict_category_sequences_pseudo_labeled[category] = []


        for epoch in range(self.config.train.epochs):
            # list(datasets_train['labeled'].dict_nested_frames['car'].keys())
            if self.config.train.val and self.config.train.epochs_to_next_test > 0 and epoch % self.config.train.epochs_to_next_test == 0:
                for dataset_val_key, dataset_val in datasets_val.items():
                    results_val = self.test(dataset_val)
                    results_val.log_with_prefix(prefix=f'val/{dataset_val.name}')
                    if dataset_val_key == 'main':
                        score_latest = results_val[score_metric_name]

                if not self.config.train.early_stopping or score_latest > score_ckpt_val:
                    score_ckpt_val = score_latest
                    self.save_checkpoint(path_checkpoint=self.fpath_checkpoint)

            if self.config.train.incremental.enabled and epoch % self.config.train.incremental.pseudo_labels_update.epochs_to_next_update == 0:
                self.update_pseudo_labels(dataset_train=dataset_unlabeled)

            self.train_dict_category_sequences_epoch = OD3D_Meta.rollup_flattened_frames(OD3D_Meta.unroll_nested_metas(self.train_dict_category_sequences_pseudo_labeled))

            dataset_pseudo_labeled = dataset_unlabeled.get_subset_by_sequences(dict_category_sequences=self.train_dict_category_sequences_epoch,
                                                                               frames_count_max_per_sequence=self.config.train.incremental.frames_count_max_per_sequence)
            decay_labeled = self.config.train.incremental.decay_labeled ** epoch
            train_datset_sub_sub = SiameseDataset(datasets=[dataset_labeled, dataset_pseudo_labeled], datasets_mix=[decay_labeled, 1.0-decay_labeled],
                                                  dataset_len=self.config.train.incremental.epoch_frames_count, datasets_pseudo_labels=[None, self.train_dict_category_sequences_pseudo_labels])
            results_epoch = self.train_epoch(dataset=train_datset_sub_sub)
            results_epoch.log_with_prefix('train')

        self.load_checkpoint(path_checkpoint=self.fpath_checkpoint)

    """
    def train_batch(self, batch) -> OD3D_Results:
        for b in range(len(batch)):
            if self.config.train.incremental.enabled:
                if batch.category[b] in self.train_dict_category_sequences_pseudo_labels and batch.sequence_name[b] in self.train_dict_category_sequences_pseudo_labels[batch.category[b]]:
                    batch.cam_tform4x4_obj[b] = tform4x4(batch.cam_tform4x4_obj[b],
                                                         self.train_dict_category_sequences_pseudo_labels[batch.category[b]][batch.sequence_name[b]].obj_tform4x4_cuboid_front.to(device=batch.cam_tform4x4_obj.device))
        return super().train_batch(batch=batch)
    """

    """
    def inference_batch_multiview(self, batch, return_samples_with_sim=True):
        results = OD3D_Results()
        B = len(batch)

        
        ##  these parameters are used in prev. version
        # batch.cam_tform4x4_obj[:, 2, 3] = 5. * 6. # 5. * 6.
        # batch.cam_tform4x4_obj[:, 0, 3] = 0.
        # batch.cam_tform4x4_obj[:, 1, 3] = 0.
        # batch.cam_intr4x4[:, 0, 0] = 3000.
        # batch.cam_intr4x4[:, 1, 1] = 3000.
        # batch.cam_intr4x4[:, 0, 2] = batch.size[1] / 2.
        # batch.cam_intr4x4[:, 1, 2] = batch.size[0] / 2.
        

        time_loaded = time.time()
        with torch.no_grad():
            feats2d_net = self.net(batch.rgb)
            feats2d_net_mask = resize(batch.mask_rgb, H_out=feats2d_net.shape[2], W_out=feats2d_net.shape[3])
            if self.config.inference.use_mask_object:
                feats2d_net_mask = feats2d_net_mask * 1. * resize(batch.mask, H_out=feats2d_net.shape[2],
                                                                  W_out=feats2d_net.shape[3])

            time_pred_net_feats2d = time.time()
            # logger.info(
            #    f"predicted net feats2d, took {(time_pred_net_feats2d - time_loaded):.3f}")
            results['time_feats2d'] = torch.Tensor([time_pred_net_feats2d - time_loaded,]) / B

            meshes_scores = []
            for mesh_id in range(len(self.meshes)):
                # logger.info(f'calc score for mesh {self.config.classes[mesh_id]}')
                bank_feats = torch.cat([self.meshes.get_feats_with_mesh_id(mesh_id), self.clutter_feats.detach()],
                                       dim=0)
                # inner_feats2d_net_bank_vts_max_vals = torch.sum(net_feats2d[:, None] * bank_feats[None, :, :, None, None], dim=2, keepdim=True).max(dim=1).values
                out_shape = feats2d_net.shape[:1] + torch.Size([1]) + feats2d_net.shape[2:]
                inner_feats2d_net_bank_vts_max_vals = torch.einsum('bchw,kc->bkhw', feats2d_net, bank_feats).max(dim=1,
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
            # logger.info(f"predicted class: {self.config.classes[int(pred_class_ids[0])]}, took {(time_pred_class - time_pred_net_feats2d):.3f}")

            results['time_class'] = torch.Tensor([time_pred_class - time_pred_net_feats2d,]) / B

            cams_multiview_tform4x4_obj, cams_multiview_intr4x4 = self.get_samples(config_sample=self.config.inference.sample,
                                                                                   cam_intr4x4=batch.cam_intr4x4[:1],
                                                                                   cam_tform4x4_obj=batch.cam_tform4x4_obj[:1],
                                                                                   feats2d_net=feats2d_net[:1],
                                                                                   categories_ids=pred_class_ids[:1],
                                                                                   feats2d_net_mask=feats2d_net_mask[:1])
            # multiview adaption
            objs_multiview_tform4x4_cuboid_front = tform4x4_broadcast(inv_tform4x4(batch.cam_tform4x4_obj[:1])[:, None], cams_multiview_tform4x4_obj)
            b_cams_multiview_tform4x4_obj = tform4x4_broadcast(batch.cam_tform4x4_obj[:, None], objs_multiview_tform4x4_cuboid_front)
            b_cams_multiview_intr4x4 = cams_multiview_intr4x4.expand(*b_cams_multiview_tform4x4_obj.shape)

            #  OPTION A: Use 2d gradient of rendered features
            sim = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                     cam_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                                                     cam_intr4x4=b_cams_multiview_intr4x4,
                                                     categories_ids=pred_class_ids,
                                                     broadcast_batch_and_cams=True,
                                                     feats2d_net_mask=feats2d_net_mask, pre_rendered=False)

            sim = sim.mean(dim=0, keepdim=True).expand(*sim.shape)

            if return_samples_with_sim:
                results['samples_cam_tform4x4_obj'] = b_cams_multiview_tform4x4_obj
                results['samples_cam_intr4x4'] = b_cams_multiview_intr4x4
                results['samples_sim'] = sim

            mesh_multiple_cams_loss = -sim


            mesh_cam_loss_min_val, mesh_cam_loss_min_id = mesh_multiple_cams_loss.min(dim=1)

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

            for epoch in range(self.config.inference.optimizer.epochs):
                # commenting this line means to enable translation optimization.
                obj_tform6_tmp.data[:, self.config.inference.refine.dims_detached] = 0.

                obj_tform4x4_cuboid_front = tform4x4(obj_tform4x4_cuboid_front.detach(), se3_exp_map(obj_tform6_tmp.detach()))
                obj_tform6_tmp.data[:, :] = 0.
                obj_tform4x4_cuboid_front = tform4x4(obj_tform4x4_cuboid_front.detach(), se3_exp_map(obj_tform6_tmp))

                sim, sim_pxl = self.get_sim_feats2d_net_with_cams(feats2d_net=feats2d_net,
                                                                  cam_tform4x4_obj=tform4x4_broadcast(batch.cam_tform4x4_obj, obj_tform4x4_cuboid_front),
                                                                  cam_intr4x4=batch.cam_intr4x4,
                                                                  categories_ids=pred_class_ids, return_sim_pxl=True,
                                                                  broadcast_batch_and_cams=False,
                                                                  feats2d_net_mask=feats2d_net_mask, pre_rendered=False)
                mesh_cam_loss = -sim

                if self.config.inference.live:
                    show_img(
                        blend_rgb(batch.rgb[0], (self.meshes.render_feats(cams_tform4x4_obj=tform4x4_broadcast(batch.cam_tform4x4_obj, obj_tform4x4_cuboid_front)[0:0 + 1],
                                                                          cams_intr4x4=batch.cam_intr4x4[0:0 + 1],
                                                                          imgs_sizes=batch.size,
                                                                          meshes_ids=pred_class_ids[0:0 + 1],
                                                                          modality=MESH_RENDER_MODALITIES.VERTS_NCDS)[
                            0]).to(dtype=batch.rgb.dtype)), duration=1)

                loss = mesh_cam_loss.mean()
                loss.backward()
                optim_inference.step()
                optim_inference.zero_grad()

            obj_tform4x4_cuboid_front = tform4x4(obj_tform4x4_cuboid_front.detach(), se3_exp_map(obj_tform6_tmp.detach()))

            results['time_pose_iterative'] = torch.Tensor([time.time() - time_before_pose_iterative,]) / B

        obj_tform4x4_cuboid_front = obj_tform4x4_cuboid_front.clone().detach()
        cam_tform4x4_obj = tform4x4_broadcast(batch.cam_tform4x4_obj, obj_tform4x4_cuboid_front)

        results['time_pose'] = torch.Tensor([time.time() - time_pred_class,]) / B

        diff_rot3x3 = rot3x3(batch.cam_tform4x4_obj[:, :3, :3].permute(0, 2, 1), cam_tform4x4_obj[:, :3, :3])

        try:
            diff_so3_log = pytorch3d.transforms.so3_log_map(diff_rot3x3.permute(0, 2, 1))
            diff_rot_angle_rad = torch.norm(diff_so3_log, dim=-1)
        except ValueError:
            logger.warning(
                f'Cannot calculate deviation in rotation angle due to rot3x3 trace being too small, setting deviation to 0.')
            diff_rot_angle_rad = 0.
        results['rot_diff_rad'] = diff_rot_angle_rad
        results['label_gt'] = batch.label
        results['label_pred'] = pred_class_ids
        results['sim'] = sim.mean(dim=0, keepdim=True).expand(*sim.shape)
        results['cam_tform4x4_obj'] = cam_tform4x4_obj
        results['obj_tform4x4_cuboid_front'] = obj_tform4x4_cuboid_front
        results['item_id'] = batch.item_id
        results['name_unique'] = batch.name_unique

        return results
    """





    """
    def train(self):
        if self.config.train.incremental.enabled:
            self.seq_obj_tform4x4_est_obj = {}
            self.seq_obj_tform4x4_est_obj_sim = {}
            self.seq_obj_tform4x4_est_obj_transl_consist = {}
            self.seq_obj_tform4x4_est_obj_rot_consist = {}
            self.seq_filtered = []
            self.seq_labeled = dataset.sequences_names[:self.config.train.sequences_tform4x4_labeled_count]

        if self.config.train.test and self.config.train.epochs_to_next_test > 0 and epoch % self.config.train.epochs_to_next_test == 0:
            pass

    def train_batch(self, batch):
        for b in range(len(batch)):
            if self.config.train.incremental.enabled:
                if batch.sequence_name[b] in self.seq_filtered:
                    batch.cam_tform4x4_obj[b] = tform4x4(batch.cam_tform4x4_obj[b],
                                                         self.seq_obj_tform4x4_est_obj[batch.sequence_name[b]])

    def get_dataset_sub(self, dataset):
        if self.config.train.epochs_to_next_forget_est_tforms4x4 > 0 and e % self.config.train.epochs_to_next_forget_est_tforms4x4 == 0:
            previous_sequences_names = self.seq_labeled + self.seq_filtered
            proposals_sequences_names = self.seq_filtered + list(
                set(dataset.sequences_names) - set(previous_sequences_names))[
                                                            :self.config.train.sequences_new_proposals_count]

            self.seq_obj_tform4x4_est_obj = {}
            self.seq_obj_tform4x4_est_obj_sim = {}
            self.seq_obj_tform4x4_est_obj_transl_consist = {}
            self.seq_obj_tform4x4_est_obj_rot_consist = {}
            self.seq_filtered = []

            for s, seq in enumerate(proposals_sequences_names):

                seq_dataset = dataset.get_subset_by_sequences([seq],
                                                              frames_count_max_per_sequence=self.config.train.sequences_tform4x4_estimated_frames_count)
                seq_dataset.transform = self.transform_test
                dataloader_train_seq = torch.utils.data.DataLoader(dataset=seq_dataset,
                                                                   batch_size=self.config.test.dataloader.batch_size,
                                                                   shuffle=False,
                                                                   collate_fn=dataset.collate_fn,
                                                                   num_workers=self.config.test.dataloader.num_workers,
                                                                   pin_memory=self.config.test.dataloader.pin_memory)
                logger.info(f'estimating obj_tform4x4_obj_est for {seq}')
                seq_obj_tform4x4_est_obj = []
                seq_obj_tform4x4_est_obj_sim = []
                for i, batch in enumerate(iter(dataloader_train_seq)):
                    batch.to(device=self.device)
                    cam_tform4x4_obj_est, est_sim, results_batch = self.inference_batch(batch,
                                                                                        config=self.config.inference,
                                                                                        visual_names_unique=[
                                                                                            batch.name_unique[j]
                                                                                            for j in
                                                                                            range(len(batch))])
                    seq_obj_tform4x4_est_obj.append(
                        tform4x4(inv_tform4x4(batch.cam_tform4x4_obj), cam_tform4x4_obj_est))
                    seq_obj_tform4x4_est_obj_sim.append(est_sim)

                seq_obj_tform4x4_est_obj_sim = torch.cat(seq_obj_tform4x4_est_obj_sim, dim=0)
                seq_obj_tform4x4_est_obj = torch.cat(seq_obj_tform4x4_est_obj, dim=0)

                seq_obj_tform4x4_est_obj = seq_obj_tform4x4_est_obj[
                                           :self.config.train.sequences_tform4x4_estimated_frames_count]
                seq_obj_tform4x4_est_obj_sim = seq_obj_tform4x4_est_obj_sim[
                                               :self.config.train.sequences_tform4x4_estimated_frames_count]

                seq_obj_tform6_est_obj = se3_log_map(seq_obj_tform4x4_est_obj)
                seq_obj_transl3_est_obj_dist_mat = (
                        seq_obj_tform6_est_obj[None, :, :3] - seq_obj_tform6_est_obj[:, None, :3]).norm(
                    dim=-1)
                seq_obj_rot3_est_obj_dist_mat = (
                        seq_obj_tform6_est_obj[None, :, 3:] - seq_obj_tform6_est_obj[:, None, 3:]).norm(
                    dim=-1)
                seq_obj_tform4x4_est_obj_transl_consist = seq_obj_transl3_est_obj_dist_mat.triu(
                    diagonal=1).sum() / torch.ones_like(seq_obj_transl3_est_obj_dist_mat).triu(diagonal=1).sum()
                seq_obj_tform4x4_est_obj_rot_consist = seq_obj_rot3_est_obj_dist_mat.triu(
                    diagonal=1).sum() / torch.ones_like(seq_obj_rot3_est_obj_dist_mat).triu(diagonal=1).sum()
                seq_obj_tform4x4_est_obj_sim = seq_obj_tform4x4_est_obj_sim.mean()
                seq_obj_tform4x4_est_obj = se3_exp_map(seq_obj_tform6_est_obj.mean(dim=0))

                self.seq_obj_tform4x4_est_obj[seq] = seq_obj_tform4x4_est_obj
                self.seq_obj_tform4x4_est_obj_sim[seq] = seq_obj_tform4x4_est_obj_sim
                self.seq_obj_tform4x4_est_obj_transl_consist[seq] = seq_obj_tform4x4_est_obj_transl_consist
                self.seq_obj_tform4x4_est_obj_rot_consist[seq] = seq_obj_tform4x4_est_obj_rot_consist

                if seq_obj_tform4x4_est_obj_sim > self.config.train.sequences_tform4x4_est_sim_min \
                        and seq_obj_tform4x4_est_obj_transl_consist < self.config.train.sequences_tform4x4_est_transl_consist_max \
                        and seq_obj_tform4x4_est_obj_rot_consist < self.config.train.sequences_tform4x4_est_rot_consist_max:

                    self.seq_filtered.append(seq)

                    if self.config.train.visualize.seq_added_tform:
                        for key, val in results_batch.items():
                            if type(val) == wandb.Image:
                                results_train['add_' + key] = val

            results_train['seq_obj_tform4x4_est_obj_sim'] = torch.stack(
                list(self.seq_obj_tform4x4_est_obj_sim.values())).mean()
            results_train['seq_obj_tform4x4_est_obj_transl_consist'] = torch.stack(
                list(self.seq_obj_tform4x4_est_obj_transl_consist.values())).mean()
            results_train['seq_obj_tform4x4_est_obj_rot_consist'] = torch.stack(
                list(self.seq_obj_tform4x4_est_obj_rot_consist.values())).mean()

            logger.info(
                f'seq_obj_tform4x4_est_obj_transl_consist {self.seq_obj_tform4x4_est_obj_transl_consist.values()}')
            # logger.info(f'estimating obj_tform4x4_obj_est_sims of {self.seq_obj_tform4x4_est_obj_sim}')

            sequences_filtered = list(self.seq_filtered)
            results_train["count_sequences"] = len(sequences_filtered + self.seq_labeled)
            dataset_sub = dataset.get_subset_by_sequences(sequences_filtered + self.seq_labeled)

            logger.info(f"Dataset contains {len(dataset_sub)} frames.")

            return dataset_sub

    """