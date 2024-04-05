import logging
logger = logging.getLogger(__name__)
from od3d.benchmark.results import OD3D_Results
from omegaconf import DictConfig
import torchvision
from od3d.methods.method import OD3D_Dataset
from typing import List
import torch
from od3d.cv.geometry.transform import rot3x3, se3_exp_map, se3_log_map, so3_log_map, so3_exp_map, transf4x4_from_rot3x3, rot3x3_to_rot6d, rot6d_to_rot3x3
from od3d.methods.method import OD3D_Method
from od3d.models.model import OD3D_Model #  backbones.backbone import OD3D_Backbone
from pathlib import Path
from torch import nn as nn
import time
from tqdm import tqdm
import torch.utils.data
import pytorch3d
import od3d.io
from typing import Dict
from od3d.cv.transforms.transform import OD3D_Transform
from od3d.cv.transforms.sequential import SequentialTransform

class Regression(OD3D_Method):
    def __init__(
        self,
        config: DictConfig,
        logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)

        self.device = 'cuda:0'

        self.net = OD3D_Model(config.model)

        self.transform_train = SequentialTransform([
            OD3D_Transform.subclasses[config.train.transform.class_name].create_from_config(config=config.train.transform),
            self.net.transform,
        ])
        self.transform_test = SequentialTransform([
            OD3D_Transform.subclasses[config.test.transform.class_name].create_from_config(config=config.test.transform),
            self.net.transform
        ])

        # self.net = torch.nn.DataParallel(self.net).cuda()
        self.net.cuda()
        self.net.eval()

        self.optim = od3d.io.get_obj_from_config(config=self.config.train.optimizer, params=list(self.net.parameters()))
        self.scheduler = od3d.io.get_obj_from_config(self.optim, config=self.config.train.scheduler)

    def rot_mat_to_repr(self, rot3x3):
        if self.net.out_dim == 3:
            return so3_log_map(rot3x3)
        elif self.net.out_dim == 6:
            return rot3x3_to_rot6d(rot3x3)
        else:
            msg = f'Unknown out_dim {self.net.out_dim}'
            raise Exception(msg)
    def rot_repr_to_mat(self, rot_repr):
        if self.net.out_dim == 3:
            return so3_exp_map(rot_repr)
        elif self.net.out_dim == 6:
            return rot6d_to_rot3x3(rot_repr)
        else:
            msg = f'Unknown out_dim {self.net.out_dim}'
            raise Exception(msg)

    def save_checkpoint(self, path_checkpoint: Path):
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path_checkpoint)

    def load_checkpoint(self, path_checkpoint):
        checkpoint = torch.load(path_checkpoint)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    @property
    def path_checkpoint(self):
        return self.logging_dir.joinpath('nemo.ckpt')


    def train(self, datasets_train: Dict[str, OD3D_Dataset], datasets_val: Dict[str, OD3D_Dataset]):
        score_metric_name = 'pose/acc_pi6'
        score_ckpt_val = 0.
        score_latest = 0.

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
                    results_val = self.test(dataset_val)
                    results_val.log_with_prefix(prefix=f'val/{dataset_val.name}')
                    if dataset_val_key == 'main':
                        score_latest = results_val[score_metric_name]

                if not self.config.train.early_stopping or score_latest > score_ckpt_val:
                    score_ckpt_val = score_latest
                    self.save_checkpoint(path_checkpoint=self.path_checkpoint)

            results_epoch = self.train_epoch(dataset=dataset_train_sub)
            results_epoch.log_with_prefix('train')
        self.load_checkpoint(path_checkpoint=self.path_checkpoint)


    def test(self, dataset: OD3D_Dataset, config_inference: DictConfig = None, pose_iterative_refine=True):
        logger.info(f'test dataset {dataset.name}')
        if config_inference is None:
            config_inference = self.config.inference
        self.net.eval()
        dataset.transform = self.transform_test

        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.test.dataloader.batch_size,
                                                 shuffle=False,
                                                 collate_fn=dataset.collate_fn,
                                                 num_workers=self.config.test.dataloader.num_workers,
                                                 pin_memory=self.config.test.dataloader.pin_memory)

        logger.info(f"Dataset contains {len(dataset)} frames.")

        results_epoch = OD3D_Results()
        for i, batch in enumerate(iter(dataloader)):
            batch.to(device=self.device)

            results_batch = self.inference_batch(batch=batch)
            results_epoch += results_batch

        count_pred_frames = len(results_epoch['item_id'])
        logger.info(f'Predicted {count_pred_frames} frames.')

        #results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset,
        #                                         rank_metric_name='rot_diff_rad',
        #                                         config_visualize=self.config.test.visualize)
        results_epoch = results_epoch.mean()
        #results_epoch += results_visual
        return results_epoch


    def train_epoch(self, dataset: OD3D_Dataset) -> OD3D_Results:
        self.net.train()
        dataset.transform = self.transform_train
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=self.config.train.dataloader.batch_size,
                                                       shuffle=True,
                                                       collate_fn=dataset.collate_fn,
                                                       num_workers=self.config.train.dataloader.num_workers,
                                                       pin_memory=self.config.train.dataloader.pin_memory)

        results_epoch = OD3D_Results()
        accumulate_steps = 0
        for i, batch in tqdm(enumerate(iter(dataloader_train))):
            results_batch: OD3D_Results = self.train_batch(batch=batch)
            results_batch.log_with_prefix('train')
            accumulate_steps += 1
            if accumulate_steps % self.config.train.batch_accumulate_to_next_step == 0:
                self.optim.step()
                self.optim.zero_grad()

            results_epoch += results_batch

        self.scheduler.step()
        self.optim.zero_grad()

        #results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset,
        #                                         rank_metric_name='sim',
        #                                         config_visualize=self.config.train.visualize)
        results_epoch = results_epoch.mean()
        #results_epoch += results_visual
        return results_epoch

    def train_batch(self, batch) -> OD3D_Results:
        results_batch = OD3D_Results()
        batch.to(device=self.device)

        pred = self.net(batch.rgb)
        loss = (pred - self.rot_mat_to_repr(batch.cam_tform4x4_obj[:, :3, :3])).norm(dim=-1).mean()
        loss.backward()
        logger.info(f'loss {loss.item()}')

        results_batch['loss'] = loss[None,]

        results_batch['item_id'] = batch.item_id
        results_batch['name_unique'] = batch.name_unique

        return results_batch

    def inference_batch(self, batch) -> OD3D_Results:
        results = OD3D_Results()
        B = len(batch)

        time_loaded = time.time()
        with torch.no_grad():
            pred = self.net(batch.rgb)
            cam_rot3x3_obj = self.rot_repr_to_mat(pred)

        results['time_pose'] = torch.Tensor([time.time() - time_loaded,])

        diff_rot3x3 = rot3x3(batch.cam_tform4x4_obj[:, :3, :3].permute(0, 2, 1), cam_rot3x3_obj[:, :3, :3])

        try:
            diff_so3_log = so3_log_map(diff_rot3x3)
            diff_rot_angle_rad = torch.norm(diff_so3_log, dim=-1)
        except ValueError:
            logger.warning(
                f'Cannot calculate deviation in rotation angle due to rot3x3 trace being too small, setting deviation to 0.')
            diff_rot_angle_rad = 0.
        results['rot_diff_rad'] = diff_rot_angle_rad
        #results['label_gt'] = batch.label
        #results['label_pred'] = pred_class_ids
        #results['sim'] = sim
        results['cam_tform4x4_obj'] = transf4x4_from_rot3x3(cam_rot3x3_obj)
        results['item_id'] = batch.item_id
        results['name_unique'] = batch.name_unique

        return results