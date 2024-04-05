import logging
logger = logging.getLogger(__name__)
from od3d.methods.method import OD3D_Method
from od3d.datasets.dataset import OD3D_Dataset
from od3d.benchmark.results import OD3D_Results
from omegaconf import DictConfig
import pytorch3d.transforms
import pandas as pd
import numpy as np
from torch.utils.data import RandomSampler
import torch
from od3d.cv.transforms.transform import OD3D_Transform
from od3d.cv.transforms.sequential import SequentialTransform
from typing import Dict
from tqdm import tqdm
from od3d.datasets.frames import OD3D_Frames

class Docker(OD3D_Method):
    def setup(self):
        pass

    def __init__(
            self,
            config: DictConfig,
            logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)

        self.device = 'cpu' #'cuda:0'

        # init Network
        # self.net = OD3D_Model(config.docker)

        self.transform_train = OD3D_Transform.subclasses[config.train.transform.class_name].create_from_config(config=config.train.transform)
        self.transform_test = OD3D_Transform.subclasses[config.test.transform.class_name].create_from_config(config=config.test.transform)

    def train(self, datasets_train: Dict[str, OD3D_Dataset], datasets_val: Dict[str, OD3D_Dataset]):
        pass


    def test(self, dataset: OD3D_Dataset, config_inference: DictConfig = None):
        logger.info(f'test dataset {dataset.name}')
        if config_inference is None:
            config_inference = self.config.inference
        dataset.transform = self.transform_test

        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.config.test.dataloader.batch_size,
                                                 shuffle=False,
                                                 collate_fn=dataset.collate_fn,
                                                 num_workers=self.config.test.dataloader.num_workers,
                                                 pin_memory=self.config.test.dataloader.pin_memory)

        logger.info(f"Dataset contains {len(dataset)} frames.")

        results_epoch = OD3D_Results()
        for i, batch in tqdm(enumerate(iter(dataloader))):
            batch.to(device=self.device)

            results_batch = self.inference_batch(batch=batch)
            results_epoch += results_batch

        count_pred_frames = len(results_epoch['item_id'])
        logger.info(f'Predicted {count_pred_frames} frames.')

        #results_visual = self.get_results_visual(results_epoch=results_epoch, dataset=dataset,
        #                                         config_visualize=self.config.test.visualize)
        results_epoch = results_epoch.mean()
        #results_epoch += results_visual
        return results_epoch

    def inference_batch(self, batch: OD3D_Frames):
        results = OD3D_Results()

        import io
        import pickle
        import torch

        import requests

        #resp = requests.post("http://localhost:5000/predict",
        #                             files={"file": open('<PATH/TO/.jpg/FILE>/cat.jpg', 'rb')})

        bytes_io = io.BytesIO()
        pickle.dump({'img': batch.rgb, 'cam_tform4x4_obj': batch.cam_tform4x4_obj}, bytes_io, pickle.HIGHEST_PROTOCOL)

        resp = requests.post("http://localhost:5000/predict",files={"file": bytes_io})

        """
        s1 = "foo"
        s1 = {"foo": torch.zeros(3,), "blub": torch.ones(4,)}
        bytes_io = io.BytesIO()
        pickle.dump(s1, bytes_io, pickle.HIGHEST_PROTOCOL)
        bytes_io.seek(0)
        s2 = pickle.load(bytes_io)
        """



        cam_tform4x4_obj = None

        from od3d.cv.geometry.transform import rot3x3

        diff_rot3x3 = rot3x3(batch.cam_tform4x4_obj[:, :3, :3].permute(0, 2, 1), cam_tform4x4_obj[:, :3, :3])

        try:
            diff_so3_log = pytorch3d.transforms.so3_log_map(diff_rot3x3.permute(0, 2, 1))
            diff_rot_angle_rad = torch.norm(diff_so3_log, dim=-1)
        except ValueError:
            logger.warning(
                f'Cannot calculate deviation in rotation angle due to rot3x3 trace being too small, setting deviation to 0.')
            diff_rot_angle_rad = torch.zeros_like(diff_rot3x3[:, 0, 0])
        results['rot_diff_rad'] = diff_rot_angle_rad
        #results['label_gt'] = batch.label
        #results['label_pred'] = pred_class_ids
        results['label_names'] = self.config.classes
        #results['sim'] = sim
        results['cam_tform4x4_obj'] = cam_tform4x4_obj
        results['item_id'] = batch.item_id
        results['name_unique'] = batch.name_unique

        B = len(batch)

        return results
