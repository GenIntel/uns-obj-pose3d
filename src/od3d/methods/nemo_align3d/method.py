import logging
logger = logging.getLogger(__name__)
import time
from typing import List
import od3d.io
from od3d.methods.method import OD3D_Method
from od3d.datasets.dataset import OD3D_Dataset
from od3d.benchmark.results import OD3D_Results
from od3d.datasets.co3d import CO3D

from omegaconf import DictConfig
import pytorch3d.transforms
import pandas as pd
import numpy as np
from torch.utils.data import RandomSampler
from od3d.cv.visual.show import show_scene
import math
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from od3d.cv.geometry.transform import se3_exp_map
from od3d.cv.visual.show import imgs_to_img
from od3d.cv.geometry.mesh import Meshes
from pathlib import Path
from od3d.cv.geometry.transform import transf4x4_from_spherical, tform4x4, rot3x3
from od3d.cv.visual.show import show_img
import torchvision
from od3d.cv.visual.blend import blend_rgb
from od3d.cv.visual.sample import sample_pxl2d_pts
from tqdm import tqdm
from od3d.cv.geometry.mesh import MESH_RENDER_MODALITIES
from od3d.cv.metric.pose import get_pose_diff_in_rad

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
from od3d.cv.optimization.ransac import ransac
from od3d.cv.geometry.fit.tform4x4 import fit_tform4x4, score_tform4x4_fit

from functools import partial
from pathlib import Path
from od3d.io import read_json
from od3d.cv.geometry.transform import inv_tform4x4, tform4x4
from od3d.datasets.co3d.enum import CAM_TFORM_OBJ_SOURCES, CUBOID_SOURCES
from od3d.datasets.co3d.enum import PCL_SOURCES
from od3d.cv.geometry.transform import transf3d_broadcast, transf3d


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

class NeMo_Align3D(OD3D_Method):
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
        #     self.transform_train = torchvision.transforms.Compose([
        #         RandomCenterZoom3D(**config.train.transform.random_center_zoom3d),
        #         #RGB_Random(),
        #         self.net.transform,
        #     ])
        # else:
        #     self.transform_train = torchvision.transforms.Compose([
        #         RandomCenterZoom3D(**config.train.transform.random_center_zoom3d),
        #         self.net.transform,
        #     ])

        #self.transform_test = torchvision.transforms.Compose([
        #    CenterZoom3D(**config.test.transform),
        #    self.net.transform
        #])

        self.meshes = None
        self.sequences_unique_names = None
        self.meshes = None

        # init Meshes / Features
        self.total_params = sum(p.numel() for p in self.net.parameters())
        self.net.eval()
        self.net.cpu()

        self.down_sample_rate = self.net.downsample_rate

        self.dir_tmp = Path('/tmp').joinpath(self.__class__.__name__)
        self.dir_tmp.mkdir(parents=True, exist_ok=True)
    def train(self, datasets_train: Dict[str, CO3D], datasets_val: Dict[str, OD3D_Dataset]):
        score_metric_name = 'pose/acc_pi18'  # 'pose/acc_pi18' 'pose/acc_pi6'
        score_ckpt_val = 0.
        score_latest = 0.

        dataset_src: CO3D = datasets_train['src']
        dataset_ref: CO3D = datasets_train['labeled']
        from od3d.cv.geometry.mesh import Meshes

        categories = dataset_src.categories

        src_sequences = dataset_src.get_sequences()
        ref_sequences = dataset_ref.get_sequences()
        for sequence in src_sequences + ref_sequences:
            if self.config.preprocess.pcl.enabled:
                sequence.preprocess_pcl(override=self.config.preprocess.pcl.override)
            if self.config.preprocess.tform_obj.enabled:
                sequence.preprocess_tform_obj(override=self.config.preprocess.tform_obj.override)
            if self.config.preprocess.mesh.enabled:
                sequence.preprocess_mesh(override=self.config.preprocess.mesh.override)
            if self.config.preprocess.mesh_feats.enabled:
                sequence.preprocess_mesh_feats(override=self.config.preprocess.mesh_feats.override)

        if self.config.preprocess.mesh_feats_dist.enabled:
            for src_sequence in src_sequences:
                for ref_sequence in ref_sequences:
                    if src_sequence.category == ref_sequence.category:
                        src_sequence.preprocess_mesh_feats_dist(sequence=ref_sequence, override=self.config.preprocess.mesh_feats_dist.override)
                        ref_sequence.preprocess_mesh_feats_dist(sequence=src_sequence, override=self.config.preprocess.mesh_feats_dist.override)

        # tform4x4(inv_tform4x4(src_frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D)), src_frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.DROID_SLAM))
        logger.info('loading mesh feats...')
        #self.sequences_mesh_feats = [seq.feats for seq in self.sequences]
        src_sequences_unique_names = [seq.name_unique for seq in src_sequences]
        ref_sequences_unique_names = [seq.name_unique for seq in ref_sequences]
        #sequences_unique_names = [seq.name_unique for seq in sequences]

        src_map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in src_sequences_unique_names])
        ref_map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in ref_sequences_unique_names])

        categories_count = len(categories)

        src_instances_count_per_category = [(src_map_seq_to_cat == c).sum().item() for c in range(categories_count)]
        ref_instances_count_per_category = [(ref_map_seq_to_cat == c).sum().item() for c in range(categories_count)]

        logger.info('loading meshes...')
        src_meshes = Meshes.load_from_meshes([seq.read_mesh() for seq in src_sequences], device=self.device)
        ref_meshes = Meshes.load_from_meshes([seq.read_mesh() for seq in ref_sequences], device=self.device)

        src_instances_count = len(src_meshes)
        ref_instances_count = len(ref_meshes)

        dtype = src_meshes.verts.dtype

        src_sequences_mesh_ids_for_verts = src_meshes.get_mesh_ids_for_verts()
        ref_sequences_mesh_ids_for_verts = ref_meshes.get_mesh_ids_for_verts()

        results_diff_log_rot = {}
        all_pred_ref_pts_offset = {}
        all_pred_ref_tform_src = {}
        all_pred_pose_dist_geo = {}
        all_pred_pose_dist_appear = {}
        for cat_id, category in enumerate(categories):
            logger.info(f'category id {cat_id} name {category}')
            src_instance_ids = torch.LongTensor(list(range(src_instances_count)))
            ref_instance_ids = torch.LongTensor(list(range(ref_instances_count)))

            results_diff_log_rot[category] = torch.zeros(
                size=(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id])).to(
                device=self.device, dtype=dtype)
            all_pred_ref_tform_src[category] = torch.zeros(
                size=(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id], 4, 4)).to(
                device=self.device, dtype=dtype)
            all_pred_pose_dist_geo[category] = torch.zeros(
                size=(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id])).to(
                device=self.device, dtype=dtype)

            all_pred_ref_pts_offset[category] = torch.zeros(size=(src_instances_count_per_category[cat_id], sum(ref_meshes.verts_counts), 3)).to(
                device=self.device, dtype=dtype)

            all_pred_pose_dist_appear[category] = torch.zeros(
                size=(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id])).to(
                device=self.device, dtype=dtype)

        for i in range(self.config.global_optimization_steps):
            for cat_id, category in enumerate(categories):
                src_instance_ids = torch.LongTensor(list(range(src_instances_count)))
                ref_instance_ids = torch.LongTensor(list(range(ref_instances_count)))

                src_mesh_ids = src_instance_ids[src_map_seq_to_cat == cat_id]
                ref_mesh_ids = ref_instance_ids[ref_map_seq_to_cat == cat_id]

                for r, ref_mesh_id in enumerate(ref_mesh_ids):
                    for s, src_mesh_id in enumerate(src_mesh_ids):
                        # src_sequences[src_mesh_id].show(show_imgs=True)
                        src_vertices_mask = src_sequences_mesh_ids_for_verts == src_mesh_id
                        #src_vertices = torch.arange(src_vertices_count).to(device=self.device)[src_vertices_mask]
                        pts_src = src_meshes.verts[src_vertices_mask].clone()
                        if r > 0 and (self.config.use_only_first_reference or self.config.global_optimization_steps > 1):
                                pred_ref_tform_src = tform4x4(inv_tform4x4(all_pred_ref_tform_src[category][0, r]), all_pred_ref_tform_src[category][0, s])
                                all_pred_ref_tform_src[category][r, s] = pred_ref_tform_src
                                all_pred_pose_dist_geo[category][r, s] = all_pred_pose_dist_geo[category][0, r] + all_pred_pose_dist_geo[category][0, s]
                                all_pred_pose_dist_appear[category][r, s] = all_pred_pose_dist_appear[category][0, r] + all_pred_pose_dist_appear[category][0, s]
                                #logger.info(pred_ref_tform_src)
                        else:
                            if self.config.global_optimization_steps > 1 and i > 0:
                                #ref_vertices_mask = self.sequences_mesh_ids_for_verts == ref_mesh_id
                                #pts = self.meshes.verts.clone().detach()
                                #pts_ref = pts[ref_vertices_mask].clone()
                                # TODO: reference points from multiple point clouds have different scale and therefore problematic to fit with correspondences over multiple points
                                pts_ref = torch.cat([transf3d_broadcast(pts3d=ref_meshes.verts[ref_sequences_mesh_ids_for_verts == _ref_mesh_id].clone(), transf4x4=all_pred_ref_tform_src[category][0, _r]) if _ref_mesh_id != src_mesh_id else torch.zeros((0, 3), device=self.device) for _r, _ref_mesh_id in enumerate(ref_mesh_ids)], dim=0)

                                dist_src_ref = torch.cat([src_sequences[src_mesh_id].read_mesh_feats_dist(
                                    ref_sequences[_ref_mesh_id]).to(device=self.device, dtype=dtype) if _ref_mesh_id != src_mesh_id else torch.zeros((pts_src.shape[0], 0), device=self.device)  for _ref_mesh_id in ref_mesh_ids], dim=-1)

                                logger.info(f'category: {category}, pts-src: {pts_src.shape}, pts-ref: {pts_ref.shape}')

                                # division by two to normalize to 0. - 1.
                                dist_src_ref = dist_src_ref / 2.

                                # four points required, otherwise rotation yields an ambiguity. like planes without normals
                                ref_tform4x4_src = ransac(pts=pts_src, fit_func=partial(fit_tform4x4, pts_ref=pts_ref,
                                                                                        dist_app_ref=dist_src_ref),
                                                          score_func=partial(score_tform4x4_fit, pts_ref=pts_ref,
                                                                             dist_app_ref=dist_src_ref,
                                                                             dist_app_weight=self.config.dist_appear_weight,
                                                                             geo_cyclic_weight_temp=self.config.geo_cyclic_weight_temp,
                                                                             app_cyclic_weight_temp=self.config.app_cyclic_weight_temp,
                                                                             score_perc=self.config.ransac.score_perc),
                                                          fits_count=self.config.ransac.samples,
                                                          fit_pts_count=4)

                                _, pose_dist_geo, pose_dist_appear = score_tform4x4_fit(pts=pts_src,
                                                                                        tform4x4=ref_tform4x4_src[None,],
                                                                                        pts_ref=pts_ref,
                                                                                        dist_app_ref=dist_src_ref,
                                                                                        return_dists=True,
                                                                                        dist_app_weight=self.config.dist_appear_weight,
                                                                                        geo_cyclic_weight_temp=self.config.geo_cyclic_weight_temp,
                                                                                        app_cyclic_weight_temp=self.config.app_cyclic_weight_temp,
                                                                                        score_perc=self.config.ransac.score_perc)
                                # logger.info(f'sim: geo: {pose_dist_geo}, app: {pose_dist_appear}')
                                all_pred_pose_dist_geo[category][r, s] = pose_dist_geo
                                all_pred_pose_dist_appear[category][r, s] = pose_dist_appear
                                pred_ref_tform_src = ref_tform4x4_src.clone()
                                #pred_ref_tform_src[:3, :3] /= torch.linalg.norm(pred_ref_tform_src[:3, :3], dim=-1,
                                #                                                keepdim=True)

                                all_pred_ref_tform_src[category][r, s] = pred_ref_tform_src
                            else:
                                ref_vertices_mask = ref_sequences_mesh_ids_for_verts == ref_mesh_id
                                pts_ref = ref_meshes.verts[ref_vertices_mask].clone()

                                logger.info(f'category: {category}, pts-src: {pts_src.shape}, pts-ref: {pts_ref.shape}')

                                dist_ref_src = ref_sequences[ref_mesh_id].read_mesh_feats_dist(
                                    src_sequences[src_mesh_id]).to(device=self.device, dtype=dtype)

                                # division by two to normalize to 0. - 1.
                                dist_ref_src = dist_ref_src / 2.

                                # four points required, otherwise rotation yields an ambiguity. like planes without normals
                                src_tform4x4_ref = ransac(pts=pts_ref,
                                                          fit_func=partial(fit_tform4x4, pts_ref=pts_src,
                                                                           dist_ref=dist_ref_src),
                                                          score_func=partial(score_tform4x4_fit, pts_ref=pts_src,
                                                                             dist_app_ref=dist_ref_src,
                                                                             dist_app_weight=self.config.dist_appear_weight,
                                                                             geo_cyclic_weight_temp=self.config.geo_cyclic_weight_temp,
                                                                             app_cyclic_weight_temp=self.config.app_cyclic_weight_temp,
                                                                             score_perc=self.config.ransac.score_perc),
                                                          fits_count=self.config.ransac.samples, fit_pts_count=4)
                                #else:
                                #    ref_tform4x4_src = torch.eye(4).to(device=self.device, dtype=dtype)
                                from od3d.cv.optimization.gradient_descent import gradient_descent_se3
                                if self.config.refine_optimization_steps > 0:
                                    src_tform4x4_ref, ref_pts_offset = gradient_descent_se3(pts=pts_ref,
                                                                            models=src_tform4x4_ref,
                                                                            score_func=partial(score_tform4x4_fit, pts_ref=pts_src,
                                                                                           dist_app_ref=dist_ref_src,
                                                                                           dist_app_weight=self.config.dist_appear_weight,
                                                                                           geo_cyclic_weight_temp=self.config.geo_cyclic_weight_temp,
                                                                                           app_cyclic_weight_temp=self.config.app_cyclic_weight_temp,
                                                                                           score_perc=self.config.ransac.score_perc),
                                                                            steps=self.config.refine_optimization_steps,
                                                                            lr=self.config.refine_lr,
                                                                            pts_weight=self.config.refine_pts_weight,
                                                                            arap_weight=self.config.refine_arap_weight,
                                                                            arap_geo_std=self.config.refine_arap_geo_std,
                                                                            reg_weight=self.config.refine_reg_weight,
                                                                            return_pts_offset=True)

                                    all_pred_ref_pts_offset[category][s][ref_vertices_mask] = ref_pts_offset

                                _, pose_dist_geo, pose_dist_appear = score_tform4x4_fit(pts=pts_ref, tform4x4=src_tform4x4_ref[None,], pts_ref=pts_src,
                                                                                        dist_app_ref=dist_ref_src, return_dists=True,
                                                                                        dist_app_weight=self.config.dist_appear_weight,
                                                                                        geo_cyclic_weight_temp=self.config.geo_cyclic_weight_temp,
                                                                                        app_cyclic_weight_temp=self.config.app_cyclic_weight_temp,
                                                                                        score_perc=self.config.ransac.score_perc)


                                # logger.info(f'sim: geo: {pose_dist_geo}, app: {pose_dist_appear}')
                                all_pred_pose_dist_geo[category][r, s] = pose_dist_geo
                                all_pred_pose_dist_appear[category][r, s] = pose_dist_appear
                                pred_ref_tform_src = src_tform4x4_ref.clone()
                                pred_ref_tform_src = inv_tform4x4(src_tform4x4_ref).clone()
                                all_pred_ref_tform_src[category][r, s] = pred_ref_tform_src

                        gt_ref_tform_src = torch.eye(4).to(device=self.device)

                        diff_rot_angle_rad = get_pose_diff_in_rad(pred_tform4x4=pred_ref_tform_src, gt_tform4x4=gt_ref_tform_src)
                        logger.info(diff_rot_angle_rad)
                        results_diff_log_rot[category][r, s] = diff_rot_angle_rad


        results = OD3D_Results()
        #results_ref = OD3D_Results()
        for cat_id, category in enumerate(categories):
            category_results = OD3D_Results()
            exclude_diagonal = dataset_src.name == dataset_ref.name

            if exclude_diagonal:
                # excluding diagonal entries as these are predicted transformation between same instance
                if self.config.gt_cam_tform_obj_source is not None:
                    category_results[f'rot_diff_rad'] = results_diff_log_rot[category][
                        torch.eye(src_instances_count_per_category[cat_id]).to(device=self.device) == 0].reshape(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id]-1).permute(1, 0)
                category_results[f'sim'] = 1.0 - all_pred_pose_dist_geo[category][
                    torch.eye(src_instances_count_per_category[cat_id]).to(device=self.device) == 0].reshape(
                    ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id] - 1).permute(1,
                                                                                                                    0)
                category_results[f'pose_sim_geo'] = 1.0 - all_pred_pose_dist_geo[category][
                   torch.eye(src_instances_count_per_category[cat_id]).to(device=self.device) == 0].reshape(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id]-1).permute(1, 0)
                category_results[f'pose_sim_appear'] = 1.0 - all_pred_pose_dist_appear[category][
                    torch.eye(src_instances_count_per_category[cat_id]).to(device=self.device) == 0].reshape(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id]-1).permute(1, 0)
            else:
                if self.config.gt_cam_tform_obj_source is not None:
                    category_results[f'rot_diff_rad'] = results_diff_log_rot[category].permute(1, 0)
                category_results[f'sim'] = 1.0 - all_pred_pose_dist_geo[category].permute(1, 0)
                category_results[f'pose_sim_geo'] = 1.0 - all_pred_pose_dist_geo[category].permute(1, 0)
                category_results[f'pose_sim_appear'] = 1.0 - all_pred_pose_dist_appear[category].permute(1, 0)

            results += category_results # .mean()
            category_results_mean = category_results.add_prefix(category)
            category_results_mean = category_results_mean.mean()
            category_results_mean.log()
            logger.info(category_results_mean)

        results_mean = results.mean()
        results_mean.log()
        logger.info(results_mean)


        ref_meshes.rgb = ref_meshes.get_verts_ncds_cat_with_mesh_ids()
        src_meshes.rgb = src_meshes.get_verts_ncds_cat_with_mesh_ids()

        ### VISUALIZATIONS
        ref_instance_ids = torch.LongTensor(list(range(ref_instances_count)))
        src_instance_ids = torch.LongTensor(list(range(src_instances_count)))

        # for suffix in ['', '_filtered', '_mesh', '_mesh_filtered']:
        #     for ref_instance_id_in_category in range(max(ref_instances_count_per_category)):
        #         aligned_name = f'{self.config.aligned_name}{suffix}/r{ref_instance_id_in_category}'
        #         aligned_path = dataset_src.path_preprocess.joinpath('aligned', aligned_name)
        #         od3d.io.rm_dir(aligned_path)

        for cat_id, category in enumerate(categories):
            logger.info(f'category {category}')

            ref_category_instance_ids = ref_instance_ids[ref_map_seq_to_cat == cat_id]
            if self.config.use_only_first_reference:
                ref_category_instance_ids = ref_category_instance_ids[:1]

            src_category_instance_ids = src_instance_ids[src_map_seq_to_cat == cat_id]

            for ref_instance_id_in_category, ref_instance_id in enumerate(ref_category_instance_ids):
                if self.config.aligned_name is not None:
                    aligned_name = f'{self.config.aligned_name}/r{ref_instance_id_in_category}'
                    aligned_filtered_name = f'{self.config.aligned_name}_filtered/r{ref_instance_id_in_category}'

                # geometry/appearance: 0.81/0.18 | 0.59/0.29 | 0.89/0.29 | 0.9/0.65 (best qualit.) | 0.9/0.55 | 0.9 / 0.6 | 0.95 0.76 | 0.92 0.53 | 0.91 0.55
                if self.config.gt_cam_tform_obj_source is not None:
                    rot_diff_rad = results_diff_log_rot[category][ref_instance_id_in_category, :]
                    accurate_pi6 = rot_diff_rad < (math.pi / 6.)
                    accurate_pi18 = rot_diff_rad < (math.pi / 18.)
                accurate_sim_geo = (1.0 - all_pred_pose_dist_geo[category][ref_instance_id_in_category, :]) > 0.91
                # accurate_sim_geo
                accurate_sim_appear = (1.0 - all_pred_pose_dist_appear[category][ref_instance_id_in_category, :]) > 0.50
                accurate_sim = accurate_sim_geo * accurate_sim_appear

                pts3d = []
                pts3d_colors = []

                src_meshes_cloned = []
                ref_meshes_cloned = []
                for src_instance_id_in_category, src_instance_id in enumerate(src_category_instance_ids):
                    # prediction
                    aligned_cuboid_tform_src = all_pred_ref_tform_src[category][ref_instance_id_in_category, src_instance_id_in_category]

                    src_vertices_mask = src_sequences_mesh_ids_for_verts == src_instance_id
                    ref_vertices_mask = ref_sequences_mesh_ids_for_verts == ref_instance_id

                    ref_mesh_cloned = ref_meshes.get_meshes_with_ids(meshes_ids=[ref_instance_id], clone=True)
                    ref_mesh_cloned.verts[:] = (ref_mesh_cloned.verts + all_pred_ref_pts_offset[category][src_instance_id_in_category][ref_vertices_mask]).detach()
                    src_mesh_cloned = src_meshes.get_meshes_with_ids(meshes_ids=[src_instance_id], clone=True)
                    src_mesh_cloned.verts[:] = (transf3d_broadcast(src_mesh_cloned.verts, aligned_cuboid_tform_src)).detach()

                    ref_verts_ncds = ref_mesh_cloned.verts.clone().detach()
                    ref_verts_ncds = (ref_verts_ncds - ref_verts_ncds.min(dim=0, keepdim=True).values) / (1e-10 + ref_verts_ncds.max(dim=0, keepdim=True).values - ref_verts_ncds.min(dim=0, keepdim=True).values)
                    ref_verts_ncds = (ref_verts_ncds + 1.) / 2.

                    if self.config.aligned_name is not None:
                        if accurate_sim[src_instance_id_in_category]:
                            src_sequences[src_instance_id].write_aligned_mesh_and_tform_obj(mesh=ref_mesh_cloned, aligned_obj_tform_obj=aligned_cuboid_tform_src, aligned_name=aligned_filtered_name)

                        src_sequences[src_instance_id].write_aligned_mesh_and_tform_obj(mesh=ref_mesh_cloned, aligned_obj_tform_obj=aligned_cuboid_tform_src, aligned_name=aligned_name)

                    dist_verts_ref = src_sequences[src_instance_id].read_mesh_feats_dist(ref_sequences[ref_instance_id]).to(device=self.device, dtype=dtype)
                    dist_verts_ref = dist_verts_ref / 2.
                    dists_verts_min_ref_vertices = dist_verts_ref.min(dim=-1)[1]

                    _, dist_ref_geometry_weight, dist_ref_appear_weight = score_tform4x4_fit(pts=src_mesh_cloned.verts,
                                                                                             tform4x4=torch.eye(4)[None,].to(device=self.device),
                                                                                             pts_ref=ref_mesh_cloned.verts,
                                                                                             dist_app_ref=dist_verts_ref,
                                                                                             return_weights=True,
                                                                                             dist_app_weight=self.config.dist_appear_weight,
                                                                                             geo_cyclic_weight_temp=self.config.geo_cyclic_weight_temp,
                                                                                             app_cyclic_weight_temp=self.config.app_cyclic_weight_temp,
                                                                                             score_perc=self.config.ransac.score_perc)

                    dist_ref_geometry_weight = dist_ref_geometry_weight / dist_ref_geometry_weight.max()
                    dist_ref_appear_weight = dist_ref_appear_weight / dist_ref_appear_weight.max()

                    ref_mesh_cloned.rgb[:] = ref_verts_ncds
                    src_mesh_cloned.rgb[:] = ref_verts_ncds[dists_verts_min_ref_vertices]
                    src_mesh_cloned.rgb *= dist_ref_appear_weight[0, :src_mesh_cloned.rgb.shape[0], None]

                    #co3d_src_tform_src = self.sequences_co3d_tform_droid_slam[instance_id]
                    #pts3d.append(transf3d_broadcast(pts3d=self.sequences[instance_id].pcl.to(device=self.device, dtype=dtype), transf4x4=tform4x4(all_pred_ref_tform_src[category][ref_instance_id_in_category, instance_id_in_category], inv_tform4x4(co3d_src_tform_src))))

                    src_pts3d, src_pts3d_colors, src_pts3d_normals = src_sequences[src_instance_id].read_pcl()
                    src_pts3d = src_pts3d.to(device=self.device, dtype=dtype)
                    src_pts3d_colors = src_pts3d_colors.to(device=self.device, dtype=dtype)
                    src_pts3d_normals = src_pts3d_normals.to(device=self.device, dtype=dtype)
                    src_pts3d = transf3d_broadcast(src_pts3d, aligned_cuboid_tform_src).detach()

                    pts3d.append(src_pts3d)
                    pts3d_colors.append(src_pts3d_colors)

                    src_meshes_cloned.append(src_mesh_cloned)
                    ref_meshes_cloned.append(ref_mesh_cloned)

                viewpoints_count = 2
                category_meshes = Meshes.load_from_meshes(src_meshes_cloned, device=self.device)
                #category_meshes = Meshes.load_from_meshes(ref_meshes_cloned, device=self.device)

                imgs = show_scene(pts3d=pts3d, pts3d_colors=pts3d_colors, return_visualization=True, viewpoints_count=viewpoints_count, meshes=category_meshes, device=self.device, meshes_add_translation=True, pts3d_add_translation=True)

                from od3d.cv.visual.draw import add_boolean_table
                if self.config.gt_cam_tform_obj_source is not None:
                    accurate_table = torch.stack([accurate_pi6, accurate_pi18, accurate_sim,  accurate_sim_geo, accurate_sim_appear], dim=0)
                else:
                    accurate_table = torch.stack(
                        [accurate_sim,  accurate_sim_geo, accurate_sim_appear], dim=0) # ,
                from od3d.cv.visual.crop import crop_white_border_from_img
                for v in range(viewpoints_count):
                    img = crop_white_border_from_img(imgs[v])
                    if self.config.gt_cam_tform_obj_source is not None:
                        img = add_boolean_table(img, table=accurate_table,
                                                text=['Label (PI/6)', 'Label (PI/18)', 'Sim.', 'Sim. Geo.', 'Sim. App.'])
                    else:
                        img = add_boolean_table(img, table=accurate_table,
                                                text=['Sim.', 'Sim. Geo.', 'Sim. App.'])

                    results_visual = OD3D_Results()
                    results_visual[f'{category}'] = image_as_wandb_image(img, caption='blub')
                    results_visual.log_with_prefix('aligned')



    def test(self, dataset: OD3D_Dataset, config_inference: DictConfig = None):
        return OD3D_Results()

