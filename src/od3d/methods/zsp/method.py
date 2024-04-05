import logging

import od3d.io

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
from od3d.datasets.co3d import CO3D
import torch
import random
import requests
import pickle
import io
from od3d.cv.geometry.transform import inv_tform4x4, tform4x4
from od3d.datasets.co3d.enum import CAM_TFORM_OBJ_SOURCES
from od3d.cv.metric.pose import get_pose_diff_in_rad

from od3d.cv.visual.show import show_scene
from od3d.cv.geometry.transform import depth2pts3d_grid, transf3d_broadcast, inv_tform4x4, proj3d2d_broadcast



class ZSP(OD3D_Method):
    def setup(self):
        pass

    def __init__(
            self,
            config: DictConfig,
            logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)

        self.device = 'cpu' #'cuda:0'
        self.docker_port = None
        # init Network
        # self.net = OD3D_Model(config.docker)

        self.transform_train = OD3D_Transform.subclasses[config.train.transform.class_name].create_from_config(config=config.train.transform)
        self.transform_test = OD3D_Transform.subclasses[config.test.transform.class_name].create_from_config(config=config.test.transform)
        self.target_data = None

    def get_gpus(self):
        if torch.cuda.is_available():
            gpus_uuids = torch.cuda._raw_device_uuid_nvml()
            gpu_uuid = gpus_uuids[torch.cuda.current_device()]
            gpus = gpu_uuid
        else:
            gpus = 'all'
        return gpus

    def is_port_in_use(self, port):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except socket.error:
                return True
            return False

    def start_docker(self):
        logger.info(f'current cuda device {torch.cuda.current_device()}')

        if self.docker_port is not None:
            logger.info('docker already running')
            return

        self.docker_port = 5000
        while self.is_port_in_use(self.docker_port):
            self.docker_port += 1

        from od3d.cli.docker import get_cmd_zsp_run
        cmd = get_cmd_zsp_run(gpus=self.get_gpus(), port=str(self.docker_port))
        od3d.io.run_cmd(cmd, logger=logger, live=False, background=True)
        from time import sleep
        sleep(15)

    def stop_docker(self):
        from od3d.cli.docker import get_cmd_zsp_stop
        cmd = get_cmd_zsp_stop(port=self.docker_port)
        od3d.io.run_cmd(cmd, logger=logger, live=True)
        self.docker_port = None
        from time import sleep
        sleep(15)

    def train(self, datasets_train: Dict[str, OD3D_Dataset], datasets_val: Dict[str, OD3D_Dataset]):
        torch.cuda.empty_cache()

        self.start_docker()

        dataset_src: CO3D = datasets_train['src']
        dataset_ref: CO3D = datasets_train['labeled']

        categories = dataset_src.categories
        self.categories = categories

        src_sequences = dataset_src.get_sequences()
        ref_sequences = dataset_ref.get_sequences()

        src_sequences_unique_names = [seq.name_unique for seq in src_sequences]
        ref_sequences_unique_names = [seq.name_unique for seq in ref_sequences]

        src_map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in src_sequences_unique_names])
        ref_map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in ref_sequences_unique_names])

        categories_count = len(categories)

        src_instances_count_per_category = [(src_map_seq_to_cat == c).sum().item() for c in range(categories_count)]
        ref_instances_count_per_category = [(ref_map_seq_to_cat == c).sum().item() for c in range(categories_count)]

        src_instances_count = len(src_sequences)
        ref_instances_count = len(ref_sequences)
        dtype= torch.float

        results_diff_log_rot = {}
        all_pred_ref_tform_src = {}
        all_pred_pose_dist_geo = {}
        all_pred_pose_dist_appear = {}
        if self.config.use_train_only_to_collect_target_data:
            self.target_data = {}

        for cat_id, category in enumerate(categories):
            if self.config.use_train_only_to_collect_target_data:
                self.target_data[cat_id] = []
            logger.info(f'category id {cat_id} name {category}')
            src_instance_ids = torch.LongTensor(list(range(src_instances_count)))
            ref_instance_ids = torch.LongTensor(list(range(ref_instances_count)))

            #src_category_instance_ids = src_instance_ids[src_map_seq_to_cat == cat_id]
            #ref_category_instance_ids = ref_instance_ids[ref_map_seq_to_cat == cat_id]

            # this ensures that we first label the axis, which are required later to fit the cuboid
            #droid_slam_labeled_tform_droid_slam = ref_sequences[
            #    ref_category_instance_ids[0]].labeled_obj_tform_obj.to(dtype=dtype, device=self.device)
            #droid_slam_labeled_cuboid_tform_droid_slam_labeled = ref_sequences[
            #    ref_category_instance_ids[0]].labeled_cuboid_obj_tform_labeled_obj.to(dtype=dtype,
            #                                                                          device=self.device)

            results_diff_log_rot[category] = torch.zeros(
                size=(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id])).to(
                device=self.device, dtype=dtype)
            all_pred_ref_tform_src[category] = torch.zeros(
                size=(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id], 4, 4)).to(
                device=self.device, dtype=dtype)

        for cat_id, category in enumerate(categories):
            logger.info(f'category {category}')
            src_instance_ids = torch.LongTensor(list(range(src_instances_count)))
            ref_instance_ids = torch.LongTensor(list(range(ref_instances_count)))

            src_mesh_ids = src_instance_ids[src_map_seq_to_cat == cat_id]
            ref_mesh_ids = ref_instance_ids[ref_map_seq_to_cat == cat_id]

            for r, ref_mesh_id in enumerate(ref_mesh_ids):


                """
                       Args:
                           ref_image (torch.Tensor): Bx3xSxS
                           all_target_images (torch.Tensor): N_TGTx3xSxS
                           ref_scalings (torch.Tensor): Bx2 (H,W) rescaling from original resolution
                           target_scalings (torch.Tensor): N_TGTx2 (H,W) rescaling from original resolution
                           ref_depth_map (torch.Tensor): Bx1xHxW
                           target_depth_map (torch.Tensor): N_TGTx1xHxW
                           ref_cam_extr (torch.Tensor): Bx4x4
                           target_cam_extr (torch.Tensor): N_TGTx4x4
                           ref_cam_intr (torch.Tensor): Bx4x4
                           target_cam_intr (torch.Tensor): N_TGTx4x4
                       Returns:
                           target_tform_ref (torch.Tensor): Bx4x4
                   """

                from od3d.datasets.co3d.enum import PCL_SOURCES

                transform_low_res = OD3D_Transform.create_by_name('scale_mask_separate_1_centerzoom224')
                transform_high_res = OD3D_Transform.create_by_name('scale_mask_separate_1_centerzoom224')
                transform_rescale = 1.

                logger.info(f'ref {r+1} out of {len(ref_mesh_ids)}')
                ref_frames_N = len(ref_sequences[ref_mesh_id].frames_names)
                ref_frames_S = 10
                ref_frames_indices = np.linspace(0, ref_frames_N-1, ref_frames_S).astype(int)
                ref_frames_low_res = [transform_low_res(ref_sequences[ref_mesh_id].get_frame_by_index(ref_frame_index)) for ref_frame_index in ref_frames_indices]
                ref_frames_high_res = [transform_high_res(ref_sequences[ref_mesh_id].get_frame_by_index(ref_frame_index)) for ref_frame_index in ref_frames_indices]

                #ref_frames = [ref_sequences[ref_mesh_id].get_frame_by_index(ref_frame_index) for ref_frame_index in ref_frames_indices]

                #ref_sequences[ref_mesh_id].show(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.DROID_SLAM, pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN, show_imgs=True)


                srcs_frames_N = [len(src_sequences[src_mesh_id].frames_names) for src_mesh_id in src_mesh_ids]
                srcs_frames_ids = [src_frames_N // 2 for src_frames_N in srcs_frames_N]
                # srcs_frames_ids = [src_frames_N // 5 for src_frames_N in srcs_frames_N]

                #src_frames_N = len(src_sequences[src_mesh_id].frames_names)
                # source_frame_index = random.choice(np.arange(src_frames_N))
                #source_frame_index = src_frames_N // 2


                src_frames_low_res = [
                    transform_low_res(src_sequences[src_mesh_id].get_frame_by_index(srcs_frames_ids[s])) for
                    s, src_mesh_id in enumerate(src_mesh_ids)]
                src_frames_high_res = [
                    transform_high_res(src_sequences[src_mesh_id].get_frame_by_index(srcs_frames_ids[s])) for
                    s, src_mesh_id in enumerate(src_mesh_ids)]

                #src_frames = [src_sequences[src_mesh_id].get_frame_by_index(srcs_frames_ids[s])  for s, src_mesh_id in enumerate(src_mesh_ids)]

                #src_frame = self.transform_train(src_sequences[src_mesh_id].get_frame_by_index(source_frame_index))


                # if dataset_ref.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ZSP_LABELED or \
                #         dataset_ref.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM or \
                #         dataset_ref.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_LABELED:
                #     ref_sequence_scale = ref_sequences[ref_mesh_id].get_a_src_scale_b_src(CAM_TFORM_OBJ_SOURCES.DROID_SLAM, CAM_TFORM_OBJ_SOURCES.CO3D)
                #     ref_cam_source = CAM_TFORM_OBJ_SOURCES.DROID_SLAM
                #
                # else:
                #     ref_cam_source = CAM_TFORM_OBJ_SOURCES.CO3D
                #     ref_sequence_scale = torch.Tensor([1.])[None,]
                #     logger.warning('ref scale = 1.')
                #

                ref_cam_source = None # CAM_TFORM_OBJ_SOURCES.PCL
                ref_sequence_scale = torch.Tensor([1.])[None,]

                # if dataset_src.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ZSP_LABELED or \
                #         dataset_src.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM or \
                #         dataset_src.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_LABELED:
                #     src_sequences_scales = torch.Tensor([src_sequences[src_mesh_id].get_a_src_scale_b_src(CAM_TFORM_OBJ_SOURCES.DROID_SLAM, CAM_TFORM_OBJ_SOURCES.CO3D) for s, src_mesh_id in enumerate(src_mesh_ids)])
                #
                #     src_cam_source = CAM_TFORM_OBJ_SOURCES.DROID_SLAM
                # else:
                #     src_sequences_scales = torch.Tensor([1. for s, src_mesh_id in enumerate(src_mesh_ids)])
                #     src_cam_source = CAM_TFORM_OBJ_SOURCES.CO3D
                #     logger.warning('src scale = 1.')
                # #show_scene(pts3d=[ref_sequences[ref_mesh_id].get_pcl(PCL_SOURCES.CO3D) for ref_mesh_id in ref_mesh_ids], pts3d_add_translation=True)

                src_cam_source = None
                src_sequences_scales = torch.Tensor([1. for s, src_mesh_id in enumerate(src_mesh_ids)])

                # data = {'img': batch.rgb, 'cam_tform4x4_obj': batch.cam_tform4x4_obj}
                # src_H, src_W =
                ref_image = torch.stack([src_frame.get_rgb() for src_frame in src_frames_low_res], dim=0)
                B = len(ref_image)
                ref_scalings = src_frames_low_res[0].size.clone()  # torch.stack([src_frame.size], dim=0)
                ref_scalings[:] = transform_rescale # * src_frame.depth_mask
                ref_depth_map = torch.stack([ src_frame.get_depth() for s, src_frame in enumerate(src_frames_high_res)], dim=0)
                # ref_depth_map[ref_depth_map > ref_depth_map.flatten(2).mean(dim=-1)[..., None, None] * 2] = 0

                ref_cam_intr = torch.stack([src_frame.get_cam_intr4x4() for src_frame in src_frames_high_res], dim=0)
                ref_cam_extr = torch.stack([src_frame.get_cam_tform4x4_obj(src_cam_source) for src_frame in src_frames_high_res], dim=0)
                ref_cam_extr[:, :3, 3] /= src_sequences_scales[:, None]

                all_target_images = torch.stack([ref_frame.get_rgb() for ref_frame in ref_frames_low_res], dim=0)[None,].repeat(B, 1, 1, 1, 1)
                N_TGT = len(ref_frames_low_res)
                target_scalings = ref_frames_low_res[
                    0].size.clone()  # torch.stack([ref_frame.size for ref_frame in ref_frames], dim=0)[None,]
                target_scalings[:] = transform_rescale # * ref_frame.depth_mask
                target_depth_map = torch.stack([ref_frame.get_depth() for r, ref_frame in enumerate(ref_frames_high_res)], dim=0)[None,].repeat(B, 1, 1, 1, 1)
                # target_depth_map[target_depth_map > target_depth_map.flatten(3).mean(dim=-1)[..., None, None] * 2 ] = 0
                target_cam_intr = torch.stack([ref_frame.get_cam_intr4x4() for ref_frame in ref_frames_high_res], dim=0)[None,].repeat(B, 1, 1, 1)
                target_cam_extr = torch.stack([ref_frame.get_cam_tform4x4_obj(ref_cam_source) for ref_frame in ref_frames_high_res], dim=0)[None,].repeat(B, 1, 1, 1)
                target_cam_extr[:, :, :3, 3] /= ref_sequence_scale[:, :, None]

                # if ref_cam_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM:
                #
                #     pts3d = ref_sequences[ref_mesh_id].get_pcl(PCL_SOURCES.DROID_SLAM_CLEAN)
                #     H, W = src_frames[0].size.to(int)
                #
                #     pxl2d = proj3d2d_broadcast(pts3d=pts3d[:, None, None], proj4x4=tform4x4(target_cam_intr, target_cam_extr)[None,]).permute(1, 2, 0, 3)
                #     pxl2d_z = transf3d_broadcast(pts3d[:, None, None], target_cam_extr[None,]).permute(1, 2, 0, 3)[:, :, :, 2]# [None, None].expand(*pxl2d.shape[:-1])
                #     depth_map = torch.zeros(size=target_depth_map.shape)
                #     from od3d.cv.select import index_MD_to_1D
                #     pxl2d_out_of_bounds = (pxl2d[:, :, :, 0] < 0) + (pxl2d[:, :, :, 0] > (W-1)) + (pxl2d[:, :, :, 1] < 0) + (pxl2d[:, :, :, 1] > (H-1))
                #     pxl2d[pxl2d_out_of_bounds] = 0
                #     pxl1d = index_MD_to_1D(indexMD=pxl2d.to(int), inputMD=depth_map, dims=[-1, -2])
                #     # target_depth_map = torch.scatter_reduce(depth_map.reshape(B, N_TGT, -1), src=pxl2d_z, index=pxl1d, reduce='amin', dim=-1, include_self=False).reshape(B, N_TGT, 1, H, W)
                #     target_depth_map = torch.scatter_reduce(depth_map.permute(0, 1, 2, 4, 3).reshape(B, N_TGT, -1), src=pxl2d_z, index=pxl1d, reduce='amin', dim=-1, include_self=False).reshape(B, N_TGT, 1, W, H).permute(0, 1, 2, 4, 3)
                #     target_depth_map[:, :, :, 0, 0] = 0.
                #
                #     srcs_pts3d = [src_sequences[src_mesh_id].get_pcl(PCL_SOURCES.DROID_SLAM_CLEAN) for src_mesh_id in src_mesh_ids]
                #     pts3d_counts_max = max([len(p) for p in srcs_pts3d])
                #     pts3d = torch.zeros(B, pts3d_counts_max, 3)
                #     srcs_pts3d_first = torch.stack([p[0] for p in srcs_pts3d], dim=0)
                #     pts3d = srcs_pts3d_first[:, None,].expand(*pts3d.shape)
                #     pxl2d = proj3d2d_broadcast(pts3d=pts3d, proj4x4=tform4x4(ref_cam_intr, ref_cam_extr)[:, None,])
                #     pxl2d_z = transf3d_broadcast(pts3d, ref_cam_extr[:, None,])[:, :, 2]# [None, None].expand(*pxl2d.shape[:-1])
                #     depth_map = torch.zeros(size=ref_depth_map.shape)
                #     from od3d.cv.select import index_MD_to_1D
                #     pxl2d_out_of_bounds = (pxl2d[:, :, 0] < 0) + (pxl2d[:, :, 0] > (W-1)) + (pxl2d[:, :, 1] < 0) + (pxl2d[:, :, 1] > (H-1))
                #     pxl2d[pxl2d_out_of_bounds] = 0
                #     pxl1d = index_MD_to_1D(indexMD=pxl2d.to(int), inputMD=depth_map, dims=[-1, -2])
                #     # target_depth_map = torch.scatter_reduce(depth_map.reshape(B, N_TGT, -1), src=pxl2d_z, index=pxl1d, reduce='amin', dim=-1, include_self=False).reshape(B, N_TGT, 1, H, W)
                #     ref_depth_map = torch.scatter_reduce(depth_map.permute(0, 1, 3, 2).reshape(B, -1), src=pxl2d_z, index=pxl1d, reduce='amin', dim=-1, include_self=False).reshape(B, 1, W, H).permute(0, 1, 3, 2)
                #     ref_depth_map[:, :, 0, 0] = 0.

                if self.config.use_train_only_to_collect_target_data:
                    #ref_cam_source = ref_frames_high_res[0].cam_tform_obj_source
                    #ref_sequence_scale = ref_sequences[ref_mesh_id].get_a_src_scale_b_src(
                    #    ref_cam_source, CAM_TFORM_OBJ_SOURCES.CO3D)
                    target_cam_extr = \
                    torch.stack([ref_frame.get_cam_tform4x4_obj() for ref_frame in ref_frames_high_res], dim=0)[
                        None,].repeat(B, 1, 1, 1)
                    target_cam_extr[:, :, :3, 3] /= ref_sequence_scale[:, :, None]

                    self.target_data[cat_id].append({
                        "all_target_images": all_target_images[0],
                        "target_scalings": target_scalings,
                        "target_depth_map": target_depth_map[0],
                        "target_cam_extr": target_cam_extr[0],
                        "target_cam_intr": target_cam_intr[0],
                    })
                    continue

                #
                # pts3d = [depth2pts3d_grid(ref_depth_map[b], cam_intr4x4=ref_cam_intr[b]) for b in range(B)]
                # pts3d = torch.cat(pts3d, dim=0)
                # pts3d = transf3d_broadcast(pts3d=pts3d.permute(0, 2, 3, 1).reshape(B, -1, 3), transf4x4=inv_tform4x4(ref_cam_extr[:, None]))
                # show_scene(pts3d=pts3d)
                #
                # pts3d = [depth2pts3d_grid(target_depth_map[0, n], cam_intr4x4=target_cam_intr[0, n]) for n in range(N_TGT)]
                # pts3d = torch.cat(pts3d, dim=0)
                # pts3d = transf3d_broadcast(pts3d=pts3d.permute(0, 2, 3, 1).reshape(N_TGT, -1, 3), transf4x4=inv_tform4x4(target_cam_extr[0, :, None]))
                # show_scene(pts3d=pts3d)

                #from od3d.cv.visual.show import show_imgs
                #show_imgs(rgbs=torch.cat([all_target_images[0], ref_image], dim=0))
                # show_imgs(rgbs=torch.cat([all_target_images[0], (target_depth_map[0].repeat(1, 3, 1, 1) * 100).clamp(0, 255)]  , dim=0)) # (target_depth_map[0].repeat(1, 3, 1, 1) * 100).clamp(0, 255)

                #show_imgs(rgbs=ref_image)
                # !!! dist scale of droid slam, does not fit to depth scale of
                # show_imgs(rgbs=ref_depth_map)
                #show_imgs(rgbs=target_depth_map[0])
                data = {
                    "ref_image": ref_image,
                    "all_target_images": all_target_images,
                    "ref_scalings": ref_scalings,
                    "target_scalings": target_scalings,
                    "ref_depth_map": ref_depth_map,
                    "target_depth_map": target_depth_map,
                    "ref_cam_extr": ref_cam_extr,
                    "target_cam_extr": target_cam_extr,
                    "ref_cam_intr": ref_cam_intr,
                    "target_cam_intr": target_cam_intr,
                }
                bytes_io = io.BytesIO()
                pickle.dump(data, bytes_io, pickle.HIGHEST_PROTOCOL)
                bytes_io.seek(0)

                try:
                    resp = requests.post(f"http://127.0.0.1:{self.docker_port}/predict", files={"file": bytes_io})
                    # all_pred_ref_tform_src = resp.json()['obj2_tform_obj1']
                    pred_ref_tform_srcs = resp.json()['obj2_tform_obj1']
                    pred_ref_tform_srcs = torch.Tensor(pred_ref_tform_srcs).to(device=self.device)
                except Exception as e:
                    logger.info(f'failed to retrieve predicted `ref_tform_source`')
                    logger.info(e)
                    pred_ref_tform_srcs = torch.eye(4)[None,].repeat(B, 1, 1).to(device=self.device) * 0.

                for s, src_mesh_id in enumerate(src_mesh_ids):
                    pred_ref_tform_src = pred_ref_tform_srcs[s]
                    all_pred_ref_tform_src[category][r, s] = pred_ref_tform_src

                    ref_labeled_obj_tform_obj = torch.eye(4).to(device=self.device)
                    src_labeled_obj_tform_obj = torch.eye(4).to(device=self.device)
                    gt_ref_tform_src = tform4x4(inv_tform4x4(ref_labeled_obj_tform_obj), src_labeled_obj_tform_obj)

                    if self.config.aligned_name is not None:
                        aligned_name = f'{self.config.aligned_name}/r{r}'
                        aligned_cuboid_tform_obj = tform4x4(ref_labeled_obj_tform_obj, pred_ref_tform_src)
                        src_sequences[src_mesh_id].write_aligned_mesh_and_tform_obj(mesh=ref_sequences[ref_mesh_id].read_mesh(), aligned_obj_tform_obj=aligned_cuboid_tform_obj, aligned_name=aligned_name)

                    diff_rot_angle_rad = get_pose_diff_in_rad(pred_tform4x4=pred_ref_tform_src, gt_tform4x4=gt_ref_tform_src)
                    logger.info(diff_rot_angle_rad)
                    results_diff_log_rot[category][r, s] = diff_rot_angle_rad

        if not self.config.use_train_only_to_collect_target_data:
            results = OD3D_Results()
            for cat_id, category in enumerate(categories):
                category_results = OD3D_Results()
                exclude_diagonal = dataset_src.name == dataset_ref.name

                if exclude_diagonal:
                    # excluding diagonal entries as these are predicted transformation between same instance
                    if self.config.use_gt_src:
                        category_results[f'rot_diff_rad'] = results_diff_log_rot[category][
                            torch.eye(src_instances_count_per_category[cat_id]).to(device=self.device) == 0].reshape(ref_instances_count_per_category[cat_id], src_instances_count_per_category[cat_id]-1).permute(1, 0)
                else:
                    if self.config.use_gt_src:
                        category_results[f'rot_diff_rad'] = results_diff_log_rot[category].permute(1, 0)

                results += category_results.mean()
                category_results_mean = category_results.add_prefix(category)
                category_results_mean = category_results_mean.mean()
                category_results_mean.log()
                logger.info(category_results_mean)

            results_mean = results.mean()
            results_mean.log()
            logger.info(results_mean)

        self.stop_docker()

    def test(self, dataset: OD3D_Dataset, config_inference: DictConfig = None):
        self.start_docker()

        if self.target_data is not None:
            logger.info(f'test dataset {dataset.name}')

            score_metric_name = 'pose/acc_pi18'  # 'pose/acc_pi18' 'pose/acc_pi6'
            score_ckpt_val = 0.
            score_latest = 0.

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
            self.stop_docker()
            return results_epoch
        else:
            self.stop_docker()
            return OD3D_Results()

    def inference_batch(self, batch: OD3D_Frames):

        # if dataset_ref.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ZSP_LABELED or \
        #         dataset_ref.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM or \
        #         dataset_ref.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_LABELED:
        #     ref_sequence_scale = ref_sequences[ref_mesh_id].get_a_src_scale_b_src(CAM_TFORM_OBJ_SOURCES.DROID_SLAM,
        #                                                                           CAM_TFORM_OBJ_SOURCES.CO3D)
        #     ref_cam_source = CAM_TFORM_OBJ_SOURCES.DROID_SLAM
        #
        # else:
        #     ref_cam_source = CAM_TFORM_OBJ_SOURCES.CO3D
        #     ref_sequence_scale = torch.Tensor([1.])[None,]
        #     logger.warning('ref scale = 1.')


        # assert all(batch.label[0] == batch.label)
        B = len(batch)
        ref_image = batch.rgb
        ref_scalings = torch.ones(size=(1,)).to(device=self.device)
        ref_depth_map = batch.depth
        ref_cam_intr = batch.cam_intr4x4
        ref_cam_extr = batch.cam_tform4x4_obj
        from od3d.datasets.objectnet3d.enum import OBJECTNET3D_SCALE_NORMALIZE_TO_REAL
        from od3d.datasets.pascal3d.enum import PASCAL3D_SCALE_NORMALIZE_TO_REAL
        from od3d.datasets.pascal3d import Pascal3D
        from od3d.datasets.objectnet3d import ObjectNet3D

        # for b in range(B):
        #     if dataset_type == Pascal3D:
        #         b_ref_scale = PASCAL3D_SCALE_NORMALIZE_TO_REAL[batch.category[b]]
        #         ref_depth_map[b] /= b_ref_scale
        #         ref_cam_extr[b, :3, 3] /= b_ref_scale
        #     elif dataset_type == ObjectNet3D:
        #         b_ref_scale = OBJECTNET3D_SCALE_NORMALIZE_TO_REAL[batch.category[b]]
        #         ref_depth_map[b] /= b_ref_scale
        #         ref_cam_extr[b, :3, 3] /= b_ref_scale

        #ref_scale = PASCAL3D_SCALE_NORMALIZE_TO_REAL[batch.category[0]]
        #ref_depth_map /= ref_scale
        #ref_cam_extr[:, :3, 3] /= ref_scale

        from od3d.cv.visual.show import show_img
        #show_img(ref_depth_map[0])
        # pts3d = [depth2pts3d_grid(ref_depth_map[b], cam_intr4x4=ref_cam_intr[b]) for b in range(B)]
        # pts3d = torch.cat(pts3d, dim=0).to(device=self.device, dtype=torch.float)
        # pts3d = transf3d_broadcast(pts3d=pts3d.permute(0, 2, 3, 1).reshape(B, -1, 3), transf4x4=inv_tform4x4(ref_cam_extr[:, None]))
        # show_scene(pts3d=pts3d, cams_intr4x4=batch.cam_intr4x4, cams_tform4x4_world=batch.cam_tform4x4_obj, cams_imgs=batch.rgb, cams_imgs_resize=True)


        # multiple_categorical_target_data = self.target_data[MAP_CATEGORIES_PASCAL3D_TO_OD3D[batch.category[0]]]

        results = OD3D_Results()

        all_rot_diff_rad = []
        all_cam_tform4x4_obj = []
        # note: assuming same amount of references for all categories (elements in batch)
        for r in range(len(self.target_data[0])):
            all_target_images = []
            target_depth_map = []
            target_cam_extr = []
            target_cam_intr = []
            target_scalings = self.target_data[0][0]['target_scalings']
            for b in range(B):
                categorical_target_data = self.target_data[batch.category_id[b].item()][r]
                all_target_images.append(categorical_target_data['all_target_images']) #[None,].repeat(B, 1, 1, 1, 1)
                target_depth_map.append(categorical_target_data['target_depth_map']) #[None,].repeat(B, 1, 1, 1, 1)
                target_cam_extr.append(categorical_target_data['target_cam_extr']) #[None,].repeat(B, 1, 1, 1)
                target_cam_intr.append(categorical_target_data['target_cam_intr']) #[None,].repeat(B, 1, 1, 1)

            all_target_images = torch.stack(all_target_images, dim=0)
            target_depth_map = torch.stack(target_depth_map, dim=0)
            target_cam_extr = torch.stack(target_cam_extr, dim=0)
            target_cam_intr = torch.stack(target_cam_intr, dim=0)
            # from od3d.cv.visual.show import show_img, show_imgs, imgs_to_img
            # from od3d.cv.visual.blend import blend_rgb
            # img_rgb = imgs_to_img(torch.cat([target_depth_map[0] == 0., ref_depth_map ==0.], dim=0))
            # img_mask = imgs_to_img(torch.cat([all_target_images[0], ref_image], dim=0))
            # show_img(blend_rgb(img_rgb, img_mask))
            # show_imgs(rgbs=torch.cat([target_depth_map[0] == 0., ref_depth_map ==0.], dim=0))
            # show_imgs(rgbs=torch.cat([all_target_images[0], ref_image], dim=0))

            data = {
                "ref_image": ref_image,
                "all_target_images": all_target_images,
                "ref_scalings": ref_scalings,
                "target_scalings": target_scalings,
                "ref_depth_map": ref_depth_map,
                "target_depth_map": target_depth_map,
                "ref_cam_extr": ref_cam_extr,
                "target_cam_extr": target_cam_extr,
                "ref_cam_intr": ref_cam_intr,
                "target_cam_intr": target_cam_intr,
            }

            bytes_io = io.BytesIO()
            pickle.dump(data, bytes_io, pickle.HIGHEST_PROTOCOL)
            bytes_io.seek(0)

            try:
                resp = requests.post(f"http://127.0.0.1:{self.docker_port}/predict", files={"file": bytes_io})
                # all_pred_ref_tform_src = resp.json()['obj2_tform_obj1']
                resp_json = resp.json()
                pred_ref_tform_srcs = resp_json['obj2_tform_obj1']
                all_imgs = resp_json['all_imgs']
                all_imgs = torch.Tensor(all_imgs)

                #show_imgs(all_imgs.permute(0, 3, 1, 2))

                pred_ref_tform_srcs = torch.Tensor(pred_ref_tform_srcs).to(device=self.device)
            except Exception as e:
                logger.info(f'failed to retrieve predicted `ref_tform_source`')
                logger.info(e)
                pred_ref_tform_srcs = torch.eye(4)[None,].repeat(B, 1, 1).to(device=self.device) * 0.

            cam_tform4x4_obj = tform4x4(batch.cam_tform4x4_obj, inv_tform4x4(pred_ref_tform_srcs))

            all_cam_tform4x4_obj.append(cam_tform4x4_obj)
            diff_rot_angle_rad = get_pose_diff_in_rad(pred_tform4x4=cam_tform4x4_obj, gt_tform4x4=batch.cam_tform4x4_obj)

            logger.info(diff_rot_angle_rad)
            all_rot_diff_rad.append(diff_rot_angle_rad)

        # r x b
        all_rot_diff_rad = torch.stack(all_rot_diff_rad, dim=0)
        results['rot_diff_rad'] = all_rot_diff_rad.permute(1, 0)
        for b in range(B):
            prefix = f"{self.categories[batch.category_id[b].item()]}_"
            if f'{prefix}rot_diff_rad' not in results.keys():
                results[f'{prefix}rot_diff_rad'] = all_rot_diff_rad.T[b][None,]  # [] torch.stack(all_rot_diff_rad, dim=0).permute(1, 0)
            else:
                results[f'{prefix}rot_diff_rad'] = torch.cat([results[f'{prefix}rot_diff_rad'], all_rot_diff_rad.T[b][None,]], dim=0)

        results['label_gt'] = batch.category_id
        #results['label_pred'] = pred_class_ids
        #results['label_names'] = self.config.classes
        #results['sim'] = sim
        results['cam_tform4x4_obj'] = all_cam_tform4x4_obj[0] #  random.choice(all_cam_tform4x4_obj)
        results['item_id'] = batch.item_id
        results['name_unique'] = batch.name_unique

        B = len(batch)

        return results
