import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass

from od3d.datasets.co3d.frame import CO3D_Frame
from od3d.datasets.sequence import OD3D_SequenceMeshMixin, OD3D_MESH_TYPES, OD3D_PCL_TYPES, OD3D_Sequence, \
    OD3D_SequenceCategoryMixin, OD3D_SequenceSfMMixin, OD3D_SEQUENCE_SFM_TYPES
from od3d.datasets.sequence_meta import OD3D_SequenceMeta, OD3D_SequenceMetaCategoryMixin

from od3d.datasets.object import OD3D_MaskTypeMixin, OD3D_CamTform4x4ObjTypeMixin, OD3D_DepthTypeMixin
from od3d.datasets.co3d.enum import MAP_CATEGORIES_CO3D_TO_OD3D
from pathlib import Path
from co3d.dataset.data_types import (
    load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
)

@dataclass
class CO3D_SequenceMeta(OD3D_SequenceMetaCategoryMixin, OD3D_SequenceMeta):
    pcl_pts_count: int
    pcl_quality_score: float
    rfpath_pcl: Path
    viewpoint_quality_score: float

    @staticmethod
    def load_from_raw(sequence_annotation: SequenceAnnotation):
        name = sequence_annotation.sequence_name
        category = sequence_annotation.category
        if sequence_annotation.point_cloud is not None:
            rfpath_pcl = sequence_annotation.point_cloud.path
            pcl_pts_count = sequence_annotation.point_cloud.n_points
            pcl_quality_score = sequence_annotation.point_cloud.quality_score
        else:
            rfpath_pcl = Path('None')
            pcl_pts_count = 0
            pcl_quality_score = float('nan')

        viewpoint_quality_score = sequence_annotation.viewpoint_quality_score

        return CO3D_SequenceMeta(category=category, name=name, pcl_pts_count=pcl_pts_count,
                                 pcl_quality_score=pcl_quality_score, rfpath_pcl=rfpath_pcl,
                                 viewpoint_quality_score=viewpoint_quality_score)

@dataclass
class CO3D_Sequence(OD3D_SequenceMeshMixin, OD3D_SequenceCategoryMixin,
                       OD3D_DepthTypeMixin, OD3D_MaskTypeMixin, OD3D_CamTform4x4ObjTypeMixin, OD3D_Sequence):
    frame_type = CO3D_Frame
    map_categories_to_od3d = MAP_CATEGORIES_CO3D_TO_OD3D
    meta_type = CO3D_SequenceMeta


    #
    # @property
    # def fname_sfm_pcl(self):
    #     return 'pointcloud.ply'


# from os import pread
#
# import open3d.geometry
# import torchvision.io
#
# from od3d.datasets.co3d.enum import CUBOID_SOURCES, CAM_TFORM_OBJ_SOURCES, CO3D_CATEGORIES, FEATURE_TYPES, REDUCE_TYPES, PCL_SOURCES
# from od3d.datasets.enum import OD3D_CATEGORIES_SIZES_IN_M
# from od3d.datasets.co3d.enum import MAP_CATEGORIES_CO3D_TO_OD3D
# from od3d.datasets.co3d.frame import CO3D_Frame, CO3D_FrameMeta
# from od3d.datasets.frame import OD3D_FrameSequenceMixin
# from od3d.cv.geometry.mesh import Mesh, Meshes
# from od3d.cv.io import read_pts3d_colors, read_pts3d, write_pts3d_with_colors, write_pts3d_with_colors_and_normals, read_pts3d_with_colors_and_normals
# from tqdm import tqdm
# import logging
# logger = logging.getLogger(__name__)
# import torch.utils.data
# from typing import List
# import os
# from od3d.cv.geometry.transform import plane4d_to_tform4x4, tform4x4_from_transl3d
# import numpy as np
# from od3d.cv.visual.resize import resize
# from od3d.io import read_dict_from_yaml, write_dict_as_yaml
#
# import torch.utils.data
# import od3d.io
#
# import subprocess
#
# from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES, OD3D_Frame
#
# from co3d.dataset.data_types import (
#     load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
# )
# from od3d.cv.geometry.transform import se3_exp_map, tform4x4, transf4x4_from_rot3x3_and_transl3, tform4x4_broadcast
# from od3d.cv.geometry.transform import rot3x3
#
# from typing import List
# import torch
# from pathlib import Path
# from od3d.cv.visual.show import show_img
# from od3d.cv.io import load_ply, save_ply
#
#
# from od3d.cv.visual.sample import sample_pxl2d_pts
#
# from od3d.cv.geometry.points_alignment import get_pca_tform_world
# from od3d.cv.geometry.transform import transf3d_broadcast, tform4x4, inv_tform4x4, proj3d2d_broadcast
# from od3d.cv.geometry.primitives import Cuboids
# from od3d.cv.geometry.points_alignment import icp
# from od3d.cv.geometry.fit.cuboid import fit_cuboid_to_pts3d
#
# from dataclasses import dataclass
# import torch.utils.data
# from od3d.cv.geometry.downsample import voxel_downsampling, random_sampling
# import od3d.io
# import cv2
# from od3d.io import run_cmd
# from od3d.cv.visual.show import show_imgs
# from od3d.cv.geometry.transform import depth2pts3d_grid, depth2normals_grid
# import re
# from od3d.cv.visual.show import show_scene
#
# from od3d.cv.geometry.fit.rays_center3d import fit_rays_center3d
#
# from od3d.datasets.sequence_meta import OD3D_SequenceMeta, OD3D_SequenceMetaCategoryMixin
# @dataclass
# class CO3D_SequenceMeta(OD3D_SequenceMetaCategoryMixin, OD3D_SequenceMeta):
#     pcl_pts_count: int
#     pcl_quality_score: float
#     rfpath_pcl: Path
#     viewpoint_quality_score: float
#
#     @staticmethod
#     def load_from_raw(sequence_annotation: SequenceAnnotation):
#         name = sequence_annotation.sequence_name
#         category = sequence_annotation.category
#         if sequence_annotation.point_cloud is not None:
#             rfpath_pcl = sequence_annotation.point_cloud.path
#             pcl_pts_count = sequence_annotation.point_cloud.n_points
#             pcl_quality_score = sequence_annotation.point_cloud.quality_score
#         else:
#             rfpath_pcl = Path('None')
#             pcl_pts_count = 0
#             pcl_quality_score = float('nan')
#
#         viewpoint_quality_score = sequence_annotation.viewpoint_quality_score
#
#         return CO3D_SequenceMeta(name=name, category=category, rfpath_pcl=rfpath_pcl,
#                                  pcl_pts_count=pcl_pts_count, pcl_quality_score=pcl_quality_score,
#                                  viewpoint_quality_score=viewpoint_quality_score)
#
#
#
# class CO3D_Sequence():
#
#     def __init__(self, path_raw: Path, path_preprocess: Path, path_meta: Path, meta: CO3D_SequenceMeta,
#                  modalities: List[OD3D_FRAME_MODALITIES], categories: List[str],
#                  cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D.value,
#                  cuboid_source=CUBOID_SOURCES.DEFAULT.value,
#                  mesh_feats_type=FEATURE_TYPES.DINOV2_AVG.value,
#                  dist_verts_mesh_feats_reduce_type=REDUCE_TYPES.MIN.value,
#                  pcl_source=PCL_SOURCES.CO3D.value,
#                  aligned_name:str =None,
#                  mesh_name: str='default',
#                  ):
#         self.path_raw: Path = path_raw
#         self.path_preprocess: Path = path_preprocess
#         self.path_meta: Path = path_meta
#         self.meta = meta
#         self.cam_tform_obj_source = cam_tform_obj_source
#         self.aligned_name = aligned_name
#         self.mesh_name = mesh_name
#         self.mesh_feats_type = mesh_feats_type
#         self.pcl_source = pcl_source
#         self.dist_verts_mesh_feats_reduce_type = dist_verts_mesh_feats_reduce_type
#         self.cuboid_source = cuboid_source
#         self.modalities = modalities
#         self.categories = categories
#         self.category_id = categories.index(self.category)
#         self._mesh_feats = None
#         self._mesh_feats_viewpoints = None
#         self._pcl = None
#         self._pcl_colors = None
#         self._pcl_clean = None
#         self._mesh = None
#         self._front_name = None
#         self._cuboid_front_tform4x4_obj = None
#         self._cuboid = None
#         self._meta_ext = None
#
#         if torch.cuda.is_available():
#             self.device = 'cuda:0'
#         else:
#             self.device = 'cpu'
#         self.dtype = torch.float
#
#     @property
#     def name(self):
#         return self.meta.name
#
#     @property
#     def name_unique(self):
#         return self.meta.name_unique
#
#     @property
#     def category(self):
#         return self.meta.category
#
#     @property
#     def fpath_cuboid_limits3d(self):
#         return self.path_preprocess.joinpath("labels", "cuboid_limits3d", f"{self.name_unique}.pt")
#
#     def preprocess_cuboid_limits3d(self, override=False):
#         import open3d as o3d
#         import numpy as np
#
#         if override or not self.fpath_cuboid_limits3d.exists():
#             pcl = self.pcl_clean
#
#             #if self.fpath_cuboid.exists() and self.fpath_cuboid_front_tform4x4_obj.exists():
#             #    logger.info(f'press "s"  to skip this lable')
#             #    k = self.cuboid.visualize(pcl=transf3d_broadcast(pts3d=pcl, transf4x4=self.cuboid_front_tform4x4_obj))
#             #    if k == ord('s'):
#             #        return
#
#             # Create an Open3D PointCloud object
#             #pcd = o3d.geometry.PointCloud()
#
#             # Set the point cloud data
#             #pcd.points = o3d.utility.Vector3dVector(pcl.numpy())
#
#             logger.info("")
#             logger.info(
#                 "1) Please pick left, right, back, front, top, bottom [shift + left click]"
#             )
#             logger.info("   Press [shift + right click] to undo point picking")
#             logger.info("2) Afther picking points, press q for close the window")
#             vis = o3d.visualization.VisualizerWithEditing()
#             vis.create_window()
#             #vis.add_geometry(pcd)
#             pcd = o3d.io.read_point_cloud(str(self.fpath_pcl_co3d))
#
#             # vis.add_geometry(pcd)
#
#             if self.fpath_cuboid.exists() and self.fpath_cuboid_front_tform4x4_obj.exists():
#                 verts3d = self.cuboid.get_verts_with_mesh_id(mesh_id=0)
#                 ncds = self.cuboid.get_verts_ncds_with_mesh_id(mesh_id=0)
#                 cuboid_pcd = o3d.geometry.PointCloud()
#                 # from od3d.cv.geometry.transform import inv_tform4x4
#                 cuboid_pcd.points = o3d.utility.Vector3dVector(transf3d_broadcast(pts3d=verts3d, transf4x4=inv_tform4x4(self.cuboid_front_tform4x4_obj)).numpy())
#                 cuboid_pcd.colors = o3d.utility.Vector3dVector(ncds.numpy())
#                 vis.add_geometry(pcd + cuboid_pcd)
#
#                 #logger.info(f'press "s"  to skip this lable')
#                 #k = self.cuboid.visualize(pcl=transf3d_broadcast(pts3d=pcl, transf4x4=self.cuboid_front_tform4x4_obj))
#                 #if k == ord('s'):
#                 #    return
#             else:
#                 vis.add_geometry(pcd)
#
#
#             vis.run()  # user picks points
#             vis.destroy_window()
#             logger.info("")
#             limits3d_ids = vis.get_picked_points()
#
#             if len(limits3d_ids) < 6:
#                 logger.warning("No labels saved due to less than 6 points selected.")
#                 return
#
#             if (torch.Tensor(limits3d_ids) > len(pcd.points)).any():
#                 logger.warning("No labels saved due to point on prev. cuboid selected")
#                 return
#
#
#             limits3d = torch.from_numpy(np.asarray(pcd.points)).to(dtype=torch.float32)[limits3d_ids]
#             self.fpath_cuboid_limits3d.parent.mkdir(parents=True, exist_ok=True)
#             torch.save(limits3d, self.fpath_cuboid_limits3d)
#
#
#     @property
#     def cuboid_limits3d(self):
#         if not self.fpath_cuboid_limits3d.exists():
#             self.preprocess_cuboid_limits3d()
#         cuboid_limits3d = torch.load(self.fpath_cuboid_limits3d).to(dtype=torch.float32)
#         return cuboid_limits3d
#
#     @property
#     def pcl(self):
#         if self._pcl is None:
#             fpath_pcl = self.fpath_pcl_co3d
#             verts, _ = load_ply(str(fpath_pcl))
#             self._pcl = verts
#
#         return self._pcl
#
#     def get_fpath_pcl(self, pcl_source: PCL_SOURCES):
#         if pcl_source == PCL_SOURCES.CO3D:
#             fpath = self.fpath_pcl_co3d
#         else:
#             fpath = self.path_preprocess.joinpath('pcl', pcl_source, self.category, self.name, 'pcl.ply')
#         return fpath
#
#     def get_pcl(self, pcl_source: PCL_SOURCES=None):
#         if pcl_source is None:
#             pcl_source = self.pcl_source
#         fpath = self.get_fpath_pcl(pcl_source=pcl_source)
#         if fpath is None:
#             logger.warning(f'Unknown pcl source {pcl_source}')
#             return None
#         return read_pts3d(fpath)
#
#
#     def get_pcl_colors(self, pcl_source: PCL_SOURCES=None):
#         if pcl_source is None:
#             pcl_source = self.pcl_source
#         fpath = self.get_fpath_pcl(pcl_source=pcl_source)
#         if fpath is None:
#             logger.warning(f'Unknown pcl source {pcl_source}')
#             return None
#         return read_pts3d_colors(fpath)
#
#
#     @property
#     def pcl_colors_co3d(self):
#         if self._pcl_colors is None:
#             self._pcl_colors = read_pts3d_colors(self.fpath_pcl_co3d)
#         return self._pcl_colors
#
#     @property
#     def fpath_pcl_co3d(self):
#         return self.path_raw.joinpath(self.meta.rfpath_pcl)
#
#     @property
#     def fpath_pcl_droid_slam(self):
#         return self.path_preprocess.joinpath('pcl', 'droid_slam', self.category, self.name, 'pcl.ply')
#
#     @property
#     def fpath_pcl_droid_slam_clean(self):
#         return self.path_preprocess.joinpath('pcl', 'droid_slam_clean', self.category, self.name, 'pcl.ply')
#
#     @property
#     def cuboid_labeled(self):
#         return self.fpath_cuboid.exists()
#
#     @property
#     def cuboid_front_tform4x4_obj_labeled(self):
#         return self.fpath_cuboid_front_tform4x4_obj.exists()
#
#     @property
#     def fpath_cuboid_front_tform4x4_obj(self):
#         return self.path_preprocess.joinpath('cuboid_front_tform4x4_obj', self.cuboid_source, self.category, self.name, 'tform4x4.pt')
#
#     # def get_cuboid_front_tform4x4_obj(self, cuboid_source: CUBOID_SOURCES):#
#         #
#         # if cuboid_source == CUBOID_SOURCES.FRONT_FRAME_AND_PCL:
#         #     pass
#         #
#         # elif cuboid_source == CUBOID_SOURCES.KPTS2D_ORIENT_AND_PCL:
#         #
#         #     from od3d.cv.geometry.transform import reproj2d3d_broadcast, so3_log_map, so3_exp_map
#         #     from od3d.cv.visual.draw import draw_pixels
#         #     from od3d.cv.geometry.fit.axis3d_from_pxl2d import axis3d_from_pxl2d
#         #     from od3d.cv.visual.show import show_img
#         #     # show_img(draw_pixels(self.rgb, pxls=self.kpts2d_orient['back-front'][1][:]))
#         #     self.cuboid_source
#         #     self._cam_tform4x4_obj[:3, :3] = axis3d_from_pxl2d(kpts2d_orient=self.kpts2d_orient, cam_intr4x4=self.cam_intr4x4) #  orients
#         #
#
#     """
#     @property
#     def front_name(self):
#         if self._front_name is None:
#             fpath_front_name = self.fpath_front_name
#             if not fpath_front_name.exists():
#                 self.preprocess_front_name()
#             self._front_name = od3d.io.read_str_from_file(fpath_front_name)
#         return self._front_name
#     """
#
#     @property
#     def cam_front_tform4x4_obj(self):
#         return torch.Tensor(self.front_frame.meta.l_cam_tform4x4_obj)
#
#     @property
#     def cam_first_tform4x4_obj(self):
#         return torch.Tensor(self.first_frame.meta.l_cam_tform4x4_obj)
#
#     def get_frames(self, frames_ids=None):
#         if frames_ids is None:
#             frames_ids = list(range(self.frames_count))
#         frames = [self.get_frame_by_index(frame_id) for frame_id in frames_ids]
#         return frames
#
#     def get_frame_by_index(self, index: int):
#         return self.get_frame_by_name(self.frames_names[index])
#
#     def get_frame_by_name(self, frame_name: str):
#         try:
#             frame_meta = CO3D_FrameMeta.load_from_meta_with_rfpath(path_meta=self.path_meta,
#                                                                    rfpath=CO3D_FrameMeta.
#                                                                    get_rfpath_frame_meta_with_category_sequence_and_frame_name(
#                                                                        category=self.category, sequence_name=self.name,
#                                                                        name=frame_name))
#             frame = CO3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
#                                meta=frame_meta, modalities=self.modalities, categories=self.categories,
#                                cam_tform_obj_source=self.cam_tform_obj_source, cuboid_source=self.cuboid_source,
#                                aligned_name=self.aligned_name, mesh_name=self.mesh_name, pcl_source=self.pcl_source)
#         except Exception as e:
#             logger.warning(f'could not retrieve frame {self.path_meta.joinpath(CO3D_FrameMeta.get_rfpath_frame_meta_with_category_sequence_and_frame_name(category=self.category, sequence_name=self.name, name=frame_name))}')
#             frame = None
#         return frame
#
#     def get_sequence_by_category_and_name(self, category: str, name: str):
#         sequence_meta = CO3D_SequenceMeta.load_from_meta_with_category_and_name(path_meta=self.path_meta,
#                                                                                 category=category, name=name)
#         return CO3D_Sequence(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
#                              meta=sequence_meta, modalities=self.modalities, categories=self.categories,
#                              aligned_name=self.aligned_name,
#                              mesh_feats_type=self.mesh_feats_type,
#                              dist_verts_mesh_feats_reduce_type=self.dist_verts_mesh_feats_reduce_type,
#                              cuboid_source=self.cuboid_source,
#                              cam_tform_obj_source=self.cam_tform_obj_source, pcl_source=self.pcl_source,
#                              mesh_name=self.mesh_name)
#
#     @property
#     def front_frame(self):
#         return self.get_frame_by_name(frame_name=self.front_name)
#
#     @property
#     def frames_names(self):
#         frames_names = CO3D_FrameMeta.get_frames_names_of_category_sequence(path_meta=self.path_meta, category=self.category, sequence_name=self.name)
#         return frames_names
#
#     @property
#     def frames_count(self):
#         return len(self.frames_names)
#
#     @property
#     def first_frame(self):
#         #first_frame_fpath = sorted(CO3D_FrameMeta.get_path_frames_meta_with_category_sequence(path_meta=self.path_meta,
#         #                                                                                      category=self.category,
#         #                                                                                      sequence=self.name).iterdir(),
#         #                           key=lambda p: int(p.stem))[0]
#         return self.get_frame_by_index(index=0)
#
#     @property
#     def cuboid_front_tform4x4_obj(self):
#         if self._cuboid_front_tform4x4_obj is None:
#             if not self.fpath_cuboid_front_tform4x4_obj.exists():
#                 self.preprocess_cuboid()
#             self._cuboid_front_tform4x4_obj = torch.load(f=str(self.fpath_cuboid_front_tform4x4_obj))
#         return self._cuboid_front_tform4x4_obj
#
#     def preprocess_pcl_clean(self, override=False):
#         fpath_pcl_clean = self.fpath_pcl_clean
#         if override or not fpath_pcl_clean.exists():
#             from od3d.datasets.co3d.dataset import CO3D
#             fpath_pcl_clean.parent.mkdir(parents=True, exist_ok=True)
#
#             pts3d_max_count = 20000
#             pts3d_prob_thresh = 0.6
#
#             dataset = CO3D(name='co3d', modalities=[OD3D_FRAME_MODALITIES.RGB, OD3D_FRAME_MODALITIES.MASK, OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJ, OD3D_FRAME_MODALITIES.CAM_INTR4X4],
#                            path_raw=self.path_raw, path_preprocess=self.path_preprocess,
#                            categories=[CO3D_CATEGORIES(self.category).value], dict_nested_frames={self.category: {self.name: None}},
#                            cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D.value,
#                            mesh_name=self.mesh_name, pcl_source=self.pcl_source)
#
#             dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=False,
#                                                      collate_fn=dataset.collate_fn,
#                                                      num_workers=0)
#
#             pts3d = self.pcl
#
#             pts3d = random_sampling(pts3d, pts3d_max_count=pts3d_max_count * 3)
#             pts3d = voxel_downsampling(pts3d, K=pts3d_max_count)
#             pts3d_prob = torch.zeros(size=(pts3d.shape[0], 1), device=pts3d.device, dtype=pts3d.dtype)
#
#             if torch.cuda.is_available():
#                 pts3d = pts3d.to(device='cuda:0')
#                 pts3d_prob = pts3d_prob.to(device='cuda:0')
#             for frames in tqdm(iter(dataloader)):
#                 if torch.cuda.is_available():
#                     frames.to(device='cuda:0')
#                 pxl2d = proj3d2d_broadcast(pts3d=pts3d, proj4x4=frames.cam_proj4x4_obj[:, None])
#                 pts3d_prob += sample_pxl2d_pts(frames.mask, pxl2d=pxl2d, padding_mode='zeros').sum(dim=0)
#
#                 # pxl2d = proj3d2d_broadcast(pts3d=pts3d_co3d, proj4x4=frame.cam_proj4x4_obj)
#                 # gb_with_mask_with_pts3d = draw_pixels(img=blend_rgb(frame.rgb, frame.mask*255.), pxls=pxl2d)
#                 # show_img(rgb_with_mask_with_pts3d)
#
#             pts3d_prob = pts3d_prob / len(dataset)
#             # pts3d_co3ds = []
#             # for prob in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:  #0.5, 0.6, 0.7, 0.8, 0.9, 0.95
#             #    pts3d = torch.zeros_like(pts3d_co3d)
#             #    pts3d[pts3d_co3d_prob[..., 0] > prob] = pts3d[pts3d_prob[..., 0] > prob]
#             #    pts3d_co3ds.append(pts3d)
#             # show_pcl(torch.stack(pts3d_co3ds, dim=0))
#             # show_pcl(pts3d[pts3d_prob[..., 0] > 0.6])
#             pts3d_clean = pts3d[pts3d_prob[..., 0] > pts3d_prob_thresh]
#             # frame0: CO3D_Frame = seq_dataset[0]
#             # show_img(frame0.rgb)
#             # show_pcl(pts3d_clean, cam_tform4x4_obj=frame0.cam_tform4x4_obj, cam_intr4x4=frame0.cam_intr4x4, img_size=frame0.size)
#
#             save_ply(fpath_pcl_clean, pts3d_clean)
#
#             del dataloader
#
#     @property
#     def fpath_front_name(self):
#         return self.path_preprocess.joinpath('front_names', self.category, self.name, 'front_name.yaml')
#     @property
#     def fpath_cuboid(self):
#         return self.path_preprocess.joinpath('cuboids', self.cuboid_source, self.category, self.name + '.ply')
#
#     @property
#     def path_mesh_feats(self):
#         return self.path_preprocess.joinpath('mesh_feats')
#
#
#
#     @property
#     def fpath_mesh_feats(self):
#         return self.path_mesh_feats.joinpath(self.pcl_source, self.mesh_name, self.mesh_feats_type, self.name_unique, 'mesh_feats.pt')
#
#     @property
#     def fpath_mesh_feats_viewpoints(self):
#         return self.path_mesh_feats.joinpath(self.pcl_source, self.mesh_name, self.mesh_feats_type, self.name_unique, 'mesh_feats_viewpoints.pt')
#
#
#     @property
#     def path_dist_verts_mesh_feats(self):
#         return self.path_preprocess.joinpath('dist_verts_mesh_feats')
#
#     def get_fpath_dist_verts_mesh_feats(self, sequence):
#         return self.path_dist_verts_mesh_feats.joinpath(self.pcl_source, self.mesh_name, self.mesh_feats_type, self.dist_verts_mesh_feats_reduce_type, self.name_unique, sequence.name_unique, 'dist_verts_mesh_feats.pt')
#
#     def get_fpath_dist_viewpoints_matched_mesh_feats(self, sequence):
#         return self.path_dist_verts_mesh_feats.joinpath(self.pcl_source, self.mesh_name, self.mesh_feats_type, self.dist_verts_mesh_feats_reduce_type, self.name_unique, sequence.name_unique, 'dist_viewpoints_matched_mesh_feats.pt')
#
#
#     def get_pcl_clean_with_masks(self, pcl, masks, cams_tform4x4_obj, cams_intr4x4, pts3d_prob_thresh=0.6, pts3d_count_min=10, pts3d_max_count=20000, return_mask=False):
#         """
#         Args:
#
#         Returns:
#
#         """
#         #cams_proj4x4_obj = tform4x4(cams_intr4x4, cams_tform4x4_obj)
#         pcl_sampled, mask_pcl = random_sampling(pcl, pts3d_max_count=pts3d_max_count, return_mask=True)
#
#         pts3d_prob = torch.zeros(size=(pcl_sampled.shape[0], 1), device=pcl.device, dtype=pcl.dtype)
#         cam_tform_pts3d = transf3d_broadcast(pts3d=pcl_sampled, transf4x4=cams_tform4x4_obj[:, None])
#         pxl2d = proj3d2d_broadcast(pts3d=cam_tform_pts3d, proj4x4=cams_intr4x4[:, None])
#         pts3d_prob += ((cam_tform_pts3d[:, :, 2:] > 0.) * sample_pxl2d_pts(masks, pxl2d=pxl2d, padding_mode='value', padding_value=0.)).sum(dim=0)
#         pts3d_prob = pts3d_prob / len(masks)
#         mask_pcl[mask_pcl == True] *= pts3d_prob[..., 0] > pts3d_prob_thresh
#         pcl_clean = pcl[mask_pcl]
#
#         if len(pcl_clean) < pts3d_count_min:
#             logger.warning(f'less than {pts3d_count_min} in cleaned pcl {self.name_unique}')
#             pcl_clean, mask_pcl = random_sampling(pcl, pts3d_max_count=pts3d_count_min, return_mask=True)
#
#         if not return_mask:
#             return pcl_clean
#         else:
#             return pcl_clean, mask_pcl
#
#     def preprocess_mesh_feats(self, override=False):
#         # fpath_mesh_feats
#         from od3d.datasets.co3d import CO3D
#         from od3d.models.model import OD3D_Model
#         from od3d.cv.transforms.transform import OD3D_Transform
#         from od3d.cv.transforms.sequential import SequentialTransform
#         import re
#
#         if not override and self.fpath_mesh_feats.exists() and self.fpath_mesh_feats_viewpoints.exists():
#             logger.info(f'mesh feats already exist at {self.fpath_mesh_feats}')
#             return
#
#         if override and (self.fpath_mesh_feats.exists() or self.fpath_mesh_feats_viewpoints.exists()):
#             logger.info(f'overriding mesh feats at {self.fpath_mesh_feats}')
#             self.remove_mesh_feats_preprocess_dependent_files()
#
#         dataset = CO3D(name='co3d', modalities=[OD3D_FRAME_MODALITIES.RGB, OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJ,
#                                                 OD3D_FRAME_MODALITIES.CAM_INTR4X4],
#                        path_raw=self.path_raw, path_preprocess=self.path_preprocess,
#                        categories=[CO3D_CATEGORIES(self.category).value],
#                        dict_nested_frames={self.category: {self.name: None}},
#                        cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.PCL,
#                        pcl_source=self.pcl_source,
#                        cuboid_source=CUBOID_SOURCES.DEFAULT,
#                        mesh_feats_type=self.mesh_feats_type,
#                        mesh_name=self.mesh_name)
#
#         dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=False,
#                                                  collate_fn=dataset.collate_fn,
#                                                  num_workers=0)
#
#         if torch.cuda.is_available():
#             device = 'cuda:0'
#         else:
#             device = 'cpu'
#
#         # e.g.: 'M_dinov2_frozen_base_T_centerzoom512_R_acc'
#         match = re.match(r"M_([a-z0-9_]+)_T_([a-z0-9_]+)_R_([a-z0-9_]+)", self.mesh_feats_type, re.I)
#         if match and len(match.groups()) == 3:
#             model_name, transform_name, reduce_type = match.groups()
#         else:
#             msg = f'could not retrieve model, transform, and reduce type from mesh feats type {self.mesh_feats_type}'
#             raise Exception(msg)
#
#         # if self.mesh_feats_type == FEATURE_TYPES.
#         model = OD3D_Model.create_by_name(model_name)
#         model.cuda()
#         model.eval()
#         transform = SequentialTransform([OD3D_Transform.create_by_name(transform_name), model.transform])
#
#         down_sample_rate = model.downsample_rate
#         feature_dim = model.out_dim
#
#         meshes = Meshes.load_from_meshes([self.get_mesh(mesh_source=CUBOID_SOURCES.DEFAULT)], device=device)
#         dataset.transform = transform
#
#         ## DEBUG BLOCK START
#         #cams_tform4x4_world, cams_intr4x4, cams_imgs = self.get_cams(CAM_TFORM_OBJ_SOURCES.PCL)
#         #show_scene(meshes=meshes, cams_tform4x4_world=cams_tform4x4_world, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs )
#         ## DEBUG BLOCK END
#
#         meshes_verts_aggregated_features = [torch.zeros((0, feature_dim), device='cpu')] * meshes.verts.shape[0]
#         meshes_verts_aggregated_viewpoints = [torch.zeros((0, 3), device='cpu')] * meshes.verts.shape[0]
#         vertices_count = len(meshes_verts_aggregated_features)
#
#         for batch in tqdm(iter(dataloader)):
#             B = len(batch)
#             batch.to(device=device)
#
#             batch.cam_tform4x4_obj = batch.cam_tform4x4_obj.detach()
#
#             vts2d, vts2d_mask = meshes.verts2d(cams_intr4x4=batch.cam_intr4x4,
#                                                cams_tform4x4_obj=batch.cam_tform4x4_obj,
#                                                imgs_sizes=batch.size, mesh_ids=[0,] * B,
#                                                down_sample_rate=down_sample_rate)
#
#
#             viewpoints3d = transf3d_broadcast(pts3d=-batch.cam_tform4x4_obj[:, None, :3, 3], transf4x4=inv_tform4x4(batch.cam_tform4x4_obj[:, None])).expand(*vts2d_mask.shape, 3)
#
#
#             # verts3d = meshes.get_verts_stacked_with_mesh_ids(mesh_ids=[0,] * B).clone()
#             # show_scene(meshes=meshes, pts3d=verts3d, lines3d=[torch.stack([verts3d, verts3d + normals3d], dim=-2)])
#
#             viewpoints3d = viewpoints3d[vts2d_mask]
#
#             N = vts2d.shape[1]
#
#             # B x C x H x W
#             feats2d_net = model(batch.rgb)
#             H, W = feats2d_net.shape[-2:]
#             xy = torch.stack(
#                 torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device),
#                                indexing='xy'), dim=0)  # HxW
#             noise2d = torch.ones(size=(vts2d.shape[0], 0, 2), device=device)
#
#             # B x F+N x C
#             net_feats = sample_pxl2d_pts(feats2d_net, pxl2d=torch.cat([vts2d, noise2d], dim=1))
#
#             C = net_feats.shape[2]
#             # args: X: Bx3xHxW, keypoint_positions: BxNx2, obj_mask: BxHxW ensures that noise is sampled outside of object mask
#             # returns: BxF+NxC
#
#             # net_feats = net_feats[:, :].reshape(-1, net_feats.shape[-1])
#             batch_vts_ids = meshes.get_verts_and_noise_ids_stacked([0,] * B, count_noise_ids=0)
#
#             # N,
#             batch_vts_ids = torch.cat([batch_vts_ids[:, :N][vts2d_mask], batch_vts_ids[:, N:].reshape(-1)],
#                                       dim=0)
#
#             # N x C
#             net_feats = torch.cat([net_feats[:, :N][vts2d_mask], net_feats[:, N:].reshape(-1, C)], dim=0)
#
#             for b, vertex_id in enumerate(batch_vts_ids):
#                 meshes_verts_aggregated_features[vertex_id] = torch.cat([net_feats[b:b + 1].detach().cpu(), meshes_verts_aggregated_features[vertex_id].detach().cpu()], dim=0)
#                 meshes_verts_aggregated_viewpoints[vertex_id] = torch.cat([viewpoints3d[b:b + 1].detach().cpu(), meshes_verts_aggregated_viewpoints[vertex_id].detach().cpu()], dim=0)
#
#         # for v in range(len(meshes_verts_aggregated_viewpoints)):
#         #     #  - meshes.get_verts_stacked_with_mesh_ids(mesh_ids=[0,] * B)
#         #     from od3d.cv.geometry.transform import transf4x4_from_normal
#         #     normal3d = meshes.normals3d(meshes_ids=[0,])[0, v]
#         #     normal3d_transf_obj = transf4x4_from_normal(normal3d)
#         #     normal3d_transf_obj[:3, 3] = -transf3d_broadcast(meshes.verts[v], normal3d_transf_obj)
#         #     viewpoints3d = meshes_verts_aggregated_viewpoints[v].clone().to(device=device)
#         #     verts3d = meshes.verts.clone()
#         #     viewpoints3d = transf3d_broadcast(viewpoints3d, normal3d_transf_obj)
#         #     verts3d = transf3d_broadcast(verts3d, normal3d_transf_obj)
#         #     show_scene(pts3d=[verts3d, viewpoints3d], lines3d=[torch.Tensor([[[0., 0., 0.], [0., -1., 0.]]])])
#
#         if reduce_type == 'acc':
#             if not self.fpath_mesh_feats.parent.exists():
#                 self.fpath_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
#             torch.save(meshes_verts_aggregated_features, f=self.fpath_mesh_feats)
#             torch.save(meshes_verts_aggregated_viewpoints, f=self.fpath_mesh_feats_viewpoints)
#             meshes_verts_aggregated_features.clear()
#         elif reduce_type == 'avg':
#             if not self.fpath_mesh_feats.parent.exists():
#                 self.fpath_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
#             meshes_verts_aggregated_features_avg = torch.stack([agg_feats.mean(dim=0) for agg_feats in meshes_verts_aggregated_features], dim=0)
#             torch.save(meshes_verts_aggregated_features_avg.detach().cpu(), f=self.fpath_mesh_feats)
#             meshes_verts_aggregated_viewpoints_avg = torch.stack([agg_viewpoints.mean(dim=0) for agg_viewpoints in meshes_verts_aggregated_viewpoints], dim=0)
#             torch.save(meshes_verts_aggregated_viewpoints_avg.detach().cpu(), f=self.fpath_mesh_feats_viewpoints)
#
#             del meshes_verts_aggregated_features_avg
#         elif reduce_type == 'avg_norm':
#             if not self.fpath_mesh_feats.parent.exists():
#                 self.fpath_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
#             meshes_verts_aggregated_features_avg_norm = torch.nn.functional.normalize(torch.stack([agg_feats.mean(dim=0) for agg_feats in meshes_verts_aggregated_features], dim=0), dim=-1)
#             torch.save(meshes_verts_aggregated_features_avg_norm.detach().cpu(), f=self.fpath_mesh_feats)
#             del meshes_verts_aggregated_features_avg_norm
#             meshes_verts_aggregated_viewpoints_avg = torch.stack([agg_viewpoints.mean(dim=0) for agg_viewpoints in meshes_verts_aggregated_viewpoints], dim=0)
#             torch.save(meshes_verts_aggregated_viewpoints_avg.detach().cpu(), f=self.fpath_mesh_feats_viewpoints)
#
#         elif reduce_type == 'min50':
#             if not self.fpath_mesh_feats.parent.exists():
#                 self.fpath_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
#
#             meshes_verts_aggregated_features_padded = torch.nn.utils.rnn.pad_sequence(meshes_verts_aggregated_features, padding_value=torch.nan, batch_first=True)
#             meshes_verts_aggregated_viewpoints_padded = torch.nn.utils.rnn.pad_sequence(meshes_verts_aggregated_viewpoints, padding_value=torch.nan, batch_first=True)
#
#             meshes_verts_aggregated_features_dists = torch.cdist(meshes_verts_aggregated_features_padded, meshes_verts_aggregated_features_padded)
#             c = torch.nanquantile(meshes_verts_aggregated_features_dists, q=0.5, dim=-1)
#             vals, indices = c.nan_to_num(torch.inf).min(dim=-1)
#             from od3d.cv.select import batched_index_select
#
#             mesh_verts_aggregated_features_min50 = batched_index_select(input=meshes_verts_aggregated_features_padded, index=indices[:, None], dim=1)[:, 0]
#             mesh_verts_aggregated_viewpoints_min50 = batched_index_select(input=meshes_verts_aggregated_viewpoints_padded, index=indices[:, None], dim=1)[:, 0]
#
#             torch.save(mesh_verts_aggregated_features_min50.detach().cpu(), f=self.fpath_mesh_feats)
#             torch.save(mesh_verts_aggregated_viewpoints_min50.detach().cpu(), f=self.fpath_mesh_feats_viewpoints)
#
#         else:
#             logger.warning(f'Unknown mesh feature reduce_type {reduce_type}.')
#
#
#
#         del dataloader
#         del dataset
#         del model
#
#     def get_dist_viewpoints_matched_to_other_sequence(self, sequence: 'CO3D_Sequence'):
#         # this requires ground truth camera poses otherwise it is meaning less
#         fpath_dist_viewpoints_matched_mesh_feats = self.get_fpath_dist_viewpoints_matched_mesh_feats(sequence)
#         if fpath_dist_viewpoints_matched_mesh_feats.exists():
#             return torch.load(fpath_dist_viewpoints_matched_mesh_feats)
#         else:
#             if torch.cuda.is_available():
#                 device = 'cuda:0'
#             else:
#                 device = 'cpu'
#
#             seq1_feats_viewpoints = self.feats_viewpoints
#             seq1_feats = self.feats
#             seq2_feats_viewpoints = sequence.feats_viewpoints
#             seq2_feats = sequence.feats
#             seq1_verts = len(seq1_feats_viewpoints)
#             seq2_verts = len(seq2_feats_viewpoints)
#
#             if isinstance(seq1_feats_viewpoints, list):
#
#                 dist_viewpoints_matched_seq1_seq2 = torch.ones(size=(seq1_verts, seq2_verts)).to(device=device)
#
#                 for i in tqdm(range(seq1_verts)):
#                     for j in range(seq2_verts):
#                         verts1_feats_viewpoints = transf3d_broadcast(seq1_feats_viewpoints[i].clone(), self.labeled_cuboid_obj_tform_obj)
#                         verts2_feats_viewpoints = transf3d_broadcast(seq2_feats_viewpoints[j].clone(), sequence.labeled_cuboid_obj_tform_obj)
#
#                         verts1_feats_viewpoints = torch.nn.functional.normalize(verts1_feats_viewpoints, dim=-1).to(device=device)
#                         verts2_feats_viewpoints = torch.nn.functional.normalize(verts2_feats_viewpoints, dim=-1).to(device=device)
#
#                         dists = torch.cdist(seq1_feats[i].to(device=device), seq2_feats[j].to(device=device))
#                         dists_viewpoints = torch.cdist(verts1_feats_viewpoints, verts2_feats_viewpoints) / 2.
#
#                         if dists.numel() == 0:
#                             dist_viewpoints_matched_seq1_seq2[i, j] = 1.
#                         else:
#                             dist_viewpoints_matched_seq1_seq2[i, j] = dists_viewpoints.flatten()[dists.flatten().argmin()]
#             else:
#                 # means we averaged over feats, cannot calculate viewpoint distance in that case
#                 dist_viewpoints_matched_seq1_seq2 = torch.zeros(size=(seq1_verts, seq2_verts)).to(device=device)
#
#             if not fpath_dist_viewpoints_matched_mesh_feats.parent.exists():
#                 fpath_dist_viewpoints_matched_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
#             torch.save(dist_viewpoints_matched_seq1_seq2.detach().cpu(), fpath_dist_viewpoints_matched_mesh_feats)
#             # dists_feats = self.get_dist_verts_mesh_feats_to_other_sequence(sequence=sequence)
#             # from od3d.cv.select import batched_index_select
#             # dists_viewpoints_matched = batched_index_select(dist_viewpoints_matched_seq1_seq2, dim=1, index=dists_feats.argmin(dim=-1)[:, None,])
#             # mesh = self.get_mesh(mesh_source=CUBOID_SOURCES.DEFAULT, device=device)
#             # mesh.rgb = dists_viewpoints_matched.expand(228, 3)
#             # show_scene(meshes=[mesh])
#             del seq1_feats_viewpoints
#             del seq2_feats_viewpoints
#             del seq1_feats
#             del seq2_feats
#             return dist_viewpoints_matched_seq1_seq2
#
#     def get_dist_verts_mesh_feats_to_other_sequence(self, sequence: 'CO3D_Sequence'):
#         fpath_dist_verts_mesh_feats = self.get_fpath_dist_verts_mesh_feats(sequence)
#         if fpath_dist_verts_mesh_feats.exists():
#             return torch.load(fpath_dist_verts_mesh_feats)
#         else:
#             if torch.cuda.is_available():
#                 device = 'cuda:0'
#             else:
#                 device = 'cpu'
#
#             seq1_feats = self.get_feats()
#             seq2_feats = sequence.get_feats()
#
#
#             from od3d.cv.cluster.embed import pca, tsne
#             # VISUALIZATION START
#             # seq12_feats_padded = torch.nn.utils.rnn.pad_sequence(seq1_feats + seq2_feats, batch_first=True, padding_value=torch.nan)
#             # seq12_feats_padded_mask = (~seq12_feats_padded.isnan().all(dim=-1))
#
#             # #seq12_feats_embed = pca(seq12_feats_padded[seq12_feats_padded_mask], C=2)
#             # seq12_feats_embed = tsne(seq12_feats_padded[seq12_feats_padded_mask], C=2)
#             # seq12_feats_embed = seq12_feats_embed.detach().cpu()
#             #
#             #
#             # seq12_colors_padded = torch.zeros(size=seq12_feats_padded.shape[:-1] + (4,), device=device, dtype=torch.float32)
#             # seq12_colors_padded[:, :, 3] = 0.2
#             #
#             # # color code: seq1 or seq2
#             # seq12_colors_padded[:len(seq1_feats), :, 1] = 1.
#             # seq12_colors_padded[len(seq1_feats):, :, 2] = 1.
#             #
#             # # color code: verts normalized
#             # seq1_verts_ncds = self.get_mesh(mesh_source=CUBOID_SOURCES.DEFAULT).verts_ncds.clone()
#             # seq2_verts_ncds = sequence.get_mesh(mesh_source=CUBOID_SOURCES.DEFAULT).verts_ncds.clone()
#             # seq12_colors_padded_verts = seq12_colors_padded.clone()
#             # seq12_colors_padded_verts[:len(seq1_feats), :, :3] = seq1_verts_ncds[:, None, :3]
#             # seq12_colors_padded_verts[len(seq1_feats):, :, :3] = seq2_verts_ncds[:, None, :3]
#             #
#             #
#             # # color code: viewpoint normalized
#             # seq1_feats_viewpoints = self.feats_viewpoints
#             # seq2_feats_viewpoints = sequence.feats_viewpoints
#             # seq12_feats_viewpoints_padded = torch.nn.utils.rnn.pad_sequence(seq1_feats_viewpoints + seq2_feats_viewpoints, batch_first=True,
#             #                                                      padding_value=torch.nan)
#             # seq12_feats_viewpoints_padded = (seq12_feats_viewpoints_padded - seq12_feats_viewpoints_padded[seq12_feats_padded_mask].min(dim=0).values[None, None]) \
#             #                                 / (seq12_feats_viewpoints_padded[seq12_feats_padded_mask].max(dim=0).values[None, None] - seq12_feats_viewpoints_padded[seq12_feats_padded_mask].min(dim=0).values[None, None])
#             #
#             # seq12_colors_padded_viewpoint = seq12_colors_padded.clone()
#             # seq12_colors_padded_viewpoint[:, :, :3] = seq12_feats_viewpoints_padded[:, :, :3]
#             #
#             # from od3d.cv.visual.show import show_scene2d
#             # show_scene2d(pts2d=[seq12_feats_embed[:], seq12_feats_embed[:]],
#             #              pts2d_colors=[seq12_colors_padded_verts[seq12_feats_padded_mask], seq12_colors_padded_viewpoint[seq12_feats_padded_mask]])
#             #
#             # VISUALIZATION END
#
#
#             dist_verts_mesh_feats_reduce_type = self.dist_verts_mesh_feats_reduce_type
#             match = re.match(r"(pca)?([0-9]*)_?([a-z_]*)", dist_verts_mesh_feats_reduce_type, re.I)
#
#             if match:
#                 embed_type, embed_dim, reduce_type = match.groups()
#                 if len(embed_dim) > 0:
#                     embed_dim = int(embed_dim)
#                 print(embed_type, embed_dim, reduce_type)
#             else:
#                 msg = f'could not retrieve embed_type, embed_dim, and reduce type from mesh feats type {dist_verts_mesh_feats_reduce_type}'
#                 raise Exception(msg)
#
#             # perform pca on seq1_feats and seq2_feats and visualize
#             if isinstance(seq1_feats, list):
#                 seq1_verts_count = len(seq1_feats)
#                 seq2_verts_count = len(seq2_feats)
#                 dist_verts_seq1_seq2 = torch.ones(size=(seq1_verts_count, seq2_verts_count)).to(device=device) * torch.inf
#
#                 # Vertices1+2 x Viewpoints x F
#                 seq12_feats_padded = torch.nn.utils.rnn.pad_sequence(seq1_feats + seq2_feats, batch_first=True, padding_value=torch.nan).to(device=device)
#                 F = seq12_feats_padded.shape[-1]
#                 V = seq12_feats_padded.shape[-2]
#                 seq12_feats_padded_mask = (~seq12_feats_padded.isnan().all(dim=-1))
#
#                 if embed_type is None:
#                     pass
#                 elif embed_type == 'pca':
#                     seq12_feats_padded_embed = torch.zeros(size=(seq12_feats_padded.shape[:-1] + (embed_dim,))).to(device=device, dtype=seq12_feats_padded.dtype)
#                     seq12_feats_padded_embed[:] = torch.nan
#                     seq12_feats_padded_embed[seq12_feats_padded_mask] = pca(seq12_feats_padded[seq12_feats_padded_mask], C=embed_dim)
#                     seq12_feats_padded = seq12_feats_padded_embed
#                     F = embed_dim
#                 else:
#                     logger.warning(f'unknown embed type {embed_type}')
#
#                 P = 8 # ensures that 11 GB are enough
#                 for p in range(P):
#                     if p < P-1:
#                         seq1_verts_partial = torch.arange(seq1_verts_count)[
#                                              (seq1_verts_count // P) * p:
#                                              (seq1_verts_count // P) * (p+1)].to(device=device)
#                     else:
#                         seq1_verts_partial = torch.arange(seq1_verts_count)[
#                                              (seq1_verts_count // P) * p:].to(
#                             device=device)
#                     seq1_verts_partial_count = len(seq1_verts_partial)
#                     # logger.info(seq1_verts_partial)
#                     # Vertices1 x Viewpoints x F
#                     seq1_feats_padded = seq12_feats_padded[seq1_verts_partial].clone()
#                     seq2_feats_padded = seq12_feats_padded[seq1_verts_count:].clone()
#
#                     seq1_feats_padded_mask = seq12_feats_padded_mask[seq1_verts_partial].clone()
#                     seq2_feats_padded_mask = seq12_feats_padded_mask[seq1_verts_count:].clone()
#
#                     # Vertices1 x Viewpoints x Vertices2 x Viewpoints
#                     dists_verts_feats_seq1_seq2 = torch.cdist(seq1_feats_padded.reshape(-1, F)[None,], seq2_feats_padded.reshape(-1, F)[None,]).reshape(seq1_verts_partial_count, V, seq2_verts_count, V)
#                     dists_verts_feats_seq1_seq2_mask = (seq1_feats_padded_mask[:, :, None, None] * seq2_feats_padded_mask[None, None, :, :])
#                     dist_verts_seq1_seq2_inf_mask = dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3).flatten(2).sum(
#                        dim=-1) == 0.
#
#                     if reduce_type == REDUCE_TYPES.MIN:
#                        # replace nan values with inf
#                        dists_verts_feats_seq1_seq2 = dists_verts_feats_seq1_seq2.nan_to_num(torch.inf)
#                        dist_verts_seq1_seq2[seq1_verts_partial] = dists_verts_feats_seq1_seq2.permute(0, 2, 1, 3).flatten(2).min(dim=-1).values
#                     elif reduce_type == REDUCE_TYPES.AVG:
#                        dists_verts_feats_seq1_seq2 = dists_verts_feats_seq1_seq2.nan_to_num(0.)
#                        dists_verts_feats_seq1_seq2_mask = dists_verts_feats_seq1_seq2_mask.nan_to_num(0.)
#                        dist_verts_seq1_seq2_partial = (dists_verts_feats_seq1_seq2.permute(0, 2, 1, 3).flatten(2) * dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3).flatten(2)).sum(dim=-1) / \
#                                               (dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3).flatten(2).sum(dim=-1) + 1e-10)
#                        dist_verts_seq1_seq2_partial[dist_verts_seq1_seq2_inf_mask] = torch.inf
#                        dist_verts_seq1_seq2[seq1_verts_partial] = dist_verts_seq1_seq2_partial
#                     elif reduce_type == REDUCE_TYPES.MIN_AVG:
#                        dists_verts_feats_seq1_seq2 = dists_verts_feats_seq1_seq2.nan_to_num(torch.inf)
#                        dists_verts_feats_seq1_seq2_mask = dists_verts_feats_seq1_seq2_mask.nan_to_num(0.)
#                        dist_verts_seq1_seq2_partial = ((
#                                                               (dists_verts_feats_seq1_seq2.permute(0, 2, 1, 3).min(dim=-1).values.nan_to_num(posinf=0.) * dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3)[:, :, :, 0]).sum(dim=-1) +
#                                                               (dists_verts_feats_seq1_seq2.permute(0, 2, 1, 3).min(dim=-2).values.nan_to_num(posinf=0.) * dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3)[:, :, 0, :]).sum(dim=-1)) /
#                                                        (dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3)[:, :, 0, :].sum(dim=-1) + dists_verts_feats_seq1_seq2_mask.permute(0, 2, 1, 3)[:, :, :, 0].sum(dim=-1) + 1e-10))
#                        dist_verts_seq1_seq2_partial[dist_verts_seq1_seq2_inf_mask] = torch.inf
#                        dist_verts_seq1_seq2[seq1_verts_partial] = dist_verts_seq1_seq2_partial
#                     else:
#                        logger.warning(f'Unknown reduce type {reduce_type}.')
#
#                     # for i in tqdm(range(seq1_verts)):
#                     #     for j in range(seq2_verts):
#                     #         dists = torch.cdist(seq1_feats[i].to(device=device), seq2_feats[j].to(device=device))
#                     #         if dists.numel() == 0:
#                     #             dist_verts_seq1_seq2[i, j] = torch.inf
#                     #         else:
#                     #             if reduce_type == REDUCE_TYPES.MIN:
#                     #                 dist_verts_seq1_seq2[i, j] = dists.min()
#                     #             elif reduce_type == REDUCE_TYPES.AVG:
#                     #                 dist_verts_seq1_seq2[i, j] = dists.mean()
#                     #             elif reduce_type == REDUCE_TYPES.MIN_AVG:
#                     #                 dist_verts_seq1_seq2[i, j] = torch.cat([dists.min(dim=-1).values, dists.min(dim=-2).values]).mean()
#                     #             elif reduce_type.startswith('pca'):
#                     #                 logger.info('pca...')
#                     #             else:
#                     #                 logger.warning(f'Unknown reduce type {reduce_type}.')
#                     #         del dists
#                     #
#                     # for f in seq1_feats:
#                     #     del f
#                     # for f in seq2_feats:
#                     #     del f
#             else:
#                 dist_verts_seq1_seq2 = torch.cdist(seq1_feats.to(device=device), seq2_feats.to(device=device))
#             if not fpath_dist_verts_mesh_feats.parent.exists():
#                 fpath_dist_verts_mesh_feats.parent.mkdir(parents=True, exist_ok=True)
#             torch.save(dist_verts_seq1_seq2.detach().cpu(), fpath_dist_verts_mesh_feats)
#             del seq1_feats
#             del seq2_feats
#             return dist_verts_seq1_seq2
#
#
#     @property
#     def feats(self):
#         if self._mesh_feats is None:
#             if not self.fpath_mesh_feats.exists():
#                 self.preprocess_mesh_feats()
#             self._mesh_feats = torch.load(self.fpath_mesh_feats)
#         return self._mesh_feats
#
#     @property
#     def feats_viewpoints(self):
#         if self._mesh_feats_viewpoints is None:
#             if not self.fpath_mesh_feats_viewpoints.exists():
#                 self.preprocess_mesh_feats()
#             self._mesh_feats_viewpoints = torch.load(self.fpath_mesh_feats_viewpoints)
#         return self._mesh_feats_viewpoints
#
#     def get_feats(self):
#         if not self.fpath_mesh_feats.exists():
#             self.preprocess_mesh_feats()
#         mesh_feats = torch.load(self.fpath_mesh_feats)
#         if isinstance(mesh_feats, List):
#             for i in range(len(mesh_feats)):
#                 mesh_feats[i] = mesh_feats[i].detach().cpu()
#         else:
#             mesh_feats = mesh_feats.detach().cpu()
#
#         return mesh_feats
#
#     @property
#     def fpath_pcl_clean(self):
#         return self.path_preprocess.joinpath('pcls', self.category, self.name, 'pcl_clean.ply') #  f'co3d_probthresh_{str(pts3d_prob_thresh).replace(".", "_")}_max_{pts3d_max_count}' + '.ply')
#     @property
#     def pcl_clean(self):
#         if self._pcl_clean is None:
#             fpath_pcl_clean = self.fpath_pcl_clean
#             if not fpath_pcl_clean.exists():
#                 self.preprocess_pcl_clean()
#             self._pcl_clean, _ = load_ply(fpath_pcl_clean)
#         return self._pcl_clean
#
#     def get_trajectory(self, cam_tform_obj_source: CAM_TFORM_OBJ_SOURCES):
#         frames = [self.get_frame_by_name(frame_name) for frame_name in self.frames_names]
#         cams_tform_obj = [f.get_cam_tform4x4_obj(cam_tform_obj_source=cam_tform_obj_source) for f in frames]
#         cams_tform_obj = torch.stack(cams_tform_obj, dim=0)
#         obj_cams_traj = inv_tform4x4(cams_tform_obj)[:, :3, 3]
#         return obj_cams_traj
#
#     def get_a_src_scale_b_src(self, src_a: CAM_TFORM_OBJ_SOURCES, src_b: CAM_TFORM_OBJ_SOURCES, device='cpu'):
#         a_tform_b = self.get_a_src_tform_b_src(src_a=src_a, src_b=src_b, estimate_scale=True, device=device)
#         return a_tform_b[:3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
#
#     def get_fpath_a_src_tform_b_src(self, src_a: CAM_TFORM_OBJ_SOURCES, src_b: CAM_TFORM_OBJ_SOURCES):
#         fpath = self.path_preprocess.joinpath('a_src_tform_b_src', f'{src_a}_{self.pcl_source}_tform_{src_b}_{self.pcl_source}', self.name_unique, f'tform.pt')
#         return fpath
#
#     def get_a_src_tform_b_src(self, src_a: CAM_TFORM_OBJ_SOURCES, src_b: CAM_TFORM_OBJ_SOURCES, estimate_scale=True, device='cpu'):
#         fpath = self.get_fpath_a_src_tform_b_src(src_a=src_a, src_b=src_b)
#         if fpath.exists():
#             a_src_tform_b_src = torch.load(fpath).to(device=device)
#         else:
#             if src_a == src_b:
#                 a_src_tform_b_src = torch.eye(4).to(device=device)
#             else:
#                 logger.info(f'preprocess for {self.name_unique} a_src_tform_b_src {src_a} tform {src_b}')
#                 obj_cams_traj_a = self.get_trajectory(cam_tform_obj_source=src_a).to(device=device)
#                 obj_cams_traj_b = self.get_trajectory(cam_tform_obj_source=src_b).to(device=device)
#                 obj_cams_traj_mask = (~obj_cams_traj_a.isnan().any(dim=-1)) * (~obj_cams_traj_b.isnan().any(dim=-1))
#
#                 if obj_cams_traj_mask.sum() < 4:
#                     msg = f'could not calculate a tform b due to missing trajectory points'
#                     raise NotImplementedError(msg=msg)
#                 #from od3d.cv.visual.show import show_scene
#                 #show_scene(pts3d=[obj_cams_traj_a[50:], obj_cams_traj_b[50:]])
#                 from od3d.cv.geometry.fit.tform4x4 import fit_tform4x4_with_matches3d3d
#                 a_src_tform_b_src = fit_tform4x4_with_matches3d3d(pts=obj_cams_traj_b[obj_cams_traj_mask], pts_ref=obj_cams_traj_a[obj_cams_traj_mask], estimate_scale=estimate_scale)
#
#                 fpath.parent.mkdir(parents=True, exist_ok=True)
#                 torch.save(a_src_tform_b_src.detach().cpu(), fpath)
#                 del obj_cams_traj_a
#                 del obj_cams_traj_b
#         return a_src_tform_b_src
#     #    co3d_src_tform_src = tform4x4(
#     #        inv_tform4x4(src_frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D)),
#     #        src_frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.DROID_SLAM))
#
#     def run_droid_slam(self):
#         stride = "1"
#         image_tag = "limpbot/droid-slam:v1"
#         path_in = self.path_raw.joinpath(self.name_unique, 'images')
#         path_out_root = self.path_preprocess.joinpath('droid_slam')
#         rpath_out = self.name_unique
#         path_out = path_out_root.joinpath(rpath_out)
#
#         fx = self.first_frame.meta.l_cam_intr4x4[0][0]
#         fy = self.first_frame.meta.l_cam_intr4x4[1][1]
#         cx = self.first_frame.meta.l_cam_intr4x4[0][2]
#         cy = self.first_frame.meta.l_cam_intr4x4[1][2]
#         if not path_out.exists():
#             path_out.mkdir(parents=True, exist_ok=True)
#
#         cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
#         if len(cuda_visible_devices) == 0:
#             cuda_visible_devices = 'all'
#         else:
#             cuda_visible_devices = f'{cuda_visible_devices}'
#
#         torch.cuda.empty_cache()
#         run_cmd(cmd=f'echo "{fx} {fy} {cx} {cy}" > {path_out_root}/{rpath_out}/calib.txt', logger=logger)
#         run_cmd(
#             cmd=f'docker run --user=$(id -u):$(id -g) --gpus device={cuda_visible_devices} -e RPATH_OUT={rpath_out} -e STRIDE={stride} -v {path_in}:/home/appuser/in -v {path_out_root}:/home/appuser/DROID-SLAM/reconstructions/out -t {image_tag}',
#             logger=logger, live=True)
#
#     def preprocess_pcl(self, override=False):
#         fpath_pcl = self.get_fpath_pcl(self.pcl_source)
#
#         pts3d_count_min = 10
#         pts3d_max_count = 20000
#         pts3d_prob_thresh = 0.6
#
#
#
#         if not override and fpath_pcl.exists():
#             logger.warning(f'pcl already exists {fpath_pcl}')
#             return
#         else:
#             logger.info(f'preprocessing pcl for {self.name_unique} with type {self.pcl_source}')
#
#         if self.pcl_source == PCL_SOURCES.DROID_SLAM_CLEAN:
#             device = 'cuda'
#             dtype = torch.float
#             cams_tform4x4_obj = torch.load(self.fpath_droid_slam_traj).to(device=device)
#             partial_disps_up = torch.load(self.path_droid_slam.joinpath('disps_up.pt'))[:, None].to(device=device)
#             partial_tstamps = torch.load(self.path_droid_slam.joinpath('tstamps.pt')).to(dtype=torch.long, device=device)
#             h0 = self.first_frame.H
#             w0 = self.first_frame.W
#             partial_disps_up = resize(partial_disps_up, mode="nearest_v2", H_out=h0, W_out=w0)
#
#             partial_cams_tform4x4_obj = cams_tform4x4_obj[partial_tstamps]
#             partial_depths = 1. / partial_disps_up
#             partial_depths = partial_depths.nan_to_num(0., posinf=0., neginf=0.).to(torch.float)
#
#             frames = self.get_frames(frames_ids=partial_tstamps)
#             partial_rgb = torch.stack([frame.rgb for frame in frames], dim=0).to(device=device)
#             partial_mask = torch.stack([frame.mask for frame in frames], dim=0).to(device=device)
#             partial_cams_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=device)
#
#             spts = 1
#
#             partial_depths_valid = (partial_disps_up > partial_disps_up.flatten(2).mean(dim=-1)[
#                 ..., None, None]) * partial_depths.isfinite() * (partial_depths > 0.) * (partial_mask > 0.5)
#             partial_depths[~partial_depths_valid] = 0.
#
#             pts3d = \
#             transf3d_broadcast(depth2pts3d_grid(partial_depths[::spts], partial_cams_intr4x4[::spts]).permute(0, 2, 3, 1),
#                                inv_tform4x4(partial_cams_tform4x4_obj[::spts])[:, None, None])[
#                 partial_depths_valid[::spts][..., 0, :, :]]
#             pts3d_colors = partial_rgb[::spts].permute(0, 2, 3, 1)[partial_depths_valid[::spts][..., 0, :, :]] / 255.
#             pts3d_normals = transf3d_broadcast(
#                 depth2normals_grid(partial_depths[::spts], partial_cams_intr4x4[::spts], shift=w0 // 200).permute(0, 2, 3,
#                                                                                                                   1),
#                 inv_tform4x4(partial_cams_tform4x4_obj[::spts])[:, None, None])[partial_depths_valid[::spts][..., 0, :, :]]
#
#             ## DEBUG BLOCK START
#             # o3d_pcl = open3d.geometry.PointCloud()
#             # o3d_pcl.points = open3d.utility.Vector3dVector(pts3d.detach().cpu().numpy())
#             # o3d_pcl.normals = open3d.utility.Vector3dVector(pts3d_normals.detach().cpu().numpy())  # invalidate existing normals
#             # o3d_pcl.colors = open3d.utility.Vector3dVector(pts3d_colors.detach().cpu().numpy())
#             # open3d.visualization.draw_geometries([o3d_pcl])
#             ## DEBUG BLOCK END
#
#             pts3d_clean, pts3d_clean_mask = self.get_pcl_clean_with_masks(pcl=pts3d, masks=partial_mask,
#                                                                           cams_intr4x4=partial_cams_intr4x4,
#                                                                           cams_tform4x4_obj=partial_cams_tform4x4_obj,
#                                                                           pts3d_prob_thresh=pts3d_prob_thresh,
#                                                                           pts3d_max_count=pts3d_max_count,
#                                                                           pts3d_count_min=pts3d_count_min,
#                                                                           return_mask=True)
#
#             # pts3d_clean, pts3d_clean_mask = self.get_pcl_clean_with_focus_point_and_plane_removal(
#             #     pts3d=pts3d, pts3d_colors=pts3d_colors, cams_tform4x4_obj=cams_tform4x4_obj,
#             #     pts3d_count_min=pts3d_count_min, return_mask=True
#             # )
#
#             pts3d_colors_clean = pts3d_colors[pts3d_clean_mask]
#             pts3d_normals_clean = pts3d_normals[pts3d_clean_mask]
#
#             ## DEBUG BLOCK START
#             # scams = 30
#             # frames = self.get_frames()
#             # rgb = torch.stack([frame.rgb for frame in frames], dim=0).to(device=device)
#             # cams_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=device)
#             # show_scene(pts3d=[pts3d_clean], pts3d_colors=[pts3d_colors_clean], cams_tform4x4_world=cams_tform4x4_obj[::scams], cams_intr4x4=cams_intr4x4[::scams], cams_imgs=rgb[::scams])
#             # o3d_pcl = open3d.geometry.PointCloud()
#             # o3d_pcl.points = open3d.utility.Vector3dVector(pts3d_clean.detach().cpu().numpy())
#             # o3d_pcl.normals = open3d.utility.Vector3dVector(
#             #     pts3d_normals_clean.detach().cpu().numpy())  # invalidate existing normals
#             # o3d_pcl.colors = open3d.utility.Vector3dVector(pts3d_colors_clean.detach().cpu().numpy())
#             # open3d.visualization.draw_geometries([o3d_pcl])
#             ## DEBUG BLOCK END
#         elif self.pcl_source == PCL_SOURCES.CO3D_CLEAN:
#             # missing normals
#             pts3d_clean, pts3d_colors_clean, pts3d_normals_clean = read_pts3d_with_colors_and_normals(self.fpath_pcl_co3d, device=self.device)
#             frames = self.get_frames()
#             cams_tform4x4_obj = torch.stack([frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D) for frame in frames], dim=0).to(device=self.device)
#             mask = torch.stack([frame.mask for frame in frames], dim=0).to(device=self.device)
#             cams_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=self.device)
#             pts3d_clean, pts3d_clean_mask = self.get_pcl_clean_with_masks(pcl=pts3d_clean, masks=mask,
#                                                                           cams_intr4x4=cams_intr4x4,
#                                                                           cams_tform4x4_obj=cams_tform4x4_obj,
#                                                                           pts3d_prob_thresh=pts3d_prob_thresh,
#                                                                           pts3d_max_count=pts3d_max_count,
#                                                                           pts3d_count_min=pts3d_count_min,
#                                                                           return_mask=True)
#             pts3d_colors_clean = pts3d_colors_clean[pts3d_clean_mask]
#             pts3d_normals_clean = torch.zeros_like(pts3d_colors_clean) # pts3d_normals_clean[pts3d_clean_mask]
#             logger.warning('missing normals in CO3Dv2 this setting...')
#
#         else:
#             logger.warning(f'unexpected pcl source to preprocess {self.pcl_source}')
#             raise NotImplementedError
#
#         center3d = pts3d_clean.mean(axis=0)
#         center3d_tform4x4_obj = tform4x4_from_transl3d(-center3d).to(device=self.device)
#
#         center3d_pts3d_clean = transf3d_broadcast(pts3d_clean, center3d_tform4x4_obj)
#
#         ## DEBUG BLOCK START
#         o3d_pcl = open3d.geometry.PointCloud()
#         o3d_pcl.points = open3d.utility.Vector3dVector(pts3d_clean.detach().cpu().numpy())
#         if (pts3d_normals_clean == 0.).all():
#             o3d_pcl.estimate_normals()
#             pts3d_normals_clean = torch.from_numpy(np.array(o3d_pcl.normals)).to(device=pts3d_normals_clean.device, dtype=pts3d_normals_clean.dtype)
#         else:
#             o3d_pcl.normals = open3d.utility.Vector3dVector(
#                 pts3d_normals_clean.detach().cpu().numpy())  # invalidate existing normals
#         o3d_pcl.colors = open3d.utility.Vector3dVector(pts3d_colors_clean.detach().cpu().numpy())
#         # open3d.visualization.draw_geometries([o3d_pcl])
#         ## DEBUG BLOCK END
#
#         # saving...
#         if fpath_pcl.exists():
#             self.remove_pcl_preprocess_dependent_files()
#
#         fpath_pcl.parent.mkdir(parents=True, exist_ok=True)
#         write_pts3d_with_colors_and_normals(fpath=fpath_pcl,
#                                             pts3d=center3d_pts3d_clean.detach().cpu(),
#                                             pts3d_colors=pts3d_colors_clean.detach().cpu(),
#                                             pts3d_normals=pts3d_normals_clean.detach().cpu())
#
#         cams_tform4x4_obj_centered = tform4x4_broadcast(cams_tform4x4_obj, inv_tform4x4(center3d_tform4x4_obj))
#         # save transformation
#         for i, cam_tform4x4_obj in enumerate(cams_tform4x4_obj_centered):
#             frame = self.get_frame_by_index(i)
#             frame.fpath_cam_tform4x4_pcl.parent.mkdir(parents=True, exist_ok=True)
#             torch.save(obj=cam_tform4x4_obj.detach().cpu(), f=frame.fpath_cam_tform4x4_pcl)
#
#
#     def preprocess_mesh(self, override=False):
#         if self.get_fpath_mesh_pcl().exists() and not override:
#             logger.warning(f'mesh already exists {self.get_fpath_mesh_pcl()}')
#             return
#         else:
#             logger.info(f'preprocessing mesh for {self.name_unique} with type {self.mesh_name}')
#
#
#         match = re.match(r"([a-z]+)([0-9]+)", self.mesh_name, re.I)
#         if match and len(match.groups()) == 2:
#             mesh_type, mesh_vertices_count = match.groups()
#             mesh_vertices_count = int(mesh_vertices_count)
#         else:
#             msg = f'could not retrieve mesh type and vertices count from mesh name {self.mesh_name}'
#             raise Exception(msg)
#
#         fpath_pcl = self.get_fpath_pcl(pcl_source=self.pcl_source)
#         if not fpath_pcl.exists():
#             self.preprocess_pcl(override=True)
#
#         if fpath_pcl.exists():
#             pts3d, pts3d_colors, pts3d_normals = read_pts3d_with_colors_and_normals(fpath_pcl)
#         else:
#             logger.warning(f'could not preprocess mesh, due to fpath to pcl {fpath_pcl} does not exists.')
#             return None
#
#         N = pts3d.shape[0]
#         if N < 4:
#             logger.warning(f'Could not estimate mesh for sequence {self.name_unique} due to too few points in raw pcl {N}')
#             return
#
#         o3d_pcl = open3d.geometry.PointCloud()
#         o3d_pcl.points = open3d.utility.Vector3dVector(pts3d.detach().cpu().numpy())
#         o3d_pcl.normals = open3d.utility.Vector3dVector(pts3d_normals.detach().cpu().numpy())  # invalidate existing normals
#         o3d_pcl.colors = open3d.utility.Vector3dVector(pts3d_colors.detach().cpu().numpy())
#
#         ## DEBUG BLOCK START
#         #open3d.visualization.draw_geometries([o3d_pcl])
#         # scams = 30
#         # frames = self.get_frames()
#         # H, W = frames[0].H, frames[0].W
#         # rgb = torch.stack([frame.rgb[:, :int(H * 0.9), :int(W * 0.9)] for frame in frames], dim=0).to(device=self.device)
#         # cams_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=self.device)
#         # cams_tform4x4_obj = torch.stack([frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.PCL) for frame in frames], dim=0).to(device=self.device)
#         # show_scene(pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj[::scams], cams_intr4x4=cams_intr4x4[::scams], cams_imgs=rgb[::scams])
#         ## DEBUG BLOCK END
#
#         # #### OPTION 1: CONVEX HULL
#         if mesh_type == 'convex':
#             o3d_obj_mesh, _ = o3d_pcl.compute_convex_hull()
#             o3d_obj_mesh.compute_vertex_normals()
#             logger.info(o3d_obj_mesh)
#             o3d_obj_mesh = o3d_obj_mesh.remove_unreferenced_vertices()
#             logger.info(o3d_obj_mesh)
#             o3d_obj_mesh = o3d_obj_mesh.simplify_quadric_decimation(mesh_vertices_count)
#             logger.info(o3d_obj_mesh)
#             obj_mesh = Mesh.from_o3d(o3d_obj_mesh, device=self.device)
#
#         elif mesh_type == 'poisson':
#             # #### OPTION 2: POISSON (requires normals)
#
#             o3d_obj_mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(o3d_pcl, depth=9,
#                                                                                                    linear_fit=False)
#             vertices_to_remove = densities < np.quantile(densities, 0.05)
#             o3d_obj_mesh.remove_vertices_by_mask(vertices_to_remove)
#             logger.info(o3d_obj_mesh)
#             o3d_obj_mesh = o3d_obj_mesh.remove_unreferenced_vertices()
#             logger.info(o3d_obj_mesh)
#             o3d_obj_mesh = o3d_obj_mesh.simplify_quadric_decimation(mesh_vertices_count)
#             logger.info(o3d_obj_mesh)
#             obj_mesh = Mesh.from_o3d(o3d_obj_mesh, device=self.device)
#
#         elif mesh_type == 'alpha':
#             # #### OPTION 3: ALPHA_SHAPE
#             pts3d = random_sampling(pts3d, pts3d_max_count=20000)
#             particle_size = torch.cdist(pts3d[None,], pts3d[None,]).quantile(dim=-1, q=5. / len(pts3d)).mean()
#             alpha = 10 * particle_size
#             o3d_obj_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3d_pcl, alpha)
#             logger.info(o3d_obj_mesh)
#             o3d_obj_mesh = o3d_obj_mesh.remove_unreferenced_vertices()
#             logger.info(o3d_obj_mesh)
#             o3d_obj_mesh = o3d_obj_mesh.simplify_quadric_decimation(mesh_vertices_count)
#             logger.info(o3d_obj_mesh)
#             obj_mesh = Mesh.from_o3d(o3d_obj_mesh, device=self.device)
#         elif mesh_type == 'voxel':
#             #### OPTION 4: VOXEL GRID
#             from pytorch3d.ops.marching_cubes import marching_cubes
#             voxel_grid, voxel_grid_range, voxel_grid_offset = voxel_downsampling(pts3d_cls=pts3d, K=mesh_vertices_count * 2,
#                                                                                  return_voxel_grid=True, min_steps=2)
#             # vol_batch(N, D, H, W) ->  (X, Y, Z).permute(2, 0, 1)
#             verts, faces = marching_cubes(vol_batch=voxel_grid.permute(2, 0, 1)[None,] * 1., return_local_coords=True)
#             faces = faces[0].to(device=self.device)
#             verts = (verts[0].to(device=self.device) + 1) / 2.
#
#             obj_mesh = Mesh(verts=voxel_grid_offset[None,] + voxel_grid_range[None,] * verts, faces=faces)
#         else:
#             msg = f'Unknown mesh type {mesh_type}'
#             raise Exception(msg)
#
#         # visualization...
#         ## DEBUG BLOCK START
#         # open3d.visualization.draw(o3d_pcl)
#         # open3d.visualization.draw(o3d_obj_mesh)
#         ## DEBUG BLOCK END
#
#         ## DEBUG BLOCK START
#         # scams = 30
#         # frames = self.get_frames()
#         # H, W = frames[0].H, frames[0].W
#         # rgb = torch.stack([frame.rgb[:, :int(H * 0.9), :int(W * 0.9)] for frame in frames], dim=0).to(device=self.device)
#         # cams_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=self.device)
#         # cams_tform4x4_obj = torch.stack([frame.cam_tform4x4_obj for frame in frames], dim=0).to(device=self.device)
#         # show_scene(meshes=[obj_mesh], pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj[::scams], cams_intr4x4=cams_intr4x4[::scams], cams_imgs=rgb[::scams])
#         # ## DEBUG BLOCK END
#
#         obj_mesh.write_to_file(fpath=self.get_fpath_mesh_pcl())
#
#
#     def get_pcl_clean_with_focus_point_and_plane_removal(self, pts3d, pts3d_colors, cams_tform4x4_obj, pts3d_count_min=10, return_mask=False):
#
#         import numpy as np
#         from od3d.cv.visual.show import show_scene
#         from od3d.cv.geometry.fit.plane4d import fit_plane, score_plane4d_fit
#         from od3d.cv.geometry.transform import plane4d_to_tform4x4, tform4x4_from_transl3d
#         from od3d.cv.optimization.ransac import ransac
#         from od3d.cv.geometry.transform import tform4x4_broadcast
#         from od3d.cv.geometry.mesh import Mesh
#         from functools import partial
#         from od3d.cv.geometry.downsample import random_sampling, voxel_downsampling
#
#         pts3d_raw = pts3d.clone()
#         if len(pts3d) >= pts3d_count_min:
#             # F x 4 x 4
#             #show_scene(pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=[self.first_frame.cam_intr4x4])
#
#             pts3d_range = max(pts3d.max(dim=0)[0] - pts3d.min(dim=0)[0]).item()
#             scene_size = pts3d_range
#             _, mask_pts3d_sampled_rand = random_sampling(pts3d, pts3d_max_count=10000, return_mask=True)
#             mask_pts3d_sampled = mask_pts3d_sampled_rand.clone()
#             # _, mask_pts3d_sampled_voxel = voxel_downsampling(pts3d_cls=pts3d[mask_pts3d_sampled_rand], K=100000, top_bins_perc=0.90, return_mask=True)
#             #mask_pts3d_sampled[mask_pts3d_sampled_rand] = mask_pts3d_sampled_voxel
#             # particle_size = torch.cdist(pts3d[mask_pts3d_sampled], pts3d[mask_pts3d_sampled]).fill_diagonal_(torch.inf).min(dim=-1).values.mean()
#             # average Kth nearest neighbor distance
#             particle_size = torch.cdist(pts3d[mask_pts3d_sampled], pts3d[mask_pts3d_sampled]).quantile(dim=-1, q=3. / mask_pts3d_sampled.sum()).mean()
#
#             plane_dist_thresh = particle_size / 1.
#             plane_not_aggregating_dist_thresh = particle_size * 2.
#             obj_agg_dist_thresh = particle_size * 10.
#
#             center3d = fit_rays_center3d(cams_tform4x4_obj=cams_tform4x4_obj)
#             center3d_tform4x4_obj = tform4x4_from_transl3d(-center3d)
#
#             cams_tform4x4_obj = tform4x4_broadcast(cams_tform4x4_obj, inv_tform4x4(center3d_tform4x4_obj))
#             pts3d = transf3d_broadcast(pts3d, transf4x4=center3d_tform4x4_obj)
#             center3d = transf3d_broadcast(center3d, transf4x4=center3d_tform4x4_obj)
#
#             center3d_mesh = Mesh.create_sphere(center3d=center3d, radius=scene_size / 30., device=self.device)
#             #show_scene(meshes=[center3d_mesh], pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=[self.first_frame.cam_intr4x4])
#
#
#             cams_traj = inv_tform4x4(cams_tform4x4_obj)[:, :3, 3]
#             plane4d = ransac(pts=pts3d[mask_pts3d_sampled], fit_func=fit_plane,
#                              score_func=partial(score_plane4d_fit, plane_dist_thresh=plane_dist_thresh,
#                                                 cams_traj=cams_traj, pts_on_plane_weight=2.), fits_count=1000, fit_pts_count=3)
#
#             plane3d_tform4x4_obj = plane4d_to_tform4x4(plane4d)
#
#             #plane_z = -plane3d_tform4x4_obj[2, 3]
#             #plane3d_tform4x4_obj[:3, 3] = 0.
#             cams_tform4x4_obj = tform4x4_broadcast(cams_tform4x4_obj, inv_tform4x4(plane3d_tform4x4_obj))
#
#             pts3d = transf3d_broadcast(pts3d, transf4x4=plane3d_tform4x4_obj)
#             center3d = transf3d_broadcast(center3d, transf4x4=plane3d_tform4x4_obj)
#
#             height = scene_size / 5.
#             radius = scene_size * 0.5
#             plan3d_mesh = Mesh.create_plane_as_cone(radius=radius, height=height, device=self.device)
#
#             # show_scene(meshes=[center3d_mesh, plan3d_mesh], pts3d=[pts3d], pts3d_colors=[pts3d_colors], cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=[self.first_frame.cam_intr4x4])
#
#             #plane_tform4x4_pts3d = transf3d_broadcast(pts3d=pts3d, transf4x4=plane3d_tform4x4_obj)
#             mask_pts3d_on_plane = mask_pts3d_sampled * (pts3d[:, 2] < plane_dist_thresh)
#             mask_pts3d_on_plane_not_aggregating = mask_pts3d_sampled * (pts3d[:, 2] < plane_not_aggregating_dist_thresh)
#             # starting with 10 percentage of points
#             if (mask_pts3d_sampled * (~mask_pts3d_on_plane)).sum() >= pts3d_count_min:
#
#                 mask_center_thresh = (pts3d[mask_pts3d_sampled * (~mask_pts3d_on_plane)] - center3d).norm(dim=-1).quantile(0.1)
#                 mask_pts3d_clean = mask_pts3d_sampled * (~mask_pts3d_on_plane) * ((pts3d - center3d).norm(dim=-1) < mask_center_thresh)
#
#                 pts3d_not_on_plane = pts3d[mask_pts3d_sampled * (~mask_pts3d_on_plane)]
#                 dists_pts3d_not_on_plane_plane = pts3d[mask_pts3d_sampled * (~mask_pts3d_on_plane), 2]
#                 if ((~mask_pts3d_on_plane_not_aggregating) * mask_pts3d_clean).sum().item() > 0:
#                     dists_pts3d_not_on_plane_obj_dists = (pts3d[(~mask_pts3d_on_plane_not_aggregating) * mask_pts3d_clean][:, None] - pts3d[mask_pts3d_sampled * (~mask_pts3d_on_plane)][None, :]).norm(dim=-1).min(dim=0)[0]
#                     while ((dists_pts3d_not_on_plane_obj_dists < obj_agg_dist_thresh) * (dists_pts3d_not_on_plane_obj_dists < dists_pts3d_not_on_plane_plane)).sum() > mask_pts3d_clean.sum():
#                         logger.info(mask_pts3d_clean.sum())
#                         mask_pts3d_clean[mask_pts3d_sampled * (~mask_pts3d_on_plane)] += (dists_pts3d_not_on_plane_obj_dists < obj_agg_dist_thresh) * (dists_pts3d_not_on_plane_obj_dists < dists_pts3d_not_on_plane_plane)
#                         if ((~mask_pts3d_on_plane_not_aggregating) * mask_pts3d_clean).sum().item() > 0:
#                             dists_pts3d_not_on_plane_obj_dists = torch.cdist(pts3d[(~mask_pts3d_on_plane_not_aggregating) * mask_pts3d_clean], pts3d_not_on_plane).min(dim=0)[0]
#                         else:
#                             break
#
#                 pts3d_clean = pts3d[mask_pts3d_clean]
#
#                 if len(pts3d_clean) >= pts3d_count_min:
#                     pts3d_colors[mask_pts3d_on_plane] = torch.Tensor([0., 0., 1.]).to(device=self.device)
#                     pts3d_colors[mask_pts3d_clean] = torch.Tensor([0., 1., 0.]).to(device=self.device)
#                     pts3d_colors[mask_pts3d_sampled * (~mask_pts3d_on_plane) * (~mask_pts3d_clean)] = torch.Tensor([1., 0., 0.]).to(device=self.device)
#                 else:
#                     pts3d_clean = []
#                     mask_pts3d_clean = []
#             else:
#                 pts3d_clean = []
#                 mask_pts3d_clean = []
#         else:
#             pts3d_clean = []
#             mask_pts3d_clean = []
#
#         if len(pts3d_clean) < pts3d_count_min:
#             pts3d_clean, mask_pts3d_clean = random_sampling(pts3d, pts3d_max_count=pts3d_count_min, return_mask=True)
#
#         if not return_mask:
#             return pts3d_raw[mask_pts3d_clean]
#         else:
#             return pts3d_raw[mask_pts3d_clean], mask_pts3d_clean
#
#     @property
#     def fpath_pcl_axis_labeled(self):
#         return self.path_preprocess.joinpath('axis', self.pcl_source, self.category, self.name, 'axis.pt')
#
#
#     def get_cams(self, cam_tform_obj_source: CAM_TFORM_OBJ_SOURCES=None, cams_count=5, show_imgs=True):
#         if cam_tform_obj_source is None:
#             cam_tform_obj_source = self.cam_tform_obj_source
#         cams_tform4x4_world = []
#         cams_intr4x4 = []
#         cams_imgs = []
#         frames_count = len(self.frames_names)
#         if cams_count == -1:
#             step_size = 1
#         else:
#             step_size = (frames_count // cams_count) + 1
#         for c in range(0, frames_count, step_size):
#             frame = self.get_frame_by_index(c)
#             cams_tform4x4_world.append(
#                 frame.get_cam_tform4x4_obj(cam_tform_obj_source=cam_tform_obj_source))
#
#             cams_intr4x4.append(frame.cam_intr4x4)
#             if show_imgs:
#                 cams_imgs.append(frame.rgb)
#         return cams_tform4x4_world, cams_intr4x4, cams_imgs
#
#     def show(self, cam_tform_obj_source: CAM_TFORM_OBJ_SOURCES=None, pcl_source: PCL_SOURCES=None, cams_count=5, show_imgs=False, show_mesh=False):
#         meshes = None
#         if cam_tform_obj_source is None:
#             cam_tform_obj_source = self.cam_tform_obj_source
#         if pcl_source is None:
#             pcl_source = self.pcl_source
#             if show_mesh:
#                 meshes = [self.mesh]
#
#         cams_tform4x4_world, cams_intr4x4, cams_imgs = self.get_cams(cam_tform_obj_source=cam_tform_obj_source, cams_count=cams_count, show_imgs=show_imgs)
#         if not show_imgs:
#             cams_imgs = None
#         pts3d = self.get_pcl(pcl_source=pcl_source)
#         pts3d_colors = self.get_pcl_colors(pcl_source=pcl_source)
#         from od3d.cv.visual.show import show_scene
#         show_scene(meshes=meshes, cams_tform4x4_world=cams_tform4x4_world, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs, pts3d=[pts3d], pts3d_colors=[pts3d_colors])
#
#     @property
#     def pcl_axis_labeled(self):
#         fpath_axis_droid_slam = self.fpath_pcl_axis_labeled
#
#         if not fpath_axis_droid_slam.exists():
#             logger.info(f'fpath_droid_slam_axis_labeled missing {fpath_axis_droid_slam}')
#             self.preprocess_label(override=False)
#
#         axis_droid_slam = torch.load(fpath_axis_droid_slam)
#         return axis_droid_slam
#
#     def preprocess_label(self, override=False):
#
#         fpath_axis_droid_slam = self.fpath_pcl_axis_labeled
#
#         if fpath_axis_droid_slam.exists() and not override and self.fpath_labeled_obj_tform_obj.exists() and self.fpath_labeled_cuboid_obj_tform_labeled_obj.exists():
#             logger.info(f'Label axis already exists {fpath_axis_droid_slam}, override disabled.')
#             return
#         fpath_axis_droid_slam.parent.mkdir(parents=True, exist_ok=True)
#         from od3d.cv.label.axis import label_axis_in_pcl
#
#         cams_tform4x4_world = []
#         cams_intr4x4 = []
#         cams_imgs = []
#         frames_count = len(self.frames_names)
#         for c in range(0, frames_count, frames_count // 4):
#             frame = self.get_frame_by_index(c)
#             cams_tform4x4_world.append(frame.get_cam_tform4x4_obj(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.PCL))
#             cams_intr4x4.append(frame.cam_intr4x4)
#             cams_imgs.append(frame.rgb)
#
#         while True:
#             if self.fpath_labeled_obj_tform_obj.exists():
#                 prev_axis = torch.load(self.fpath_labeled_obj_tform_obj)
#             else:
#                 prev_axis = None
#             axis_pcl = label_axis_in_pcl(pts3d=self.get_pcl(),
#                                          pts3d_colors=self.get_pcl_colors(),
#                                          prev_labeled_pcl_tform_pcl=prev_axis,
#                                          cams_tform4x4_world=cams_tform4x4_world,
#                                          cams_intr4x4=cams_intr4x4,
#                                          cams_imgs=cams_imgs)
#
#             from od3d.cv.geometry.fit.axis_tform_from_pts3d import axis_tform4x4_obj_from_pts3d
#
#             if axis_pcl is None or axis_pcl.shape != (3, 2, 3):
#                 break
#
#             pcl_labeled_tform_pcl = axis_tform4x4_obj_from_pts3d(axis_pts3d=axis_pcl)
#
#
#             if (torch.linalg.det(pcl_labeled_tform_pcl[:3, :3]) - 1.).abs() <= 1e-5:
#                 break
#
#             logger.warning(f'labeled determinant is not ~1, but {torch.linalg.det(pcl_labeled_tform_pcl[:3, :3])}')
#
#         if axis_pcl is not None and axis_pcl.shape == (3, 2, 3):
#
#             logger.info(f'storing axis labeled, cuboid tform, and cuboid ')
#             torch.save(axis_pcl, f=fpath_axis_droid_slam)
#
#             self.fpath_labeled_obj_tform_obj.parent.mkdir(parents=True, exist_ok=True)
#             torch.save(pcl_labeled_tform_pcl, self.fpath_labeled_obj_tform_obj)
#
#             size = OD3D_CATEGORIES_SIZES_IN_M[MAP_CATEGORIES_CO3D_TO_OD3D[self.category]]
#             pcl_labeled_tform_pts3d = transf3d_broadcast(
#                 pts3d=self.get_pcl(),
#                 transf4x4=pcl_labeled_tform_pcl)
#             pcl_labeled_cuboid, pcl_labeled_cuboid_tform_pcl_labeled = \
#                 fit_cuboid_to_pts3d(pts3d=pcl_labeled_tform_pts3d, size=size, optimize_rot=False,
#                                     optimize_transl=True)
#
#             self.fpath_obj_labeled_cuboid.parent.mkdir(parents=True, exist_ok=True)
#             pcl_labeled_cuboid.write_to_file(fpath=self.fpath_obj_labeled_cuboid)
#
#             self.fpath_labeled_cuboid_obj_tform_labeled_obj.parent.mkdir(parents=True, exist_ok=True)
#             torch.save(pcl_labeled_cuboid_tform_pcl_labeled.detach().cpu(),
#                        f=self.fpath_labeled_cuboid_obj_tform_labeled_obj)
#         else:
#             logger.info(f'not storing labeled axis.')
#
#     def remove_pcl_preprocess_dependent_files(self):
#         logger.info('removing pcl dependent files...')
#         from glob import glob
#         paths = []
#         paths_mesh_feats = self.path_mesh_feats.joinpath(self.pcl_source, '*', '*', self.name_unique) #, 'mesh_feats.pt')
#         paths += glob(str(paths_mesh_feats))
#         paths_dist_mesh_feats = self.path_dist_verts_mesh_feats.joinpath(self.pcl_source, '*', '*', '*', self.name_unique, '*')
#         paths += glob(str(paths_dist_mesh_feats))
#         paths_dist_mesh_feats = self.path_dist_verts_mesh_feats.joinpath(self.pcl_source, '*', '*', '*', '*', self.name_unique)
#         paths += glob(str(paths_dist_mesh_feats))
#         paths_meshs = self.path_preprocess.joinpath('mesh', f'{self.pcl_source}', '*', self.category, self.name)
#         paths += glob(str(paths_meshs))
#
#         for path in paths:
#             od3d.io.rm_dir(path)
#
#     def remove_mesh_feats_preprocess_dependent_files(self):
#         logger.info('removing mesh feats dependen files...')
#         from glob import glob
#         paths = [] # alpha500/M_dino_vits8_frozen_base_T_centerzoom512_R_acc
#         paths_mesh_feats = self.path_mesh_feats.joinpath(self.pcl_source, self.mesh_name, self.mesh_feats_type, self.name_unique) #, 'mesh_feats.pt')
#         paths += glob(str(paths_mesh_feats))
#         paths_dist_mesh_feats = self.path_dist_verts_mesh_feats.joinpath(self.pcl_source, self.mesh_name, self.mesh_feats_type, self.dist_verts_mesh_feats_reduce_type, self.name_unique, '*')
#         paths += glob(str(paths_dist_mesh_feats))
#         paths_dist_mesh_feats = self.path_dist_verts_mesh_feats.joinpath(self.pcl_source, self.mesh_name, self.mesh_feats_type, self.dist_verts_mesh_feats_reduce_type, '*', self.name_unique)
#         paths += glob(str(paths_dist_mesh_feats))
#         for path in paths:
#             od3d.io.rm_dir(path)
#
#     @property
#     def fpath_labeled_obj_tform_obj(self):
#         return self.get_fpath_a_src_tform_b_src(src_a=CAM_TFORM_OBJ_SOURCES.LABELED, src_b=CAM_TFORM_OBJ_SOURCES.PCL)
#
#     @property
#     def path_zsp_labels(self):
#         return Path('third_party/zero-shot-pose/data/class_labels')
#
#     @property
#     def sequence_ref_zsp(self):
#         ref_seq_name = sorted(list(self.path_zsp_labels.joinpath(self.category).iterdir()))[0].stem
#         ref_seq = self.get_sequence_by_category_and_name(category=self.category, name=ref_seq_name)
#         return ref_seq
#
#     @property
#     def co3dv1_zsp_obj_tform_co3dv1_obj(self):
#         import numpy as np
#         from od3d.io import read_json
#         path_zsp = self.path_zsp_labels
#         fpath_src_gt = path_zsp.joinpath(self.name_unique + '.json')
#         #if fpath_src_gt.exists():
#         gt_co3d_global_tform_co3d_src = inv_tform4x4(torch.from_numpy(np.array(read_json(fpath_src_gt)['trans'])))
#         gt_co3d_global_tform_co3d_src = gt_co3d_global_tform_co3d_src.to(torch.float)
#         return gt_co3d_global_tform_co3d_src
#
#
#     @property
#     def labeled_obj_tform_obj(self):
#         fpath_pcl_labeled_tform_pcl = self.fpath_labeled_obj_tform_obj
#         if not self.fpath_labeled_obj_tform_obj.exists():
#             logger.info(f'pcl_labeled_tform_pcl missing {fpath_pcl_labeled_tform_pcl}')
#             self.preprocess_label()
#
#         pcl_labeled_tform_pcl = torch.load(fpath_pcl_labeled_tform_pcl)
#         return pcl_labeled_tform_pcl
#
#
#     @property
#     def co3dv1_zsp_obj_tform_obj(self):
#         co3dv1_obj_tform_obj = self.get_a_src_tform_b_src(src_a=CAM_TFORM_OBJ_SOURCES.CO3DV1,
#                                                           src_b=CAM_TFORM_OBJ_SOURCES.PCL)
#         co3dv1_zsp_obj_tform_obj = tform4x4(self.co3dv1_zsp_obj_tform_co3dv1_obj, co3dv1_obj_tform_obj)
#         return co3dv1_zsp_obj_tform_obj
#
#
#     @property
#     def zsp_labeled_cuboid_ref_tform_obj(self):
#         ref_seq = self.sequence_ref_zsp
#         zsp_labeled_cuboid_ref_tform_obj = tform4x4(ref_seq.labeled_cuboid_obj_tform_labeled_obj, tform4x4(ref_seq.labeled_obj_tform_obj, tform4x4(inv_tform4x4(ref_seq.co3dv1_zsp_obj_tform_obj), self.co3dv1_zsp_obj_tform_obj)))
#         # note: as zsp does not offer scale, we cannot retrieve actual translation, therefore we use this pcl center
#         zsp_labeled_cuboid_ref_tform_obj[:3, 3] = 0
#         zsp_labeled_cuboid_ref_tform_obj[:3, 3] = -transf3d_broadcast(self.get_pcl().mean(dim=0), transf4x4=zsp_labeled_cuboid_ref_tform_obj)
#         return zsp_labeled_cuboid_ref_tform_obj
#
#     @property
#     def fpath_obj_labeled_cuboid(self):
#         return self.path_preprocess.joinpath('mesh', 'cuboid', self.pcl_source, self.name_unique, 'mesh.ply')
#
#     @property
#     def obj_labeled_cuboid(self):
#         fpath_droid_slam_labeled_cuboid = self.fpath_obj_labeled_cuboid
#
#         if not fpath_droid_slam_labeled_cuboid.exists():
#             logger.info(f'fpath_droid_slam_labeled_cuboid missing {fpath_droid_slam_labeled_cuboid}')
#             self.preprocess_label(override=False)
#
#         labeled_cuboid = Meshes.load_from_files([fpath_droid_slam_labeled_cuboid])
#         return labeled_cuboid
#
#     #@property
#     #def fpath_droid_slam_labeled_cuboid_tform_droid_slam_labeled(self):
#     #    return self.path_preprocess.joinpath('a_src_tform_b_src', 'droid_slam_labeled_cuboid_tform_droid_slam_labeled', self.name_unique, 'tform.pt')
#
#     @property
#     def fpath_labeled_cuboid_obj_tform_labeled_obj(self):
#         return self.get_fpath_a_src_tform_b_src(src_a=CAM_TFORM_OBJ_SOURCES.LABELED_CUBOID, src_b=CAM_TFORM_OBJ_SOURCES.LABELED)
#
#     @property
#     def labeled_cuboid_obj_tform_obj(self):
#         return tform4x4(self.labeled_cuboid_obj_tform_labeled_obj, self.labeled_obj_tform_obj)
#
#     @property
#     def labeled_cuboid_obj_tform_labeled_obj(self):
#         fpath_labeled_cuboid_obj_tform_labeled_obj = self.fpath_labeled_cuboid_obj_tform_labeled_obj
#         #logger.info(f'fpath labeled cuboid {fpath_droid_slam_labeled_cuboid_tform_droid_slam_labeled}')
#         if not fpath_labeled_cuboid_obj_tform_labeled_obj.exists():
#             logger.info(f'fpath_labeled_cuboid_obj_tform_labeled_obj missing {fpath_labeled_cuboid_obj_tform_labeled_obj}')
#             self.preprocess_label(override=False)
#
#         labeled_cuboid_obj_tform_labeled_obj = torch.load(fpath_labeled_cuboid_obj_tform_labeled_obj)
#         return labeled_cuboid_obj_tform_labeled_obj
#
#     def write_aligned_obj_tform_obj(self, aligned_obj_tform_obj: torch.Tensor, aligned_name: str):
#         fpath_tform_aligned = self.get_fpath_aligned_obj_tform_obj_with_aligned_name(aligned_name=aligned_name)
#         fpath_tform_aligned.parent.mkdir(parents=True, exist_ok=True)
#         torch.save(aligned_obj_tform_obj.detach().cpu(), f=fpath_tform_aligned)
#
#     def write_aligned_cuboid(self, cuboid: Meshes, aligned_name: str):
#         fpath_mesh_aligned = self.path_preprocess.joinpath('aligned', aligned_name, 'mesh', self.category, 'mesh.ply')
#         fpath_mesh_aligned.parent.mkdir(parents=True, exist_ok=True)
#         cuboid.write_to_file(fpath=fpath_mesh_aligned)
#
#     @property
#     def aligned_obj_tform_obj(self):
#         fpath_tform_aligned = self.fpath_aligned_obj_tform_obj
#         droid_slam_aligned_tform_droid_slam = torch.load(fpath_tform_aligned).to('cpu')
#         return droid_slam_aligned_tform_droid_slam
#
#
#     @property
#     def fpath_meta_ext(self):
#         return self.path_meta.joinpath('sequences_ext', self.name_unique, 'meta.yaml')
#
#     def write_meta_ext(self, key, value):
#         self.meta_ext[key] = value
#         write_dict_as_yaml(_dict=self.meta_ext, fpath=self.fpath_meta_ext)
#
#     @property
#     def meta_ext(self):
#         if self._meta_ext is None:
#             if self.fpath_meta_ext.exists():
#                 self._meta_ext = read_dict_from_yaml(fpath=self.fpath_meta_ext)
#             else:
#                 self._meta_ext = {}
#         return self._meta_ext
#
#     @property
#     def mask_coverage(self):
#         if 'mask_coverage' not in self.meta_ext.keys():
#             _ = self.good_cam_movement
#         return self.meta_ext['mask_coverage']
#
#     @property
#     def centered_accuracy(self):
#         if 'centered_accuracy' not in self.meta_ext.keys():
#             _ = self.good_cam_movement
#         return self.meta_ext['centered_accuracy']
#
#     @property
#     def viewpoint_coverage(self):
#         if 'viewpoints_coverage' not in self.meta_ext.keys():
#             _ = self.good_cam_movement
#         return self.meta_ext['viewpoints_coverage']
#
#
#     @property
#     def good_cam_movement(self):
#         name = f'{self.pcl_source}_good_cam_movement'
#         if name not in self.meta_ext:
#             device = 'cuda'
#             frames = self.get_frames()
#             frames_mask = torch.stack([frame.mask for frame in frames], dim=0).to(device=device)
#             frames_mask_coverage = frames_mask.flatten(1).mean(dim=-1).mean()
#
#             frames_cam_tform4x4_obj = torch.stack([frame.meta.cam_tform4x4_obj for frame in frames], dim=0).to(device=device)
#             frames_cam_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=device)
#             frames_H = torch.Tensor([frame.H for frame in frames]).to(device=device)
#             frames_W = torch.Tensor([frame.W for frame in frames]).to(device=device)
#             frames_WH = torch.stack([frames_W, frames_H], dim=-1).to(device=device)
#
#             frames_cam_proj4x4_obj = tform4x4(frames_cam_intr4x4, frames_cam_tform4x4_obj)
#             center3d = fit_rays_center3d(cams_tform4x4_obj=frames_cam_tform4x4_obj)
#             center2d = proj3d2d_broadcast(proj4x4=frames_cam_proj4x4_obj, pts3d=center3d)
#             center2d_offset_small_acc = (((abs(center2d - (frames_WH / 2.)) / frames_WH) < 0.3).all(dim=-1) * 1.).mean()
#
#             viewpoints3d = inv_tform4x4(frames_cam_tform4x4_obj)[:, :3, 3] - center3d
#             viewpoints3d_norm = torch.nn.functional.normalize(viewpoints3d, dim=-1)
#             steps_dim = 5
#             viewpoints3d_uniform = torch.stack(torch.meshgrid(
#                 torch.linspace(start=-1., end=1., steps=steps_dim, device=device),
#                 torch.linspace(start=-1., end=1., steps=steps_dim, device=device),
#                 torch.linspace(start=-1., end=1., steps=steps_dim, device=device), indexing='xy'), dim=-1).reshape(-1,
#                                                                                                                    3)
#             viewpoints3d_uniform = viewpoints3d_uniform[abs(viewpoints3d_uniform.norm(dim=-1) - 1) < (1. / steps_dim)]
#
#             dist_viewpoints_versus_uniform = torch.cdist(viewpoints3d_norm, viewpoints3d_uniform)
#             viewpoints_coverage = len(dist_viewpoints_versus_uniform.min(dim=-1).indices.unique()) / len(
#                 viewpoints3d_uniform)
#
#             self.write_meta_ext('viewpoints_coverage', viewpoints_coverage)
#             self.write_meta_ext('centered_accuracy', center2d_offset_small_acc.item())
#             self.write_meta_ext('mask_coverage', frames_mask_coverage.item())
#
#             if viewpoints_coverage < 0.15:
#                 logger.warning(f'did skip pcl {self.name} due too few viewpoints coverage {viewpoints_coverage}')
#                 good_cam_movement = False
#             elif center2d_offset_small_acc < 0.8:
#                 logger.warning(f'did skip pcl {self.name} due too small centering accuracy {center2d_offset_small_acc}')
#                 good_cam_movement = False
#             elif frames_mask_coverage < 0.1:
#                 logger.warning(f'did skip pcl {self.name} due too few mask coverage {frames_mask_coverage}')
#                 good_cam_movement= False
#             else:
#                 good_cam_movement = True
#
#             logger.info(f'viewpoints coverage {viewpoints_coverage}')
#             logger.info(f'centering accuracy {center2d_offset_small_acc}')
#             logger.info(f'mask coverage {frames_mask_coverage}')
#
#             # DEBUG BLOCK START
#             # from od3d.cv.visual.show import show_scene
#             # show_scene(pts3d=[viewpoints3d_uniform, viewpoints3d_norm])
#             # DEBUG BLOCK END
#
#             self.write_meta_ext(name, good_cam_movement)
#
#         else:
#             good_cam_movement = self.meta_ext[name]
#
#         if not good_cam_movement:
#             logger.warning(f'there is no {name} in {self.name_unique}.')
#
#         return good_cam_movement
#
#
#     @property
#     def no_missing_frames(self):
#         if 'no_missing_frames' not in self.meta_ext:
#             from od3d.datasets.frame import OD3D_Meta
#             frames_tstamps = torch.Tensor([OD3D_Meta.atoi(re.split(r'(\d+)', frame.meta.rfpath_rgb.stem)[-2]) for frame in self.get_frames()])
#             missing_frames = (~((frames_tstamps[1:] - frames_tstamps[:-1]) == 1)).sum().item()
#             no_missing_frames = missing_frames <= 3
#             self.write_meta_ext('no_missing_frames', no_missing_frames)
#
#         else:
#             no_missing_frames = self.meta_ext['no_missing_frames']
#
#         if not no_missing_frames:
#             logger.warning(f'there are more than 3 missing frames in {self.name_unique}.')
#
#         return no_missing_frames
#     @property
#     def gt_pose_available(self):
#         if self.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.ALIGNED:
#             return self.fpath_aligned_obj_tform_obj.exists()
#         elif self.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.LABELED:
#             return self.fpath_labeled_obj_tform_obj.exists()
#         elif self.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.LABELED_CUBOID:
#             return self.fpath_labeled_cuboid_obj_tform_labeled_obj.exists()
#         #elif self.cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.ZSP_LABELED:
#         #    return self.fpath_co
#         else:
#             return True
#     @property
#     def fpath_aligned_obj_tform_obj(self):
#         return self.get_fpath_aligned_obj_tform_obj_with_aligned_name(aligned_name=self.aligned_name)
#
#     def get_fpath_aligned_obj_tform_obj_with_aligned_name(self, aligned_name: str):
#         return self.path_preprocess.joinpath('aligned', aligned_name, 'aligned_obj_tform_obj', self.name_unique, 'tform.pt')
#
#     @property
#     def fpath_mesh(self):
#
#         fpath_mesh = self.get_fpath_mesh(mesh_source=self.cuboid_source)
#         # logger.info(f'fpath mesh {fpath_mesh}')
#         return fpath_mesh
#
#     def get_fpath_mesh(self, mesh_source: CUBOID_SOURCES):
#         if mesh_source == CUBOID_SOURCES.ALIGNED: # CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ALIGNED:
#             return self.path_preprocess.joinpath('aligned', self.aligned_name, 'mesh', self.category, 'mesh.ply')
#         elif mesh_source == CUBOID_SOURCES.ZSP_REF_CUBOID: #  == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ZSP_LABELED:
#             ref_seq = self.sequence_ref_zsp
#             #_ = ref_seq.droid_slam_labeled_cuboid # leads to infinity loop
#             return ref_seq.fpath_obj_labeled_cuboid
#         elif mesh_source == CUBOID_SOURCES.LABELED: # CAM_TFORM_OBJ_SOURCES.DROID_SLAM_LABELED:
#             return self.fpath_obj_labeled_cuboid
#         else:
#             return self.get_fpath_mesh_pcl()
#             #return self.path_preprocess.joinpath('mesh', f'{self.pcl_source}', f'{self.mesh_name}', self.category, self.name, f'mesh.ply')
#             #return self.path_preprocess.joinpath('mesh', f'{self.mesh_name}', self.category, self.name, f'mesh.ply')
#
#     def get_fpath_mesh_pcl(self):
#         return self.path_preprocess.joinpath('mesh', f'{self.pcl_source}', f'{self.mesh_name}', self.category,
#                                              self.name, f'mesh.ply')
#
#     def get_mesh(self, mesh_source: CUBOID_SOURCES, add_rgb_from_pca=False, device='cpu'):
#         fpath_mesh = self.get_fpath_mesh(mesh_source=mesh_source)
#         if not fpath_mesh.exists():
#             self.preprocess_mesh()
#         mesh = Mesh.load_from_file(fpath=fpath_mesh, device=device)
#         if add_rgb_from_pca:
#             instance_feats = self.feats
#             pca_V = self.categorical_pca_V.to(device=device)
#             if isinstance(instance_feats, List):
#                 verts_feats_pca = torch.stack(
#                     [torch.matmul(vert_feats.to(device=device), pca_V[:, 3:6]).mean(dim=0) for vert_feats in instance_feats], dim=0)
#             else:
#                 verts_feats_pca = torch.matmul(instance_feats, pca_V[:, 3:6])
#             mesh.rgb = (verts_feats_pca.nan_to_num() + 0.5 ).clamp(0, 1)
#         return mesh
#
#     @property
#     def mesh(self):
#         if self._mesh is None:
#             fpath_mesh = self.fpath_mesh
#             if not fpath_mesh.exists():
#                 self.preprocess_mesh()
#             self._mesh = Mesh.load_from_file(fpath=self.fpath_mesh)
#         return self._mesh
#
#     @mesh.setter
#     def mesh(self, value: torch.Tensor):
#             self._mesh = value
#
#     @property
#     def path_droid_slam(self):
#         return self.path_preprocess.joinpath('droid_slam', self.category, self.name)
#
#     @property
#     def fpath_droid_slam_pcl(self):
#         return self.path_preprocess.joinpath('droid_slam', self.category, self.name, 'pcl.ply')
#
#     @property
#     def fpath_droid_slam_traj(self):
#         return self.path_preprocess.joinpath('droid_slam', self.category, self.name, 'traj_est.pt')
#
#     def preprocess_droid_slam(self, override=False):
#         if not override and self.path_droid_slam.exists():
#             logger.info(f'droid slam already exists at {self.path_droid_slam}')
#             return
#
#         self.run_droid_slam()
#
#
#     @property
#     def com(self):
#         return self.pcl_clean.mean(dim=0)
#
#     @property
#     def fpath_categorical_pca_V(self):
#         return self.path_preprocess.joinpath('pca', 'categorical_pca_V', self.mesh_feats_type, self.category)
#
#     @property
#     def categorical_pca_V(self):
#         if self.fpath_categorical_pca_V.exists():
#             return torch.load(self.fpath_categorical_pca_V)
#         else:
#             logger.warning(f'Categorical pca V does not exist for sequence {self.name_unique}')
#             feats = self.feats
#             if isinstance(feats, List):
#                 feats = torch.cat([vert_feats for vert_feats in feats], dim=0)
#
#             _, _, categorical_pca_V = torch.pca_lowrank(feats)
#             return categorical_pca_V
#
#     @categorical_pca_V.setter
#     def categorical_pca_V(self, value: torch.Tensor):
#         self.fpath_categorical_pca_V.parent.mkdir(parents=True, exist_ok=True)
#         torch.save(value.detach().cpu(), f=self.fpath_categorical_pca_V)
#
#     @property
#     def cuboid(self):
#         if self._cuboid is None:
#             fpath_cuboid = self.fpath_cuboid
#             if not fpath_cuboid.exists():
#                 self.preprocess_cuboid()
#             self._cuboid = Cuboids.load_from_files(fpaths_meshes=[fpath_cuboid])
#         return self._cuboid
#
