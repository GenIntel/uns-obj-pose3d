import logging
logger = logging.getLogger(__name__)
from typing import List
import torch
from pathlib import Path

from od3d.datasets.object import OD3D_CAM_TFORM_OBJ_TYPES
from od3d.datasets.frame_meta import OD3D_FrameMeta, OD3D_FrameMetaRGBMixin, OD3D_FrameMetaDepthMixin,  \
    OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaMaskMixin, OD3D_FrameMetaSizeMixin, OD3D_FrameMetaSequenceMixin, \
    OD3D_FrameMetaDepthMaskMixin, OD3D_FrameMetaCamIntr4x4Mixin, OD3D_FrameMetaCamTform4x4ObjMixin

from od3d.datasets.frame import OD3D_Frame, OD3D_FrameSizeMixin, OD3D_FrameRGBMixin, OD3D_FrameMaskMixin, \
    OD3D_FrameRGBMaskMixin, OD3D_FrameRaysCenter3dMixin, OD3D_FrameMeshMixin, OD3D_FrameTformObjMixin, \
    OD3D_FrameSequenceMixin, OD3D_FrameCategoryMixin, OD3D_FrameCamIntr4x4Mixin, OD3D_FrameCamTform4x4ObjMixin, \
    OD3D_CamProj4x4ObjMixin, OD3D_FRAME_MASK_TYPES, OD3D_FrameDepthMixin, OD3D_FrameDepthMaskMixin
from dataclasses import dataclass
import numpy as np
from od3d.datasets.object import OD3D_PCLTypeMixin, OD3D_MeshTypeMixin, OD3D_SequenceSfMTypeMixin
from od3d.datasets.co3d.enum import MAP_CATEGORIES_CO3D_TO_OD3D
from od3d.datasets.co3d.enum import CO3D_FRAME_TYPES
from od3d.cv.io import read_image, read_co3d_depth_image
from co3d.dataset.data_types import load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
from od3d.cv.geometry.transform import transf4x4_from_rot3x3_and_transl3

@dataclass
class CO3D_FrameMeta(OD3D_FrameMetaCamIntr4x4Mixin, OD3D_FrameMetaCamTform4x4ObjMixin, OD3D_FrameMetaDepthMaskMixin,
                     OD3D_FrameMetaDepthMixin, OD3D_FrameMetaMaskMixin, OD3D_FrameMetaRGBMixin, OD3D_FrameMetaSizeMixin,
                     OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaSequenceMixin, OD3D_FrameMeta):
    depth_scale: float
    co3d_frame_type: CO3D_FRAME_TYPES

    @staticmethod
    def get_rfpath_frame_meta_with_category_sequence_and_frame_name(category: str, sequence_name: str, name: str):
        return CO3D_FrameMeta.get_rfpath_metas().joinpath(category, sequence_name, name + '.yaml')

    @staticmethod
    def get_fpath_frame_meta_with_category_sequence_and_frame_name(path_meta: Path, category: str, sequence_name: str, name: str):
        return path_meta.joinpath(CO3D_FrameMeta.get_rfpath_frame_meta_with_category_sequence_and_frame_name(category=category, sequence_name=sequence_name, name=name))

    @staticmethod
    def load_from_raw(frame_annotation: FrameAnnotation):
        category = frame_annotation.image.path.split('/')[0]
        sequence_name = frame_annotation.sequence_name
        if frame_annotation.meta is not None:
            co3d_frame_type = CO3D_FRAME_TYPES(frame_annotation.meta['frame_type'])
        else:
            co3d_frame_type = CO3D_FRAME_TYPES.CO3DV1
        name = f'{frame_annotation.frame_number}'

        rfpath_mask = Path(frame_annotation.mask.path)

        rfpath_rgb = Path(frame_annotation.image.path)

        depth_scale = frame_annotation.depth.scale_adjustment
        rfpath_depth = Path(frame_annotation.depth.path)

        rfpath_depth_mask = Path(frame_annotation.depth.mask_path)

        cam_tform4x4_obj = transf4x4_from_rot3x3_and_transl3(rot3x3=torch.Tensor(frame_annotation.viewpoint.R).T, transl3=torch.Tensor(frame_annotation.viewpoint.T))
        default_tform_t3d = torch.Tensor([[-1., 0., 0., 0.],
                                         [0., -1., 0., 0.],
                                         [0., 0., 1., 0.],
                                         [0., 0., 0., 1.]])
        cam_tform4x4_obj = torch.bmm(default_tform_t3d[None,], cam_tform4x4_obj[None,])[0]

        H, W = frame_annotation.image.size
        size = torch.Tensor([H, W])

        if frame_annotation.viewpoint.intrinsics_format == 'ndc_isotropic':
            # see https://pytorch3d.org/docs/cameras
            s = min(H, W)
            focal_length = torch.Tensor(frame_annotation.viewpoint.focal_length) * s / 2.
            principal_point = -torch.Tensor(frame_annotation.viewpoint.principal_point) * s / 2. + size.flip(
                dims=(0,)) / 2.

        elif frame_annotation.viewpoint.intrinsics_format == 'ndc_norm_image_bounds':
            focal_length = torch.Tensor(frame_annotation.viewpoint.focal_length)
            focal_length[0] *= W / 2.
            focal_length[1] *= H / 2.
            principal_point = -torch.Tensor(frame_annotation.viewpoint.principal_point)
            principal_point[0] *= W / 2.
            principal_point[1] *= H / 2.
            principal_point += size.flip(dims=(0,)) / 2.
        else:
            logger.warning(f'Unknown viewpoint intrinsics format {frame_annotation.viewpoint.intrinsics_format}.')
            raise NotImplementedError

        cam_intr4x4 = torch.Tensor([[focal_length[0], 0., principal_point[0], 0.],
                           [0., focal_length[1], principal_point[1], 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])

        l_size = size.tolist()
        l_cam_intr4x4 = cam_intr4x4.tolist()
        l_cam_tform4x4_obj = cam_tform4x4_obj.tolist()
        return CO3D_FrameMeta(rfpath_rgb=rfpath_rgb, category=category, sequence_name=sequence_name, l_size=l_size,
                              name=name, rfpath_mask=rfpath_mask, rfpath_depth=rfpath_depth,
                              rfpath_depth_mask=rfpath_depth_mask, l_cam_intr4x4=l_cam_intr4x4,
                              l_cam_tform4x4_obj=l_cam_tform4x4_obj, depth_scale=depth_scale, co3d_frame_type=co3d_frame_type)

@dataclass
class CO3D_Frame(OD3D_FrameMeshMixin, OD3D_FrameRaysCenter3dMixin, OD3D_FrameTformObjMixin, OD3D_CamProj4x4ObjMixin,
                 OD3D_FrameRGBMaskMixin, OD3D_FrameMaskMixin, OD3D_FrameRGBMixin, OD3D_FrameDepthMixin,
                 OD3D_FrameDepthMaskMixin, OD3D_FrameCategoryMixin, OD3D_FrameSequenceMixin, OD3D_FrameSizeMixin,
                 OD3D_MeshTypeMixin, OD3D_PCLTypeMixin, OD3D_SequenceSfMTypeMixin, OD3D_Frame):
    meta_type = CO3D_FrameMeta
    map_categories_to_od3d = MAP_CATEGORIES_CO3D_TO_OD3D

    def __post_init__(self):
        # hack: prevents circular import
        from od3d.datasets.co3d.sequence import CO3D_Sequence
        self.sequence_type = CO3D_Sequence

    def get_depth(self):
        if self.depth is None:
            self.depth = read_co3d_depth_image(self.fpath_depth) * self.meta.depth_scale
        return self.depth


# from od3d.datasets.co3d.enum import CAM_TFORM_OBJ_SOURCES, CUBOID_SOURCES
# from od3d.datasets.frame import OD3D_FrameMeta, OD3D_Frame
# from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES
# from omegaconf import DictConfig, OmegaConf
# from co3d.dataset.data_types import (
#     load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation)
#
# import torch
# from od3d.cv.geometry.transform import transf4x4_from_rot3x3_and_transl3
# from pathlib import Path
#
#
# from od3d.cv.geometry.transform import inv_tform4x4
#
# from od3d.cv.geometry.transform import tform4x4
#
# import logging
# logger = logging.getLogger(__name__)
# from dataclasses import dataclass
# import torch.utils.data
# from typing import List
# import numpy as np
# from od3d.datasets.co3d.enum import CO3D_FRAME_TYPES, PCL_SOURCES
#
# from od3d.datasets.frame_meta import OD3D_FrameMeta, \
#     OD3D_FrameMetaSequenceMixin, OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaRGBMixin, \
#     OD3D_FrameMetaSizeMixin, OD3D_FrameMetaMaskMixin, OD3D_FrameMetaDepthMixin, OD3D_FrameMetaDepthMaskMixin, \
#     OD3D_FrameMetaCamTform4x4ObjMixin, OD3D_FrameMetaCamIntr4x4Mixin
#
# @dataclass
# class CO3D_FrameMeta(OD3D_FrameMetaCamTform4x4ObjMixin, OD3D_FrameMetaCamIntr4x4Mixin,
#                      OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaSequenceMixin, OD3D_FrameMetaDepthMaskMixin,
#                      OD3D_FrameMetaDepthMixin, OD3D_FrameMetaMaskMixin, OD3D_FrameMetaRGBMixin,
#                      OD3D_FrameMetaSizeMixin, OD3D_FrameMeta):
#     depth_scale: float
#     frame_type: str
#
#     @property
#     def name_unique(self):
#         return f'{self.category}/{self.sequence_name}/{self.name}'
#
#     @staticmethod
#     def get_name_unique_with_category_sequence_and_name(category: str, sequence_name: str, name: str):
#         return f'{category}/{sequence_name}/{name}'
#
#     @staticmethod
#     def load_from_raw(frame_annotation: FrameAnnotation):
#         category = frame_annotation.image.path.split('/')[0]
#         sequence_name = frame_annotation.sequence_name
#         if frame_annotation.meta is not None:
#             frame_type = frame_annotation.meta['frame_type']
#         else:
#             frame_type = CO3D_FRAME_TYPES.CO3DV1.value
#         name = f'{frame_annotation.frame_number}'
#
#         rfpath_mask = Path(frame_annotation.mask.path)
#
#         rfpath_rgb = Path(frame_annotation.image.path)
#
#         depth_scale = frame_annotation.depth.scale_adjustment
#         rfpath_depth = Path(frame_annotation.depth.path)
#
#         rfpath_depth_mask = Path(frame_annotation.depth.mask_path)
#
#         cam_tform4x4_obj = transf4x4_from_rot3x3_and_transl3(rot3x3=torch.Tensor(frame_annotation.viewpoint.R).T, transl3=torch.Tensor(frame_annotation.viewpoint.T))
#         default_tform_t3d = torch.Tensor([[-1., 0., 0., 0.],
#                                          [0., -1., 0., 0.],
#                                          [0., 0., 1., 0.],
#                                          [0., 0., 0., 1.]])
#         cam_tform4x4_obj = torch.bmm(default_tform_t3d[None,], cam_tform4x4_obj[None,])[0]
#
#         H, W = frame_annotation.image.size
#         size = torch.Tensor([H, W])
#
#         if frame_annotation.viewpoint.intrinsics_format == 'ndc_isotropic':
#             # see https://pytorch3d.org/docs/cameras
#             s = min(H, W)
#             focal_length = torch.Tensor(frame_annotation.viewpoint.focal_length) * s / 2.
#             principal_point = -torch.Tensor(frame_annotation.viewpoint.principal_point) * s / 2. + size.flip(
#                 dims=(0,)) / 2.
#
#         elif frame_annotation.viewpoint.intrinsics_format == 'ndc_norm_image_bounds':
#             focal_length = torch.Tensor(frame_annotation.viewpoint.focal_length)
#             focal_length[0] *= W / 2.
#             focal_length[1] *= H / 2.
#             principal_point = -torch.Tensor(frame_annotation.viewpoint.principal_point)
#             principal_point[0] *= W / 2.
#             principal_point[1] *= H / 2.
#             principal_point += size.flip(dims=(0,)) / 2.
#         else:
#             logger.warning(f'Unknown viewpoint intrinsics format {frame_annotation.viewpoint.intrinsics_format}.')
#             raise NotImplementedError
#
#         cam_intr4x4 = torch.Tensor([[focal_length[0], 0., principal_point[0], 0.],
#                            [0., focal_length[1], principal_point[1], 0.],
#                            [0., 0., 1., 0.],
#                            [0., 0., 0., 1.]])
#
#         l_size = size.tolist()
#         l_cam_intr4x4 = cam_intr4x4.tolist()
#         l_cam_tform4x4_obj = cam_tform4x4_obj.tolist()
#         return CO3D_FrameMeta(category=category, frame_type=frame_type,
#                    name=name, rfpath_mask=rfpath_mask, rfpath_depth=rfpath_depth, rfpath_depth_mask=rfpath_depth_mask,
#                    rfpath_rgb=rfpath_rgb, l_size=l_size, l_cam_intr4x4=l_cam_intr4x4,
#                    sequence_name=sequence_name,
#                    l_cam_tform4x4_obj=l_cam_tform4x4_obj, depth_scale=depth_scale)
#
#     @staticmethod
#     def get_subset_frames_names_uniform(frames_names, count_max_per_sequence=None):
#         if count_max_per_sequence is not None:
#             frames_names = [frames_names[fid] for fid in
#                             np.linspace(0, len(frames_names) - 1, count_max_per_sequence).astype(int).tolist()]
#         return frames_names
#
#     @staticmethod
#     def load_from_meta_with_category_sequence_and_frame_name(path_meta: Path, category: str, sequence_name:str, frame_name: str):
#         return CO3D_FrameMeta.load_from_meta_with_rfpath(path_meta=path_meta,
#                                                          rfpath=CO3D_FrameMeta.get_rfpath_frame_meta_with_category_sequence_and_frame_name(category=category, sequence_name=sequence_name, name=frame_name))
#
#     @staticmethod
#     def get_rfpath_frame_meta_with_category_sequence_and_frame_name(category: str, sequence_name: str, name: str):
#         return CO3D_FrameMeta.get_rfpath_metas().joinpath(category, sequence_name, name + '.yaml')
#
#     @staticmethod
#     def get_path_frames_meta_with_category_sequence(path_meta: Path, category: str, sequence: str):
#         return path_meta.joinpath(CO3D_FrameMeta.get_rpath_frames_meta_with_category_sequence_name(category=category, sequence=sequence))
#
#     @staticmethod
#     def get_rpath_frames_meta_with_category_sequence_name(category: str, sequence: str):
#         return CO3D_FrameMeta.get_rfpath_metas().joinpath(category, sequence)
#
#     @staticmethod
#     def get_fpath_frame_meta_with_category_sequence_and_frame_name(path_meta: Path, category: str, sequence_name: str, name: str):
#         return path_meta.joinpath(CO3D_FrameMeta.get_rfpath_frame_meta_with_category_sequence_and_frame_name(category=category, sequence_name=sequence_name, name=name))
#
#     @staticmethod
#     def get_frames_names_of_category_sequence(path_meta: Path, category: str, sequence_name: str):
#         dict_nested_frames = CO3D_FrameMeta.complete_nested_metas(path_meta=path_meta, dict_nested_metas={category: {sequence_name: None}})
#         frames_names = dict_nested_frames[category][sequence_name]
#         return frames_names
#
#     """
#     @staticmethod
#     def load_from_meta_with_rfpath(path_meta: Path, rfpath: Path):
#         return CO3D_FrameMeta(**CO3D_FrameMeta.load_omega_conf_with_rfpath(path_meta=path_meta, rfpath=rfpath))
#
#
#     @staticmethod
#     def load_from_meta_with_name_unique(path_meta: Path, name_unique: str):
#
#         return CO3D_FrameMeta.load_from_meta_with_rfpath(path_meta=path_meta, rfpath=CO3D_FrameMeta.get_rfpath_frame_meta_with_category_sequence_and_frame_name(category=category, sequence_name=sequence_name, name=name))
#     """
#
#
#     """"
#     @staticmethod
#     def meta_rfpath_to_sequence_name(rfpath: Path):
#         return rfpath.parent.stem
#     @staticmethod
#     def meta_rfpath_to_category(rfpath: Path):
#         return rfpath.parent.parent.stem
#
#     @staticmethod
#     def meta_fpath_to_name(fpath: Path):
#         return fpath.stem
#     @staticmethod
#     def meta_rfpath_to_name(rfpath: Path):
#         return rfpath.stem
#
#
#
#
#
#     """
#
#     """
#     @staticmethod
#     def get_frames_names_with_category_sequence_name(path_meta, category, sequence_name, count_max_per_sequence=None):
#         frames_fpath = list(
#             CO3D_FrameMeta.get_path_frames_meta_with_category_sequence(path_meta=path_meta, category=category,
#                                                                        sequence=sequence_name).iterdir())
#         frames_names = [CO3D_FrameMeta.meta_fpath_to_name(fpath) for fpath in frames_fpath]
#
#         frames_names = CO3D_FrameMeta.get_subset_frames_names_uniform(frames_names,
#                                                                       count_max_per_sequence=count_max_per_sequence)
#
#         frames_names = sorted(frames_names, key=lambda frame_name: int(frame_name))
#         return frames_names
#     @staticmethod
#     def get_dict_category_sequence_name_frames_names(path_meta, categories, dict_category_sequences_names,
#                                                      count_max_per_sequence=None, dict_category_sequence_name_frames_names=None):
#         new_dict_category_sequence_name_frames_names = {}
#         for category in dict_category_sequences_names.keys():
#             if dict_category_sequence_name_frames_names is None or (category in dict_category_sequence_name_frames_names.keys()):
#                 new_dict_category_sequence_name_frames_names[category] = {}
#                 for sequence_name in dict_category_sequences_names[category]:
#                     if dict_category_sequence_name_frames_names is None or \
#                             dict_category_sequence_name_frames_names[category] is None or \
#                             dict_category_sequence_name_frames_names[category][sequence_name] is None:
#                         frames_names = CO3D_FrameMeta.get_frames_names_with_category_sequence_name(path_meta=path_meta,
#                                                                                                    category=category,
#                                                                                                    sequence_name=sequence_name,
#                                                                                                    count_max_per_sequence=
#                                                                                                    count_max_per_sequence)
#                     else:
#                         frames_names = dict_category_sequence_name_frames_names[category][sequence_name]
#                         frames_names = CO3D_FrameMeta.get_subset_frames_names_uniform(frames_names,
#                                                                                       count_max_per_sequence=count_max_per_sequence)
#                     new_dict_category_sequence_name_frames_names[category][sequence_name] = frames_names
#         return new_dict_category_sequence_name_frames_names
#
#     @staticmethod
#     def get_dict_category_sequence_name_frames_names_with_names_unique(names_unique: List[str]):
#         dict_category_sequence_name_frames_names = {}
#         for name_unique in names_unique:
#             category, sequence_name, name = name_unique.split('/')
#             if category not in dict_category_sequence_name_frames_names:
#                 dict_category_sequence_name_frames_names[category] = {}
#             if sequence_name not in dict_category_sequence_name_frames_names[category]:
#                 dict_category_sequence_name_frames_names[category][sequence_name] = []
#             dict_category_sequence_name_frames_names[category][sequence_name].append(name_unique)
#         return dict_category_sequence_name_frames_names
#     """
#     """
#     # legacy code
#     @staticmethod
#     def get_rfpaths_frames_meta(path_meta, categories, map_category_sequences_names, count_max_per_sequence=None, return_sequences_lengths=False):
#         sequences_lengths = []
#         frames_rfpaths = []
#         for category in categories:
#             for sequence_name in map_category_sequences_names[category]:
#                 frames_fpaths_partial = list(CO3D_FrameMeta.get_path_frames_meta_with_category_sequence(path_meta=path_meta, category=category, sequence=sequence_name).iterdir())
#                 frames_rfpaths_partial = [CO3D_FrameMeta.get_rfpath_frame_meta_with_category_sequence_name(category=category, sequence=sequence_name, name=fpath.stem) for fpath in frames_fpaths_partial]
#                 if count_max_per_sequence is not None:
#                     frames_rfpaths_partial = [frames_rfpaths_partial[fid] for fid in np.linspace(0, len(frames_rfpaths_partial)-1, count_max_per_sequence).astype(int).tolist()]
#                 frames_rfpaths_partial = sorted(frames_rfpaths_partial, key=lambda rfpath: int(rfpath.stem))
#                 frames_rfpaths += frames_rfpaths_partial
#                 sequences_lengths.append(len(frames_rfpaths_partial))
#         if return_sequences_lengths:
#             return frames_rfpaths, sequences_lengths
#         else:
#             return frames_rfpaths
#     """
#
# class CO3D_Frame(OD3D_Frame):
#     def __init__(self, path_raw: Path, path_preprocess: Path, meta: CO3D_FrameMeta, path_meta: Path,
#                  modalities: List[OD3D_FRAME_MODALITIES], categories: List[str],
#                  cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D.value,
#                  cuboid_source=CUBOID_SOURCES.DEFAULT.value,
#                  pcl_source: PCL_SOURCES=PCL_SOURCES.CO3D,
#                  aligned_name:str=None, mesh_name: str='default'):
#         super().__init__(path_raw=path_raw, path_preprocess=path_preprocess, path_meta=path_meta, meta=meta, modalities=modalities, categories=categories)
#
#         self.meta = meta
#         self.path_meta: Path = path_meta
#         self._sequence = None
#         self.cam_tform_obj_source = cam_tform_obj_source
#         self.aligned_name = aligned_name
#         self.pcl_source = pcl_source
#         self.cuboid_source = cuboid_source
#         self.mesh_name = mesh_name
#         # the following variables can be configured dynamically
#         # self._config = None
#
#     @staticmethod
#     def create_with_config(config, **kwargs):
#         co3d_seq = CO3D_Frame(**kwargs)
#         co3d_seq._config = config
#         return co3d_seq
#
#     #@property
#     #def config(self):
#     #    if self._config is None:
#     #        self._config = OmegaConf.create()
#     #        self._config.cam_tform_obj_source = CAM_TFORM_OBJ_SOURCES.KPTS2D_ORIENT_AND_PCL.value
#     #       self._config.cuboid_source = CUBOID_SOURCES.KPTS2D_ORIENT_AND_PCL.value
#     #    return self._config
#
#
#     @property
#     def sequence(self):
#         if self._sequence is None:
#             from od3d.datasets.co3d.sequence import CO3D_Sequence, CO3D_SequenceMeta
#             sequence_meta = CO3D_SequenceMeta.load_from_meta_with_category_and_name(path_meta=self.path_meta,
#                                                                                     category=self.category,
#                                                                                     name=self.meta.sequence_name)
#             self._sequence = CO3D_Sequence(path_raw=self.path_raw, path_preprocess=self.path_preprocess,
#                                            path_meta=self.path_meta, meta=sequence_meta, modalities=self.modalities,
#                                            categories=self.all_categories, cam_tform_obj_source=self.cam_tform_obj_source,
#                                            aligned_name=self.aligned_name, mesh_name=self.mesh_name, pcl_source=self.pcl_source,
#                                            cuboid_source=self.cuboid_source)
#         return self._sequence
#
#     @property
#     def depth(self):
#         if self._depth is None:
#             self._depth = read_co3d_depth_image(self.fpath_depth) * self.meta.depth_scale
#         return self._depth
#
#     @depth.setter
#     def depth(self, value: torch.Tensor):
#             self._depth = value
#
#     @property
#     def fpath_mesh(self):
#         return self.sequence.fpath_mesh
#
#     @property
#     def mesh(self):
#         return self.sequence.mesh
#
#     @property
#     def fpath_cam_tform4x4_pcl(self):
#         fpath = self.path_preprocess.joinpath('cam_tform4x4_obj', self.pcl_source, self.name_unique + '.pt')
#         if not fpath.exists():
#             self.sequence.preprocess_pcl()
#         return fpath
#
#     @property
#     def cam_tform4x4_obj(self):
#         if self._cam_tform4x4_obj is None:
#             self._cam_tform4x4_obj = self.get_cam_tform4x4_obj(cam_tform_obj_source=self.cam_tform_obj_source)
#         return self._cam_tform4x4_obj
#
#     @cam_tform4x4_obj.setter
#     def cam_tform4x4_obj(self, value: torch.Tensor):
#             self._cam_tform4x4_obj = value
#
#
#     def get_cam_tform4x4_obj(self, cam_tform_obj_source: CAM_TFORM_OBJ_SOURCES):
#         # droid_slam_zsp_labeled
#         if cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.LABELED:
#             _cam_tform4x4_obj = torch.load(self.fpath_cam_tform4x4_pcl)
#             _cam_tform4x4_obj = tform4x4(_cam_tform4x4_obj, inv_tform4x4(self.sequence.labeled_obj_tform_obj))
#             # note: note alignment of droid slam may include scale, therefore remove this scale.
#             # note: projection does not change as we scale the depth z to the object as well
#             _scale = _cam_tform4x4_obj[:3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
#             _cam_tform4x4_obj[:3] = _cam_tform4x4_obj[:3] / _scale
#             # logger.info(f'det: {torch.linalg.det(_cam_tform4x4_obj[:3, :3])}')
#         elif cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.LABELED_CUBOID:
#             _cam_tform4x4_obj = torch.load(self.fpath_cam_tform4x4_pcl)
#             _cam_tform4x4_obj = tform4x4(_cam_tform4x4_obj, inv_tform4x4(self.sequence.labeled_cuboid_obj_tform_obj))
#             # note: not alignment of droid slam may include scale, therefore remove this scale.
#             # note: projection does not change as we scale the depth z to the object as well
#             _scale = _cam_tform4x4_obj[:3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
#             _cam_tform4x4_obj[:3] = _cam_tform4x4_obj[:3] / _scale
#             # logger.info(f'det: {torch.linalg.det(_cam_tform4x4_obj[:3, :3])}')
#         elif cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.ALIGNED:
#             _cam_tform4x4_obj = torch.load(self.fpath_cam_tform4x4_pcl)
#             _aligned_obj_tform_obj = self.sequence.aligned_obj_tform_obj
#             _cam_tform4x4_obj = tform4x4(_cam_tform4x4_obj, inv_tform4x4(_aligned_obj_tform_obj))
#             # note: not alignment of droid slam may include scale, therefore remove this scale.
#             # note: projection does not change as we scale the depth z to the object as well
#             _scale = _cam_tform4x4_obj[:3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
#             _cam_tform4x4_obj[:3] = _cam_tform4x4_obj[:3] / _scale
#         elif cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.ZSP_LABELED:
#             _cam_tform4x4_obj_obj = torch.load(self.fpath_cam_tform4x4_pcl)
#             co3dv1_zsp_obj_tform_obj = self.sequence.co3dv1_zsp_obj_tform_obj
#             _cam_tform4x4_obj = tform4x4(_cam_tform4x4_obj_obj, inv_tform4x4(co3dv1_zsp_obj_tform_obj))
#             # note: not alignment of droid slam may include scale, therefore remove this scale.
#             # note: projection does not change as we scale the depth z to the object as well
#             _scale = _cam_tform4x4_obj[:3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
#             _cam_tform4x4_obj[:3] = _cam_tform4x4_obj[:3] / _scale
#         elif cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.ZSP_LABELED_CUBOID_REF:
#             _cam_tform4x4_obj_obj = torch.load(self.fpath_cam_tform4x4_pcl)
#             zsp_labeled_cuboid_ref_tform_obj = self.sequence.zsp_labeled_cuboid_ref_tform_obj
#             _cam_tform4x4_obj = tform4x4(_cam_tform4x4_obj_obj, inv_tform4x4(zsp_labeled_cuboid_ref_tform_obj))
#             # note: not alignment of droid slam may include scale, therefore remove this scale.
#             # note: projection does not change as we scale the depth z to the object as well
#             _scale = _cam_tform4x4_obj[:3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
#             _cam_tform4x4_obj[:3] = _cam_tform4x4_obj[:3] / _scale
#             # logger.info(f'det: {torch.linalg.det(_cam_tform4x4_obj[:3, :3])}')
#         # if cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.FRONT_FRAME_AND_PCL and self.sequence.fpath_cuboid_front_tform4x4_obj.exists():
#         #     _cam_tform4x4_obj = tform4x4(torch.Tensor(self.meta.l_cam_tform4x4_obj),
#         #                                       inv_tform4x4(self.sequence.cuboid_front_tform4x4_obj))
#         # elif cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.KPTS2D_ORIENT_AND_PCL and self.sequence.fpath_cuboid_front_tform4x4_obj.exists():
#         #     _cam_tform4x4_obj = tform4x4(torch.Tensor(self.meta.l_cam_tform4x4_obj),
#         #                                       inv_tform4x4(self.sequence.cuboid_front_tform4x4_obj))
#         # elif cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.LIMITS3D:
#         #     _cam_tform4x4_obj = tform4x4(
#         #         torch.Tensor(self.meta.l_cam_tform4x4_obj) and self.sequence.fpath_cuboid_front_tform4x4_obj.exists(),
#         #         inv_tform4x4(self.sequence.cuboid_front_tform4x4_obj))
#         # elif cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM and self.fpath_cam_tform4x4_obj_droid_slam.exists():
#         #     _cam_tform4x4_obj = torch.load(self.fpath_cam_tform4x4_obj_droid_slam)
#         # elif cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ALIGNED and self.fpath_cam_tform4x4_obj_droid_slam.exists():
#         #     _cam_tform4x4_obj_droid_slam = torch.load(self.fpath_cam_tform4x4_obj_droid_slam)
#         #     _droid_slam_aligned_tform_droid_slam = self.sequence.droid_slam_aligned_tform_droid_slam
#         #     _cam_tform4x4_obj = tform4x4(_cam_tform4x4_obj_droid_slam, inv_tform4x4(_droid_slam_aligned_tform_droid_slam))
#         #     # note: not alignment of droid slam may include scale, therefore remove this scale.
#         #     # note: projection does not change as we scale the depth z to the object as well
#         #     _scale = _cam_tform4x4_obj[:3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
#         #     _cam_tform4x4_obj[:3] = _cam_tform4x4_obj[:3] / _scale
#         # elif cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ZSP and self.fpath_cam_tform4x4_obj_droid_slam.exists():
#         #     _cam_tform4x4_obj_droid_slam = torch.load(self.fpath_cam_tform4x4_obj_droid_slam)
#         #     co3dv1_zsp_obj_tform_droid_slam_obj = self.sequence.co3dv1_zsp_obj_tform_droid_slam_obj
#         #     _cam_tform4x4_obj = tform4x4(_cam_tform4x4_obj_droid_slam, inv_tform4x4(co3dv1_zsp_obj_tform_droid_slam_obj))
#         #     # note: not alignment of droid slam may include scale, therefore remove this scale.
#         #     # note: projection does not change as we scale the depth z to the object as well
#         #     _scale = _cam_tform4x4_obj[:3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
#         #     _cam_tform4x4_obj[:3] = _cam_tform4x4_obj[:3] / _scale
#         # elif cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_ZSP_LABELED:
#         #     #logger.info(f'droid slam frame fpath {self.fpath_cam_tform4x4_obj_droid_slam}')
#         #
#         #     _cam_tform4x4_obj_droid_slam = torch.load(self.fpath_cam_tform4x4_obj_droid_slam)
#         #     zsp_labeled_ref_tform_droid_slam_obj = self.sequence.zsp_labeled_cuboid_ref_tform_droid_slam_obj
#         #     _cam_tform4x4_obj = tform4x4(_cam_tform4x4_obj_droid_slam, inv_tform4x4(zsp_labeled_ref_tform_droid_slam_obj))
#         #     # note: not alignment of droid slam may include scale, therefore remove this scale.
#         #     # note: projection does not change as we scale the depth z to the object as well
#         #     _scale = _cam_tform4x4_obj[:3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
#         #     _cam_tform4x4_obj[:3] = _cam_tform4x4_obj[:3] / _scale
#         #     # logger.info(f'det: {torch.linalg.det(_cam_tform4x4_obj[:3, :3])}')
#         # elif cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.DROID_SLAM_LABELED:
#         #     _cam_tform4x4_obj_droid_slam = torch.load(self.fpath_cam_tform4x4_obj_droid_slam)
#         #     droid_slam_labeled_cuboid_tform_droid_slam = tform4x4(self.sequence.labeled_cuboid_obj_tform_labeled_obj, self.sequence.labeled_obj_tform_obj)
#         #     _cam_tform4x4_obj = tform4x4(_cam_tform4x4_obj_droid_slam, inv_tform4x4(droid_slam_labeled_cuboid_tform_droid_slam))
#         #     # note: not alignment of droid slam may include scale, therefore remove this scale.
#         #     # note: projection does not change as we scale the depth z to the object as well
#         #     _scale = _cam_tform4x4_obj[:3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
#         #     _cam_tform4x4_obj[:3] = _cam_tform4x4_obj[:3] / _scale
#         #     # logger.info(f'det: {torch.linalg.det(_cam_tform4x4_obj[:3, :3])}')
#         elif cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.CO3DV1:
#             fpath_metav1 = self.path_preprocess.joinpath('..', 'CO3Dv1_Preprocess', 'meta', self.meta.rfpath)
#             if fpath_metav1.exists():
#                 metav1 = CO3D_FrameMeta.load_from_meta_with_rfpath(path_meta=self.path_preprocess.joinpath('..', 'CO3Dv1_Preprocess', 'meta'), rfpath=self.meta.rfpath)
#                 if metav1 is None:
#                     raise NotImplementedError
#                 _cam_tform4x4_obj = torch.Tensor(metav1.l_cam_tform4x4_obj)
#             else:
#                 logger.warning(f'missing CO3Dv1 fpath to meta {fpath_metav1}')
#                 _cam_tform4x4_obj = torch.ones(size=(4, 4)) * torch.nan
#         elif cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.PCL and not self.pcl_source == PCL_SOURCES.CO3D:
#
#             _cam_tform4x4_obj = torch.load(self.fpath_cam_tform4x4_pcl).detach().cpu()
#         else:
#             _cam_tform4x4_obj = torch.Tensor(self.meta.l_cam_tform4x4_obj)
#             if cam_tform_obj_source != CAM_TFORM_OBJ_SOURCES.CO3D and not (cam_tform_obj_source == CAM_TFORM_OBJ_SOURCES.PCL and self.pcl_source == PCL_SOURCES.CO3D):
#                 logger.warning(f'No fpath available for cam source {cam_tform_obj_source}.')
#         return _cam_tform4x4_obj