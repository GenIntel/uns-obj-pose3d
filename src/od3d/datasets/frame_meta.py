import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import torch
from dataclasses import dataclass
from typing import List, Union, Dict
from od3d.datasets.meta import OD3D_Meta

@dataclass
class OD3D_FrameMetaBBoxMixin():
    # x0, y0, x1, y1
    l_bbox: List[float]
    @property
    def bbox(self):
        return torch.Tensor(self.l_bbox)

@dataclass
class OD3D_FrameMetaBBoxsMixin():
    l_bboxs: List[List[float]]
    @property
    def bboxs(self):
        return torch.Tensor(self.l_bboxs)

    @property
    def bbox(self):
        return self.bboxs[0]

@dataclass
class OD3D_FrameMetaKpts2D3DMixin():
    l_kpts2d_annot: List[List[float]]
    l_kpts2d_annot_vsbl: List[bool]
    l_kpts3d: List[List[float]]
    kpts_names: List[str]

    @property
    def kpts2d_annot(self):
        return torch.Tensor(self.l_kpts2d_annot)
    @property
    def kpts2d_annot_vsbl(self):
        return torch.Tensor(self.l_kpts2d_annot_vsbl).to(dtype=bool)
    @property
    def kpts3d(self):
        return torch.Tensor(self.l_kpts3d)

@dataclass
class OD3D_FrameMetaSubsetMixin:
    subset: str

@dataclass
class OD3D_FrameMetaMeshMixin():
    rfpath_mesh: Path

@dataclass
class OD3D_FrameMetaMeshsMixin():
    rfpaths_meshs: List[Path]

    @property
    def rfpath_mesh(self):
        return self.rfpaths_meshs[0]

@dataclass
class OD3D_FrameMetaSequenceMixin:
    sequence_name: str
    @property
    def name_unique(self):
        return f'{self.sequence_name}/{super().name_unique}'

    @property
    def sequence_name_unique(self):
        return f'{self.sequence_name}'


@dataclass
class OD3D_FrameMetaCategoryMixin:
    category: str

    @property
    def name_unique(self):
        return f'{self.category}/{super().name_unique}'

    @property
    def sequence_name_unique(self):
        return f'{self.category}/{super().sequence_name_unique}'


@dataclass
class OD3D_FrameMetaCategoriesMixin:
    categories: List[str]

    @property
    def category(self):
        return self.categories[0]

@dataclass
class OD3D_FrameMetaCamIntr4x4Mixin:
    l_cam_intr4x4: List[List[float]]  # torch.Tensor

    @property
    def cam_intr4x4(self):
        return torch.Tensor(self.l_cam_intr4x4)

@dataclass
class OD3D_FrameMetaCamTform4x4ObjMixin:
    l_cam_tform4x4_obj: List[List[float]]  # torch.Tensor

    @property
    def cam_tform4x4_obj(self):
        return torch.Tensor(self.l_cam_tform4x4_obj)

@dataclass
class OD3D_FrameMetaCamTform4x4ObjsMixin:
    l_cam_tform4x4_objs: List[List[List[float]]]  # torch.Tensor

    @property
    def cam_tform4x4_objs(self):
        return torch.Tensor(self.l_cam_tform4x4_objs)

    @property
    def cam_tform4x4_obj(self):
        return self.cam_tform4x4_objs[0]

@dataclass
class OD3D_FrameMetaSizeMixin:
    l_size: List[float] # torch.Tensor
    @property
    def size(self):
        return torch.Tensor(self.l_size)
    @property
    def H(self):
        return self.size[0]
    @property
    def W(self):
        return self.size[1]

@dataclass
class OD3D_FrameMetaRGBMixin:
    rfpath_rgb: Path

@dataclass
class OD3D_FrameMetaMaskMixin:
    rfpath_mask: Path

@dataclass
class OD3D_FrameMetaMasksMixin():
    rfpaths_masks: List[Path]

    @property
    def rfpaths_mask(self):
        return self.rfpaths_masks[0]

@dataclass
class OD3D_FrameMetaDepthMixin:
    rfpath_depth: Path

@dataclass
class OD3D_FrameMetaDepthMaskMixin:
    rfpath_depth_mask: Path

@dataclass
class OD3D_FrameMetaPCLMixin:
    rfpath_pcl: Path

@dataclass
class OD3D_FrameMeta(OD3D_Meta):
    @classmethod
    def get_rfpath_metas(cls):
        return Path("frames")


OD3D_FrameMetaClasses = Union[OD3D_FrameMeta, OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaCategoriesMixin,
                              OD3D_FrameMetaCamTform4x4ObjMixin, OD3D_FrameMetaCamIntr4x4Mixin,
                              OD3D_FrameMetaMaskMixin, OD3D_FrameMetaDepthMaskMixin, OD3D_FrameMetaDepthMixin,
                              OD3D_FrameMetaPCLMixin, OD3D_FrameMetaSizeMixin, OD3D_FrameMetaRGBMixin,
                              OD3D_FrameMetaMeshMixin, OD3D_FrameMetaSequenceMixin, OD3D_FrameMetaSubsetMixin,
                              OD3D_FrameMetaBBoxMixin, OD3D_FrameMetaKpts2D3DMixin]