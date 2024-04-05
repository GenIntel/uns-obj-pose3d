import logging
logger = logging.getLogger(__name__)
from typing import List
import torch
from pathlib import Path

from od3d.datasets.object import OD3D_CAM_TFORM_OBJ_TYPES
from od3d.datasets.frame_meta import OD3D_FrameMeta, OD3D_FrameMetaRGBMixin,  \
    OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaMaskMixin, OD3D_FrameMetaSizeMixin, OD3D_FrameMetaSequenceMixin

from od3d.datasets.frame import OD3D_Frame, OD3D_FrameSizeMixin, OD3D_FrameRGBMixin, OD3D_FrameMaskMixin, \
    OD3D_FrameRGBMaskMixin, OD3D_FrameRaysCenter3dMixin, OD3D_FrameMeshMixin, OD3D_FrameTformObjMixin, \
    OD3D_FrameSequenceMixin, OD3D_FrameCategoryMixin, OD3D_FrameCamIntr4x4Mixin, OD3D_FrameCamTform4x4ObjMixin, \
    OD3D_CamProj4x4ObjMixin, OD3D_FRAME_MASK_TYPES
from dataclasses import dataclass
import numpy as np
from od3d.datasets.object import OD3D_PCLTypeMixin, OD3D_MeshTypeMixin, OD3D_SequenceSfMTypeMixin
from od3d.datasets.monolmb.enum import MAP_CATEGORIES_MONOLMB_TO_OD3D


@dataclass
class MonoLMB_FrameMeta(OD3D_FrameMetaRGBMixin, OD3D_FrameMetaSizeMixin, OD3D_FrameMetaCategoryMixin,
                        OD3D_FrameMetaSequenceMixin, OD3D_FrameMeta):

    @staticmethod
    def load_from_raw(name: str, category: str, sequence_name: str, rfpath_rgb: Path, l_size: List):
        return MonoLMB_FrameMeta(rfpath_rgb=rfpath_rgb, category=category, sequence_name=sequence_name, l_size=l_size, name=name)

@dataclass
class MonoLMB_Frame(OD3D_FrameMeshMixin, OD3D_FrameRaysCenter3dMixin, OD3D_FrameTformObjMixin, OD3D_CamProj4x4ObjMixin,
                    OD3D_FrameRGBMaskMixin, OD3D_FrameMaskMixin, OD3D_FrameRGBMixin, OD3D_FrameCategoryMixin,
                    OD3D_FrameSequenceMixin, OD3D_FrameSizeMixin, OD3D_MeshTypeMixin,
                    OD3D_PCLTypeMixin, OD3D_SequenceSfMTypeMixin, OD3D_Frame):
    meta_type = MonoLMB_FrameMeta
    map_categories_to_od3d = MAP_CATEGORIES_MONOLMB_TO_OD3D

    def __post_init__(self):
        # hack: prevents circular import
        from od3d.datasets.monolmb.sequence import MonoLMB_Sequence
        self.sequence_type = MonoLMB_Sequence

    def get_cam_intr4x4(self):
        if self.cam_intr4x4 is None:
            self.cam_intr4x4 = self.read_cam_intr4x4()
        return self.cam_intr4x4

    def read_cam_intr4x4(self):
        if self.cam_intr4x4 is None:
            self.cam_intr4x4 = torch.eye(4)
            px = self.meta.l_size[1] / 2
            py = self.meta.l_size[0] / 2
            fov = 1.7  # fov in radians
            fx = max(self.meta.l_size[0], self.meta.l_size[1]) / np.tan(fov / 2)
            fy = fx
            self.cam_intr4x4[0, 0] = fx
            self.cam_intr4x4[1, 1] = fy
            self.cam_intr4x4[0, 2] = px
            self.cam_intr4x4[1, 2] = py
        return self.cam_intr4x4.clone()
