import logging
logger = logging.getLogger(__name__)
from omegaconf import OmegaConf, DictConfig
import torch
from typing import List
from pathlib import Path
from dataclasses import dataclass
from od3d.datasets.frame import OD3D_FrameMeta, OD3D_Frame
import scipy.io
import math
import numpy as np
from od3d.cv.geometry.transform import transf4x4_from_spherical
from od3d.datasets.dataset import OD3D_Frames, OD3D_FRAME_MODALITIES
from od3d.datasets.pascal3d.enum import PASCAL3D_SCALE_NORMALIZE_TO_REAL, PASCAL3D_CATEGORIES
from od3d.datasets.pascal3d.frame import Pascal3DFrame, Pascal3DFrameMeta
from od3d.cv.io import read_image, write_mask_image
from od3d.cv.geometry.mesh import Mesh, Meshes

from od3d.datasets.frame_meta import OD3D_FrameMetaMaskMixin


@dataclass
class Pascal3D_OccFrameMeta(OD3D_FrameMetaMaskMixin, Pascal3DFrameMeta):
    pass

class Pascal3D_OccFrame(OD3D_Frame):
    def __init__(self, path_raw: Path, path_preprocess: Path, path_meta: Path, path_meshes: Path, meta: Pascal3DFrameMeta, modalities: List[OD3D_FRAME_MODALITIES], categories: List[str]):
        super().__init__(path_raw=path_raw, path_preprocess=path_preprocess, path_meta=path_meta, meta=meta, modalities=modalities, categories=categories)
        self.path_meshes = path_meshes

    @property
    def mask(self):
        if self._mask is None:
            fpath = self.fpath_mask
            annot = np.load(fpath)
            self._mask = (torch.from_numpy(annot['mask'])[None,] / 255. - torch.from_numpy(annot['occluder_mask'])[None,] * 1.).clamp(0., 1.) # occluder_mask, mask
        return self._mask
    @mask.setter
    def mask(self, value: torch.Tensor):
            self._mask = value

    @property
    def cam_tform4x4_obj(self):
        if self._cam_tform4x4_obj is None:
            self._cam_tform4x4_obj = torch.Tensor(self.meta.l_cam_tform4x4_obj)
            self._cam_tform4x4_obj[2, 3] *= PASCAL3D_SCALE_NORMALIZE_TO_REAL[self.category]
        return self._cam_tform4x4_obj

    @property
    def kpts3d(self):
        if self._kpts3d is None:
            self._kpts3d = self.meta.kpts3d * PASCAL3D_SCALE_NORMALIZE_TO_REAL[self.category]
        return self._kpts3d

    @property
    def fpath_mesh(self):
        return self.path_meshes.parent.joinpath(self.meta.rfpath_mesh)

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = Mesh.load_from_file(fpath=self.fpath_mesh, scale=PASCAL3D_SCALE_NORMALIZE_TO_REAL[self.category])
        return self._mesh

    @staticmethod
    def calc_cam_tform_obj(azimuth, elevation, theta, distance):
        cam_tform4x4_obj = transf4x4_from_spherical(
            azim=torch.Tensor([azimuth]),
            elev=torch.Tensor([elevation]),
            theta=torch.Tensor([theta]),
            dist=torch.Tensor([distance]))[0]
        # cam_tform4x4_obj[0, :] = cam_tform4x4_obj[0, :]
        # cam_tform4x4_obj[1, :] = -cam_tform4x4_obj[1, :]
        # cam_tform4x4_obj[2, :] = -cam_tform4x4_obj[2, :]
        # cam_tform4x4_obj[2, 2:3] = -cam_tform4x4_obj[2, 2:3]
        return cam_tform4x4_obj