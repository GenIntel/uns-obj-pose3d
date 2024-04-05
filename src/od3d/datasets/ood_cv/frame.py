import logging

import torchvision.io.image

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
import torchvision
from od3d.datasets.frame_meta import OD3D_FrameMetaMaskMixin

@dataclass
class OOD_CV_FrameMeta(Pascal3DFrameMeta):
    pass

    @staticmethod
    def load_from_raw_annotation(annotation, subset: str, category: str, path_raw: Path, rpath_meshes: Path, rfpath_rgb: Path, path_pascal3d_raw: Path=None):
        name = annotation['record']['filename'][0][0][0].split('.')[0]

        objects = annotation['record']['objects'][0][0][0]

        objects = list(filter(lambda obj: hasattr(obj, 'dtype') and obj.dtype.names is not None and 'viewpoint' in obj.dtype.names and hasattr(obj['viewpoint'], 'dtype') and obj['viewpoint'].dtype.names is not None , objects))
        # assert len(objects) == 1
        if len(objects) < 1:
            incomplete_reason = f"num objects = {len(objects)}"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None

        object = objects[0]

        if not hasattr(object, 'dtype') or object.dtype.names is None or 'viewpoint' not in object.dtype.names: #['focal'][0][0][0][0] == 0:
            incomplete_reason = "viewpoint missing"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None

        if not hasattr(object['viewpoint'], 'dtype') or object['viewpoint'].dtype.names is None: #['focal'][0][0][0][0] == 0:
            incomplete_reason = "viewpoint missing"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None

        #if 'focal' not in object['viewpoint'].dtype.names or object['viewpoint']['focal'][0][0][0][0] == 0:
        #    object['viewpoint']['focal'][0][0][0][0] = 3000

        if object['viewpoint']['px'][0][0][0][0] < 0:
            incomplete_reason = f"negative px {object['viewpoint']['px'][0][0][0][0]}"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None
            #object['viewpoint']['px'][0][0][0][0] = -object['viewpoint']['px'][0][0][0][0]

        if object['viewpoint']['py'][0][0][0][0] < 0:
            incomplete_reason = f"negative py {object['viewpoint']['py'][0][0][0][0]}"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None
            #object['viewpoint']['py'][0][0][0][0] = -object['viewpoint']['py'][0][0][0][0]

        if object['viewpoint']['distance'][0][0][0][0] < 0.01:
            incomplete_reason = f"distance negative {object['viewpoint']['distance'][0][0][0][0]}"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None

        if not hasattr(object['anchors'][0][0], 'dtype') or object['anchors'][0][0].dtype.names is None:
            incomplete_reason = "kpts names missing"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None

        fpath_rgb = Path(path_raw).joinpath(rfpath_rgb)
        img = torchvision.io.image.read_image(str(fpath_rgb))
        size = torch.LongTensor([*img.shape[1:]])


        category, mesh_index, rfpath_mesh, bbox, kpts_names, kpts2d_annot, kpts2d_annot_vsbl, cam_tform4x4_obj, \
            cam_intr4x4 \
            = Pascal3DFrameMeta.load_category_mesh_bbox_kpts2d_cam_from_object_annotation_raw(object=object,
                                                                                              rpath_meshes=rpath_meshes)

        # logger.info(cam_intr4x4)



        kpts3d = Pascal3DFrameMeta.load_kpts3d_from_raw(path_pascal3d_raw, rpath_meshes, category, mesh_index, kpts_names)

        return OOD_CV_FrameMeta(subset=subset, name=name, rfpath_rgb=rfpath_rgb, rfpath_mesh=rfpath_mesh,
                                l_bbox=bbox.tolist(), kpts_names=kpts_names, l_kpts2d_annot=kpts2d_annot.tolist(),
                                l_kpts2d_annot_vsbl=kpts2d_annot_vsbl.tolist(), l_size=size.tolist(),
                                l_cam_tform4x4_obj=cam_tform4x4_obj.tolist(), l_cam_intr4x4=cam_intr4x4.tolist(),
                                l_kpts3d=kpts3d.tolist(), category=category)


class OOD_CV_Frame(Pascal3DFrame):
    def __init__(self, path_raw: Path, path_preprocess: Path, path_meta: Path, path_meshes: Path, meta: Pascal3DFrameMeta, modalities: List[OD3D_FRAME_MODALITIES], categories: List[str]):
        super().__init__(path_raw=path_raw, path_preprocess=path_preprocess, path_meta=path_meta, meta=meta, modalities=modalities, categories=categories, path_meshes=path_meshes)
        self.path_meshes = path_meshes
