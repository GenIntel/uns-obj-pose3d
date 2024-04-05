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
from od3d.cv.io import read_image, write_mask_image, write_depth_image, read_depth_image
from od3d.cv.geometry.mesh import Mesh, Meshes
from od3d.cv.geometry.mesh import Meshes, MESH_RENDER_MODALITIES

from od3d.datasets.frame_meta import OD3D_FrameMeta, \
    OD3D_FrameMetaMeshMixin, OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaRGBMixin, \
    OD3D_FrameMetaSizeMixin, OD3D_FrameMetaKpts2D3DMixin, OD3D_FrameMetaBBoxMixin, OD3D_FrameMetaSubsetMixin, \
    OD3D_FrameMetaCamTform4x4ObjMixin, OD3D_FrameMetaCamIntr4x4Mixin


@dataclass
class Pascal3DFrameMeta(OD3D_FrameMetaKpts2D3DMixin, OD3D_FrameMetaBBoxMixin, OD3D_FrameMetaSubsetMixin,
                        OD3D_FrameMetaMeshMixin, OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaCamTform4x4ObjMixin,
                        OD3D_FrameMetaCamIntr4x4Mixin, OD3D_FrameMetaRGBMixin, OD3D_FrameMetaSizeMixin, OD3D_FrameMeta):

    @property
    def name_unique(self):
        return f'{self.subset}/{self.category}/{self.name}'

    @staticmethod
    def get_name_unique_from_category_subset_name(category, subset, name):
        return f'{subset}/{category}/{name}'

    @staticmethod
    def load_category_mesh_bbox_kpts2d_cam_from_object_annotation_raw(object, rpath_meshes):

        category = object['class'][0]

        mesh_index = object['cad_index'][0][0] - 1
        # label = classes.index(meta.category)
        bbox = torch.from_numpy(object['bbox'][0])
        kpts_names = list(object['anchors'][0][0].dtype.names)
        kpts2d_annot = np.stack([object['anchors'][0][0][n]['location'][0][0][0] if object['anchors'][0][0][n][
                                                                                        'status'] == 1 else np.array(
            [0, 0]) for n in kpts_names])
        kpts2d_annot = torch.from_numpy(kpts2d_annot)
        kpts2d_annot_vsbl = np.array([True if object['anchors'][0][0][n]['status'] == 1 else False for n in kpts_names])
        kpts2d_annot_vsbl = torch.from_numpy(kpts2d_annot_vsbl)

        viewpoint = object['viewpoint']
        azimuth = viewpoint['azimuth'][0][0][0][0] * math.pi / 180
        elevation = viewpoint['elevation'][0][0][0][0] * math.pi / 180
        distance = viewpoint['distance'][0][0][0][0]
        focal = viewpoint['focal'][0][0][0][0]


        theta = viewpoint['theta'][0][0][0][0] * math.pi / 180
        principal = np.array([viewpoint['px'][0][0][0][0],
                              viewpoint['py'][0][0][0][0]])
        viewport = viewpoint['viewport'][0][0][0][0]

        cam_tform4x4_obj = Pascal3DFrame.calc_cam_tform_obj(azimuth=azimuth, elevation=elevation, theta=theta,
                                                            distance=distance)
        # cam_tform4x4_obj = torch.from_numpy(cam_tform4x4_obj)

        cam_intr3x3 = np.array([[1. * viewport * focal, 0, principal[0]],
                                [0, 1. * viewport * focal, principal[1]],
                                [0, 0, 1.]])
        cam_intr4x4 = np.hstack((cam_intr3x3, [[0], [0], [0]]))
        cam_intr4x4 = np.vstack((cam_intr4x4, [0, 0, 0, 1]))
        cam_intr4x4 = torch.from_numpy(cam_intr4x4).to(dtype=cam_tform4x4_obj.dtype)

        rfpath_mesh = rpath_meshes.joinpath(category, f"{(mesh_index + 1):02d}.off")
        return category, mesh_index, rfpath_mesh, bbox, kpts_names, kpts2d_annot, kpts2d_annot_vsbl, cam_tform4x4_obj, cam_intr4x4

    @staticmethod
    def load_kpts3d_from_raw(path_raw, rpath_meshes, category, mesh_index, kpts_names):
        fpath_mesh_kpoints3d = path_raw.joinpath(rpath_meshes, f"{category}.mat")
        annotation_mesh3d = scipy.io.loadmat(fpath_mesh_kpoints3d)
        kpts3d = np.stack([annotation_mesh3d[category][n][0][mesh_index][0] if len(
            annotation_mesh3d[category][n][0][mesh_index]) > 0 else np.array([np.inf, np.inf, np.inf]) for n in kpts_names])
        kpts3d = torch.from_numpy(kpts3d)
        return kpts3d
    @staticmethod
    def load_from_raw_annotation(annotation, subset: str, category: str, path_raw: Path, rpath_meshes: Path, rfpath_rgb: Path):
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

        W = int(annotation['record'][0][0]['size']['width'][0][0][0][0])
        H = int(annotation['record'][0][0]['size']['height'][0][0][0][0])  # self.rgb.shape[1:]
        size = torch.Tensor([H, W])

        category, mesh_index, rfpath_mesh, bbox, kpts_names, kpts2d_annot, kpts2d_annot_vsbl, cam_tform4x4_obj, \
            cam_intr4x4 \
            = Pascal3DFrameMeta.load_category_mesh_bbox_kpts2d_cam_from_object_annotation_raw(object=object,
                                                                                              rpath_meshes=rpath_meshes)

        # logger.info(cam_intr4x4)

        if cam_intr4x4[0, 0] == 0. or cam_intr4x4[1, 1] == 0.:
            incomplete_reason = "focal = 0"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None

        kpts3d = Pascal3DFrameMeta.load_kpts3d_from_raw(path_raw, rpath_meshes, category, mesh_index, kpts_names)

        return Pascal3DFrameMeta(subset=subset, name=name, rfpath_rgb=rfpath_rgb, rfpath_mesh=rfpath_mesh,
                                 l_bbox=bbox.tolist(), kpts_names=kpts_names, l_kpts2d_annot=kpts2d_annot.tolist(),
                                 l_kpts2d_annot_vsbl=kpts2d_annot_vsbl.tolist(), l_size=size.tolist(),
                                 l_cam_tform4x4_obj=cam_tform4x4_obj.tolist(), l_cam_intr4x4=cam_intr4x4.tolist(),
                                 l_kpts3d=kpts3d.tolist(), category=category)
    @staticmethod
    def load_from_raw(frame_name: str, subset: str, category: str, path_raw: Path, rpath_meshes: Path):
        frame_rfpath = f"{category}_imagenet/{frame_name}"
        rfpath_annotation = Path("Annotations").joinpath(f"{frame_rfpath}.mat")
        rfpath_rgb = Path("Images").joinpath(f"{frame_rfpath}.JPEG")

        annotation = scipy.io.loadmat(path_raw.joinpath(rfpath_annotation))
        return Pascal3DFrameMeta.load_from_raw_annotation(annotation=annotation, rfpath_rgb=rfpath_rgb,
                                                          rpath_meshes=rpath_meshes, category=category, subset=subset,
                                                          path_raw=path_raw)



from od3d.datasets.pascal3d.enum import MAP_CATEGORIES_PASCAL3D_TO_OD3D
from od3d.datasets.frame import OD3D_FrameMeshMixin, OD3D_FrameTformObjMixin, OD3D_CamProj4x4ObjMixin, \
    OD3D_FrameRGBMaskMixin, OD3D_FrameMaskMixin, OD3D_FrameRGBMixin, OD3D_FrameDepthMixin, OD3D_FrameDepthMaskMixin, \
    OD3D_FrameCategoryMixin, OD3D_FrameSizeMixin, OD3D_Frame, OD3D_FrameBBoxMixin, OD3D_FrameKpts2d3dMixin

from od3d.datasets.object import OD3D_MESH_TYPES
from od3d.cv.geometry.transform import inv_tform4x4, tform4x4

@dataclass
class Pascal3DFrame(OD3D_FrameBBoxMixin, OD3D_FrameMeshMixin, OD3D_FrameTformObjMixin, OD3D_FrameKpts2d3dMixin,
                    OD3D_CamProj4x4ObjMixin, OD3D_FrameRGBMaskMixin, OD3D_FrameMaskMixin, OD3D_FrameRGBMixin,
                    OD3D_FrameDepthMixin, OD3D_FrameDepthMaskMixin, OD3D_FrameCategoryMixin, OD3D_FrameSizeMixin,
                    OD3D_Frame):
    meta_type = Pascal3DFrameMeta
    map_categories_to_od3d = MAP_CATEGORIES_PASCAL3D_TO_OD3D

    @staticmethod
    def get_rpath_raw_categorical_meshes(category: str):
        return Path("CAD", f'{category}')

    @staticmethod
    def get_rfpath_pp_categorical_mesh(mesh_type: OD3D_MESH_TYPES, category: str):
        return Path("mesh", f'{mesh_type}', f'{category}', 'mesh.ply')

    def get_fpath_mesh(self, mesh_type=None):
        if mesh_type is None:
            mesh_type = self.mesh_type
        if mesh_type == OD3D_MESH_TYPES.META:
            return self.path_raw.joinpath(self.meta.rfpath_mesh)
        else:
            return self.path_preprocess.joinpath(self.get_rfpath_pp_categorical_mesh(mesh_type=mesh_type, category=self.category))

    def read_mesh(self, mesh_type=None):
        if mesh_type is None:
            mesh_type = self.mesh_type
        if mesh_type == OD3D_MESH_TYPES.META:
            mesh = Mesh.load_from_file(fpath=self.get_fpath_mesh(mesh_type=mesh_type), scale=PASCAL3D_SCALE_NORMALIZE_TO_REAL[self.category])
        else:
            # note: preprocessed meshes are in real scale
            mesh = Mesh.load_from_file(fpath=self.get_fpath_mesh(mesh_type=mesh_type))

        if mesh_type is None or mesh_type == self.mesh_type:
            self.mesh = mesh
        return mesh

    def get_mesh(self, mesh_type=None, clone=False):
        if (mesh_type is None or mesh_type == self.mesh_type) and self.mesh is not None:
            mesh = self.mesh
        else:
            mesh = self.read_mesh(mesh_type=mesh_type)

        if not clone:
            return mesh
        else:
            return mesh.clone()

    def read_cam_tform4x4_obj_raw(self):
        cam_tform4x4_obj = torch.Tensor(self.meta.cam_tform4x4_obj)
        cam_tform4x4_obj[:3, 3] *= PASCAL3D_SCALE_NORMALIZE_TO_REAL[self.category]
        return cam_tform4x4_obj

    def get_fpath_tform_obj(self, tform_obj_type=None):
        if tform_obj_type is None:
            tform_obj_type = self.tform_obj_type
        return self.path_preprocess.joinpath('tform_obj', f'{tform_obj_type}', 'tform_obj.pt')

    def read_cam_tform4x4_obj(self, cam_tform4x4_obj_type=None, tform_obj_type =None):
        cam_tform4x4_obj = self.read_cam_tform4x4_obj_raw()

        tform_obj = self.get_tform_obj(tform_obj_type=tform_obj_type)
        if tform_obj is not None:
            cam_tform4x4_obj = tform4x4(cam_tform4x4_obj, inv_tform4x4(tform_obj))

        # note: note alignment of droid slam may include scale, therefore remove this scale.
        # note: projection does not change as we scale the depth z to the object as well
        scale = cam_tform4x4_obj[:3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        cam_tform4x4_obj[:3] = cam_tform4x4_obj[:3] / scale

        if (cam_tform4x4_obj_type is None or cam_tform4x4_obj_type == self.cam_tform4x4_obj_type) and (tform_obj_type == self.tform_obj_type or tform_obj_type is None) :
            self.cam_tform4x4_obj = cam_tform4x4_obj
        return cam_tform4x4_obj

    def read_kpts3d(self):
        kpts3d = self.meta.kpts3d.clone() * PASCAL3D_SCALE_NORMALIZE_TO_REAL[self.category]
        return kpts3d

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

