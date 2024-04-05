import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass
from od3d.datasets.frame_meta import OD3D_FrameMeta, \
    OD3D_FrameMetaMeshsMixin, OD3D_FrameMetaCategoriesMixin, OD3D_FrameMetaRGBMixin, \
    OD3D_FrameMetaSizeMixin, OD3D_FrameMetaBBoxsMixin, OD3D_FrameMetaSubsetMixin, \
    OD3D_FrameMetaCamTform4x4ObjsMixin, OD3D_FrameMetaCamIntr4x4Mixin
from od3d.datasets.frame import OD3D_Frame, OD3D_FRAME_MODALITIES
from od3d.datasets.pascal3d.frame import Pascal3DFrame
from pathlib import Path
import scipy.io
import numpy as np
import torch
import math
from typing import List
from omegaconf import OmegaConf
from od3d.cv.io import read_image, write_mask_image
from od3d.cv.geometry.mesh import Mesh, Meshes, MESH_RENDER_MODALITIES
from od3d.cv.io import read_depth_image, write_depth_image

from od3d.datasets.objectnet3d.enum import OBJECTNET3D_SCALE_NORMALIZE_TO_REAL

@dataclass
class ObjectNet3D_FrameMeta(OD3D_FrameMetaCamTform4x4ObjsMixin, OD3D_FrameMetaMeshsMixin, OD3D_FrameMetaBBoxsMixin,
                            OD3D_FrameMetaCategoriesMixin, OD3D_FrameMetaRGBMixin, OD3D_FrameMetaSubsetMixin,
                            OD3D_FrameMetaCamIntr4x4Mixin, OD3D_FrameMetaSizeMixin, OD3D_FrameMeta):
    @property
    def name_unique(self):
        return f'{self.subset}/{self.name}'

    #import h5py
    #f = h5py.File('/misc/lmbraid19/sommerl/datasets/ObjectNet3D/Annotations/n03761084_13592.mat')

    @staticmethod
    def load_from_raw(path_raw: Path, rfpath_annotations: Path, rfpath_images: Path, rfpath_meshes: Path, subset: str,  name: str):
        fpath_annotation = path_raw.joinpath(rfpath_annotations, name + '.mat')

        try:
            annotation = scipy.io.loadmat(fpath_annotation)


            rfpath_rgb = rfpath_images.joinpath(name + '.JPEG')

            name = annotation['record']['filename'][0][0][0].split('.')[0]


            W = int(annotation['record'][0][0]['size']['width'][0][0][0][0])
            H = int(annotation['record'][0][0]['size']['height'][0][0][0][0])
            size = torch.Tensor([H, W])

            record = annotation['record'][0][0]
            objects = record['objects'][0]


            #['objects'][0] # [0][0]
            objs_categories = []
            objs_cam_tform4x4_obj = []
            objs_bbox = []
            objs_cam_intr4x4 = []
            objs_rfpaths_meshes = []

            for i in range(objects.shape[0]):

                category = objects[i]['class'].item()
                #truncated = objects[i]['truncated'].item()
                #occluded = objects[i]['occluded'].item()
                #difficult = objects[i]['difficult'].item()
                cad_index = objects[i]['cad_index'].item() - 1
                viewpoint = objects[i]['viewpoint'][0][0]
                try:
                    azimuth = viewpoint['azimuth'].item() * math.pi / 180
                    elevation = viewpoint['elevation'].item() * math.pi / 180
                except:
                    azimuth = viewpoint['azimuth_coarse'].item() * math.pi / 180
                    elevation = viewpoint['elevation_coarse'].item() * math.pi / 180
                theta = viewpoint['theta'].item() * math.pi / 180
                distance = viewpoint['distance'].item()
                px = viewpoint['px'].item()
                py = viewpoint['py'].item()
                viewport = viewpoint['viewport'].item()
                focal = viewpoint['focal'].item()
                #try:
                #    azimuth = str(viewpoint['azimuth'].item())
                #    elevation = str(viewpoint['elevation'].item())
                #except:
                #    azimuth = str(viewpoint['azimuth_coarse'].item())
                #    elevation = str(viewpoint['elevation_coarse'].item())
                # if azimuth == '0' and elevation == '0' and inplane_rotation == '0':
                #    continue
                #mesh_index = objects[i]['cad_index'][0][0] - 1
                # kpts_names = list(object['anchors'][0][0].dtype.names)
                # kpts2d_annot = np.stack([object['anchors'][0][0][n]['location'][0][0][0] if object['anchors'][0][0][n][
                #                                                                                 'status'] == 1 else np.array(
                #     [0, 0]) for n in kpts_names])
                # kpts2d_annot = torch.from_numpy(kpts2d_annot)
                # kpts2d_annot_vsbl = np.array([True if object['anchors'][0][0][n]['status'] == 1 else False for n in kpts_names])
                # kpts2d_annot_vsbl = torch.from_numpy(kpts2d_annot_vsbl)
                if focal == 0:
                    continue

                objs_categories.append(category)
                objs_bbox.append(torch.from_numpy((objects[i]['bbox'][0]).astype('int')))

                cam_tform4x4_obj = Pascal3DFrame.calc_cam_tform_obj(azimuth=azimuth, elevation=elevation, theta=theta,
                                                                    distance=distance)

                objs_cam_tform4x4_obj.append(cam_tform4x4_obj)
                # cam_tform4x4_obj = torch.from_numpy(cam_tform4x4_obj)

                cam_intr3x3 = np.array([[1. * viewport * focal, 0, px],
                                        [0, 1. * viewport * focal, py],
                                        [0, 0, 1.]])
                cam_intr4x4 = np.hstack((cam_intr3x3, [[0], [0], [0]]))
                cam_intr4x4 = np.vstack((cam_intr4x4, [0, 0, 0, 1]))
                cam_intr4x4 = torch.from_numpy(cam_intr4x4).to(dtype=cam_tform4x4_obj.dtype)
                objs_cam_intr4x4.append(cam_intr4x4)

                objs_rfpaths_meshes.append(rfpath_meshes.joinpath(category, f"{(cad_index + 1):02d}.off"))

                #fpath_mesh_kpoints3d = path_meshes.joinpath(f"{category}.mat")
                #annotation_mesh3d = scipy.io.loadmat(fpath_mesh_kpoints3d)
                #kpts3d = np.stack([annotation_mesh3d[category][n][0][mesh_index][0] if len(
                #    annotation_mesh3d[category][n][0][mesh_index]) > 0 else np.array([np.inf, np.inf, np.inf]) for n in
                #                   kpts_names])
                #kpts3d = torch.from_numpy(kpts3d)

            objs_count = len(objs_cam_tform4x4_obj)
            if objs_count == 0:
                incomplete_reason = "focal = 0 for all objects"
                logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
                return None
            else:
                objs_cam_tform4x4_obj = torch.stack(objs_cam_tform4x4_obj, dim=0)
                objs_cam_intr4x4 = torch.stack(objs_cam_intr4x4, dim=0)
                objs_bbox = torch.stack(objs_bbox, dim=0)

                if not ((objs_cam_intr4x4 - objs_cam_intr4x4[:1]) == 0.).all():
                    incomplete_reason = "cam intrinsics is not the same for all objects"
                    logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
                #rfpath_mesh = objs_rfpaths_meshes[0]
                #category = objs_categories[0]
                #cam_tform4x4_obj = objs_cam_tform4x4_obj[0]
                cam_intr4x4 = objs_cam_intr4x4[0]
                bbox = objs_bbox[0]

        except Exception as e:
            logger.info(e)
            incomplete_reason = f"could not read file {fpath_annotation}"
            logger.warning(f"Skip frame {name}, due to {incomplete_reason}.")
            return None

        return ObjectNet3D_FrameMeta(subset=subset, name=name,
                                     rfpath_rgb=rfpath_rgb, rfpaths_meshs=objs_rfpaths_meshes,
                                     l_bboxs=objs_bbox.tolist(), l_size=size.tolist(),
                                     l_cam_tform4x4_objs=objs_cam_tform4x4_obj.tolist(), l_cam_intr4x4=cam_intr4x4.tolist(),
                                     categories=objs_categories)

    @staticmethod
    def get_path_frames_meta_with_subset(path_meta: Path, subset: str):
        return ObjectNet3D_FrameMeta.get_path_metas(path_meta=path_meta).joinpath(subset)


from od3d.datasets.frame import OD3D_FrameMeshMixin, OD3D_FrameTformObjMixin, OD3D_CamProj4x4ObjMixin, \
    OD3D_FrameRGBMaskMixin, OD3D_FrameMaskMixin, OD3D_FrameRGBMixin, OD3D_FrameDepthMixin, OD3D_FrameDepthMaskMixin, \
    OD3D_FrameCategoryMixin, OD3D_FrameSizeMixin, OD3D_Frame, OD3D_FrameBBoxMixin, OD3D_FrameKpts2d3dMixin

from od3d.datasets.object import OD3D_MESH_TYPES
from od3d.cv.geometry.transform import inv_tform4x4, tform4x4
from od3d.datasets.objectnet3d.enum import MAP_CATEGORIES_OBJECTNET3D_TO_OD3D, OBJECTNET3D_SCALE_NORMALIZE_TO_REAL
@dataclass
class ObjectNet3D_Frame(OD3D_FrameBBoxMixin, OD3D_FrameMeshMixin, OD3D_FrameTformObjMixin,
                    OD3D_CamProj4x4ObjMixin, OD3D_FrameRGBMaskMixin, OD3D_FrameMaskMixin, OD3D_FrameRGBMixin,
                    OD3D_FrameDepthMixin, OD3D_FrameDepthMaskMixin, OD3D_FrameCategoryMixin, OD3D_FrameSizeMixin,
                    OD3D_Frame):
    meta_type = ObjectNet3D_FrameMeta
    map_categories_to_od3d = MAP_CATEGORIES_OBJECTNET3D_TO_OD3D

    @staticmethod
    def get_rpath_raw_categorical_meshes(category: str):
        return Path("CAD", "off", f'{category}')

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
            mesh = Mesh.load_from_file(fpath=self.get_fpath_mesh(mesh_type=mesh_type), scale=OBJECTNET3D_SCALE_NORMALIZE_TO_REAL[self.category])
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
        cam_tform4x4_obj[2, 3] *= OBJECTNET3D_SCALE_NORMALIZE_TO_REAL[self.category]
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
