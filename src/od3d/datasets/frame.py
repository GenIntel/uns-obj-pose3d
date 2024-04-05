import logging
logger = logging.getLogger(__name__)

import torch
from od3d.cv.geometry.transform import tform4x4, inv_tform4x4
from od3d.cv.io import read_image, write_mask_image
import torchvision
from dataclasses import dataclass
from typing import List, Union
from enum import Enum
from od3d.cv.geometry.mesh import Mesh
from od3d.datasets.object import OD3D_Object, OD3D_CamTform4x4ObjTypeMixin, OD3D_MaskTypeMixin, OD3D_MeshTypeMixin, \
    OD3D_CAM_TFORM_OBJ_TYPES, OD3D_MESH_TYPES, OD3D_FRAME_MASK_TYPES, OD3D_FrameModalitiesMixin, OD3D_TformObjMixin, \
    OD3D_MeshFeatsTypeMixin, OD3D_FRAME_DEPTH_TYPES, OD3D_DepthTypeMixin

from od3d.datasets.frame_meta import OD3D_FrameMeta
from pathlib import Path
import numpy as np
from od3d.cv.io import write_depth_image, read_depth_image


class OD3D_FRAME_MODALITIES(str, Enum):
    NAME = 'name'
    CAM_INTR4X4 = 'cam_intr4x4'
    CAM_TFORM4X4_OBJ = 'cam_tform4x4_obj'
    CAM_TFORM4X4_OBJS = 'cam_tform4x4_objs'
    CATEGORY = 'category'
    CATEGORY_ID = 'category_id'
    CATEGORIES = 'categories'
    PCL = 'pcl'
    SIZE = 'size'
    RGB = 'rgb'
    RGB_MASK = 'rgb_mask'
    MASK = 'mask'
    MASKS = 'masks'
    DEPTH = 'depth'
    DEPTH_MASK = 'depth_mask'
    MESH = 'mesh'
    MESHS = 'meshs'
    KPTS2D_ANNOT = 'kpts2d_annot'
    KPTS2D_ANNOT_VSBL = 'kpts2d_annot_vsbl'
    KPTS3D = 'kpts3d'
    KPTS_NAMES = 'kpts_names'
    BBOX = 'bbox'
    BBOXS = 'bboxs'
    SEQUENCE = 'sequence'
    SEQUENCE_NAME_UNIQUE = 'sequence_name_unique'
    FRAME = 'frame'
    FRAME_NAME_UNIQUE = 'frame_name_unique'
    RAYS_CENTER3D = 'rays_center3d'

class OD3D_FRAME_MODALITIES_STACKABLE(str, Enum):
    CAM_INTR4X4 = 'cam_intr4x4'
    CAM_TFORM4X4_OBJ = 'cam_tform4x4_obj'
    CAM_TFORM4X4_OBJS = 'cam_tform4x4_objs'
    CATEGORY_ID = 'category_id'
    SIZE = 'size'
    RGB = 'rgb'
    RGB_MASK = 'rgb_mask'
    MASK = 'mask'
    MASKS = 'masks'
    DEPTH = 'depth'
    DEPTH_MASK = 'depth_mask'
    BBOX = 'bbox'
    BBOXS = 'bboxs'
    RAYS_CENTER3D = 'rays_center3d'

class OD3D_FRAME_KPTS2D_ANNOT_TYPES(str, Enum):
    META = 'meta'
    LABEL = 'label'

@dataclass
class OD3D_Frame(OD3D_FrameModalitiesMixin, OD3D_Object):
    meta_type = OD3D_FrameMeta

    def get_modality(self, modality: OD3D_FRAME_MODALITIES):
        if modality in self.modalities:
            if modality == OD3D_FRAME_MODALITIES.CAM_INTR4X4:
                return self.get_cam_intr4x4()
            elif modality == OD3D_FRAME_MODALITIES.CAM_TFORM4X4_OBJ:
                return self.get_cam_tform4x4_obj()
            elif modality == OD3D_FRAME_MODALITIES.CATEGORY:
                return self.category
            elif modality == OD3D_FRAME_MODALITIES.CATEGORY_ID:
                return self.category_id
            elif modality == OD3D_FRAME_MODALITIES.CATEGORIES:
                return self.categories
            elif modality == OD3D_FRAME_MODALITIES.PCL:
                return self.get_pcl()
            elif modality == OD3D_FRAME_MODALITIES.SIZE:
                return self.size
            elif modality == OD3D_FRAME_MODALITIES.RGB:
                return self.get_rgb()
            elif modality == OD3D_FRAME_MODALITIES.RGB_MASK:
                return self.get_rgb_mask()
            elif modality == OD3D_FRAME_MODALITIES.MASK:
                return self.get_mask()
            elif modality == OD3D_FRAME_MODALITIES.DEPTH:
                return self.get_depth()
            elif modality == OD3D_FRAME_MODALITIES.DEPTH_MASK:
                return self.get_depth_mask()
            elif modality == OD3D_FRAME_MODALITIES.MESH:
                return self.get_mesh()
            elif modality == OD3D_FRAME_MODALITIES.BBOX:
                return self.get_bbox()
            elif modality == OD3D_FRAME_MODALITIES.BBOXS:
                return self.get_bboxs()
            elif modality == OD3D_FRAME_MODALITIES.KPTS2D_ANNOT:
                return self.get_kpts2d_annot()
            elif modality == OD3D_FRAME_MODALITIES.KPTS2D_ANNOT_VSBL:
                return self.get_kpts2d_annot_vsbl()
            elif modality == OD3D_FRAME_MODALITIES.KPTS3D:
                return self.get_kpts3d()
            elif modality == OD3D_FRAME_MODALITIES.KPTS_NAMES:
                return self.kpts_names
            elif modality == OD3D_FRAME_MODALITIES.RAYS_CENTER3D:
                return self.rays_center3d
            elif modality == OD3D_FRAME_MODALITIES.SEQUENCE:
                return self.sequence
            elif modality == OD3D_FRAME_MODALITIES.SEQUENCE_NAME_UNIQUE:
                return self.sequence.name_unique
            elif modality == OD3D_FRAME_MODALITIES.FRAME:
                return self
            elif modality == OD3D_FRAME_MODALITIES.FRAME_NAME_UNIQUE:
                return self.name_unique

        logger.warning(f"modality {modality} not supported")
        return None
    #@property
    #def meta(self):
    #    return OD3D_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.name_unique)

    # pass
    # def __init__(self, path_raw: Path, path_preprocess: Path, name_unique: Path, modalities: List[OD3D_FRAME_MODALITIES], categories: List[str]):
    #     self.path_raw: Path = path_raw
    #     self.path_preprocess: Path = path_preprocess
    #     self.all_categories = categories
    #     self.name_unique = name_unique
        # self.modalities = modalities
        #self.meta: OD3D_FrameMetaClasses = meta
        # self.path_meta: Path = path_meta
        # #self.category_id = categories.index(self.category)
        # self.item_id = None
        # self._rgb = None
        # self._depth = None
        # self._depth_mask = None
        # self._kpts2d_orient = None
        # self._mesh = None
        # self._kpts3d = None



@dataclass
class OD3D_FrameMaskMixin(OD3D_MaskTypeMixin):
    mask = None

    @property
    def fpath_mask(self):
        if self.mask_type == OD3D_FRAME_MASK_TYPES.META:
            return self.path_raw.joinpath(self.meta.rfpath_mask)
        elif self.mask_type == OD3D_FRAME_MASK_TYPES.MESH:
            return self.path_preprocess.joinpath("mask", f"{self.mask_type}", self.mesh_type_unique, f"{self.name_unique}.png")
        else:
            return self.path_preprocess.joinpath("mask", f"{self.mask_type}", f"{self.name_unique}.png")

    def write_mask(self, value: torch.Tensor):
        if self.fpath_mask.parent.exists() is False:
            self.fpath_mask.parent.mkdir(parents=True, exist_ok=True)
        if value.dtype == torch.bool:
            value_write = value.to(torch.uint8).detach().cpu() * 255
        elif value.dtype == torch.uint8:
            value_write = value.detach().cpu()
        else:
            value_write = (value * 255).to(torch.uint8).detach().cpu()
        torchvision.io.write_png(input=value_write, filename=str(self.fpath_mask))
        self.mask = value

    def get_mask(self):
        if self.mask is None:
            self.mask = read_image(self.fpath_mask) / 255.
        return self.mask

@dataclass
class OD3D_FrameSizeMixin(OD3D_Object):
    _size = None

    @property
    def size(self):
        if self._size is None:
            self._size = self.meta.size
        return self._size
    @size.setter
    def size(self, value: torch.Tensor):
        self._size = value

    @property
    def H(self):
        return int(self.size[0].item())

    @property
    def W(self):
        return int(self.size[1].item())

@dataclass
class OD3D_FrameRGBMaskMixin(OD3D_FrameSizeMixin):
    rgb_mask = None

    def get_rgb_mask(self):
        if self.rgb_mask is None:
            self.rgb_mask = torch.ones(size=(1, self.H, self.W), dtype=torch.bool)
        return self.rgb_mask


@dataclass
class OD3D_FrameCamTform4x4ObjMixin(OD3D_CamTform4x4ObjTypeMixin):
    cam_tform4x4_obj = None


    def read_cam_tform4x4_obj(self, cam_tform4x4_obj_type = None):
        if cam_tform4x4_obj_type is None:
            cam_tform4x4_obj_type = self.cam_tform4x4_obj_type

        if cam_tform4x4_obj_type == OD3D_CAM_TFORM_OBJ_TYPES.META:
            cam_tform4x4_obj = self.meta.cam_tform4x4_obj
        elif cam_tform4x4_obj_type == OD3D_CAM_TFORM_OBJ_TYPES.SFM:
            cam_tform4x4_obj = self.sequence.get_sfm_cam_tform4x4_obj(f"{Path(self.name_unique).stem}")
        else:
            raise ValueError(f"cam_tform4x4_obj_type {self.cam_tform4x4_obj_type} not supported")

        if cam_tform4x4_obj_type == self.cam_tform4x4_obj_type:
            self.cam_tform4x4_obj = cam_tform4x4_obj
        return cam_tform4x4_obj

    def get_cam_tform4x4_obj(self, cam_tform4x4_obj_type = None):
        if self.cam_tform4x4_obj is not None and (cam_tform4x4_obj_type is None or self.cam_tform4x4_obj_type == cam_tform4x4_obj_type):
            cam_tform4x4_obj = self.cam_tform4x4_obj
        else:
            cam_tform4x4_obj = self.read_cam_tform4x4_obj(cam_tform4x4_obj_type=cam_tform4x4_obj_type)
        return cam_tform4x4_obj

@dataclass
class OD3D_FrameMeshMixin(OD3D_MeshFeatsTypeMixin, OD3D_MeshTypeMixin):
    mesh: Mesh = None

    @property
    def fpath_mesh(self):
        return self.get_fpath_mesh()

    def read_mesh(self, mesh_type=None):
        return self.sequence.read_mesh()

    def get_fpath_mesh(self, mesh_type=None):
        return self.sequence.get_fpath_mesh(mesh_type=mesh_type)

    def get_mesh(self):
        return self.sequence.get_mesh()

@dataclass
class OD3D_FrameCamIntr4x4Mixin(OD3D_Frame):
    cam_intr4x4 = None

    def get_cam_intr4x4(self):
        if self.cam_intr4x4 is None:
            self.cam_intr4x4 = self.read_cam_intr4x4()
        return self.cam_intr4x4

    def read_cam_intr4x4(self):
        if self.cam_intr4x4 is None:
            self.cam_intr4x4 = self.meta.cam_intr4x4.clone()
        return self.cam_intr4x4

@dataclass
class OD3D_CamProj4x4ObjMixin(OD3D_FrameCamTform4x4ObjMixin, OD3D_FrameCamIntr4x4Mixin):
    @property
    def cam_proj4x4_obj(self):
        return tform4x4(self.get_cam_intr4x4(), self.get_cam_tform4x4_obj())

@dataclass
class OD3D_FrameCategoryMixin(OD3D_Object):
    all_categories: List[str]
    map_categories_to_od3d = None

    @property
    def category(self):
        return self.meta.category
    @property
    def category_id(self):
        return self.all_categories.index(self.category)

@dataclass
class OD3D_FrameCategoriesMixin(OD3D_Object):
    all_categories: List[str]

    @property
    def categories(self):
        return self.meta.categories

    @property
    def categories_ids(self):
        return torch.LongTensor([self.all_categories.index(cat) for cat in self.categories])

@dataclass
class OD3D_FrameBBoxMixin(OD3D_Object):
    bbox = None

    def read_bbox(self):
        return self.meta.bbox.clone()

    def get_bbox(self):
        if self.bbox is None:
            self.bbox = self.read_bbox()
        return self.bbox


@dataclass
class OD3D_FrameKpts2d3dMixin(OD3D_Object):
    kpts2d_annot_type: OD3D_FRAME_KPTS2D_ANNOT_TYPES
    kpts3d = None
    kpts2d_annot = None
    kpts2d_annot_vsbl = None

    @property
    def kpts_names(self):
        return self.meta.kpts_names

    @property
    def fpath_kpts2d_annot(self):
        if self.kpts2d_annot_type == OD3D_FRAME_KPTS2D_ANNOT_TYPES.META:
            raise ValueError("Meta kpts2d_annot is not saved in a file")
        else:
            return self.path_preprocess.joinpath("kpts2d_annot", self.kpts2d_annot_type, f"{self.name_unique}.pt")

    def read_kpts2d_annot(self):
        if self.kpts2d_annot_type == OD3D_FRAME_KPTS2D_ANNOT_TYPES.META:
            return self.meta.kpts2d_annot.clone()
        else:
            return torch.load(self.fpath_kpts2d_annot)

    def get_kpts2d_annot(self):
        if self.kpts2d_annot is None:
            self.kpts2d_annot = self.read_kpts2d_annot()
        return self.kpts2d_annot

    @property
    def kpts2d_annot_labeled(self):
        if self.kpts2d_annot_type == OD3D_FRAME_KPTS2D_ANNOT_TYPES.META:
            return True
        else:
            return

    def read_kpts2d_annot_vsbl(self):
        return self.meta.kpts2d_annot_vsbl.clone()

    def get_kpts2d_annot_vsbl(self):
        if self.kpts2d_annot_vsbl is None:
            self.kpts2d_annot_vsbl = self.read_kpts2d_annot_vsbl()
        return self.kpts2d_annot_vsbl

    def read_kpts3d(self):
        return self.meta.kpts3d.clone()

    def get_kpts3d(self):
        if self.kpts3d is None:
            self.kpts3d = self.read_kpts3d()
        return self.kpts3d


    # @property
    # def kpts3d(self):
    #     if self._kpts3d is None:
    #         self._kpts3d = self.meta.kpts3d
    #     return self._kpts3d
    #
    # @kpts3d.setter
    # def kpts3d(self, value: torch.Tensor):
    #         self._kpts3d = value

    # @property
    # def fpath_kpts2d_orient(self):
    #     return self.path_preprocess.joinpath("labels", "kpts2d_orient", f"{self.name_unique}.pt")
    #
    # @property
    # def kpts2d_orient_labeled(self):
    #     return self.fpath_kpts2d_orient.exists()
    # @property
    # def kpts2d_orient(self):
    #     kpts2d_orient = torch.load(self.fpath_kpts2d_orient)
    #     return kpts2d_orient


#class OD3D_Kpts3dMixin(OD3D_Object):

class OD3D_FrameRGBMixin(OD3D_Object):
    rgb = None

    @property
    def fpath_rgb(self):
        return self.path_raw.joinpath(self.meta.rfpath_rgb)

    def read_rgb(self):
        rgb = torchvision.io.read_image(str(self.fpath_rgb), mode=torchvision.io.ImageReadMode.RGB)
        return rgb
    def get_rgb(self):
        if self.rgb is None:
            self.rgb = self.read_rgb()
        return self.rgb

class OD3D_FrameDepthMixin(OD3D_DepthTypeMixin):
    depth = None

    @property
    def fpath_depth(self):
        if self.depth_type == OD3D_FRAME_DEPTH_TYPES.META:
            return self.path_raw.joinpath(self.meta.rfpath_depth)
        elif self.depth_type == OD3D_FRAME_DEPTH_TYPES.MESH:
            return self.path_preprocess.joinpath("depth", f"{self.depth_type}", self.mesh_type_unique, f"{self.name_unique}.png")
        else:
            return self.path_preprocess.joinpath("depth", f"{self.depth_type}", f"{self.name_unique}.png")

    def write_depth(self, value: torch.Tensor):
        write_depth_image(value, path=self.fpath_depth)
        self.depth = value

    def read_depth(self):
        if self.depth_type == OD3D_FRAME_DEPTH_TYPES.META:
            depth = torchvision.io.read_image(str(self.fpath_depth), mode=torchvision.io.ImageReadMode.UNCHANGED)
        elif self.depth_type == OD3D_FRAME_DEPTH_TYPES.MESH:
            depth = read_depth_image(self.fpath_depth)
        else:
            raise NotImplementedError
        return depth

    def get_depth(self):
        if self.depth is None:
            self.depth = self.read_depth()
        return self.depth


class OD3D_FrameDepthMaskMixin(OD3D_DepthTypeMixin):
    depth_mask = None

    @property
    def fpath_depth_mask(self):
        if self.depth_type == OD3D_FRAME_DEPTH_TYPES.META:
            return self.path_raw.joinpath(self.meta.rfpath_depth_mask)
        elif self.depth_type == OD3D_FRAME_DEPTH_TYPES.MESH:
            return self.path_preprocess.joinpath("depth_mask", f"{self.depth_type}", self.mesh_type_unique, f"{self.name_unique}.png")
        else:
            return self.path_preprocess.joinpath("depth_mask", f"{self.depth_type}", f"{self.name_unique}.png")

    def read_depth_mask(self):
        depth_mask = read_image(self.fpath_depth_mask)
        return depth_mask

    def get_depth_mask(self):
        if self.depth_mask is None:
            self.depth_mask = self.read_depth_mask()
        return self.depth_mask

    def write_depth_mask(self, value: torch.Tensor):
        write_mask_image(value, path=self.fpath_depth_mask)
        self.depth_mask = value

@dataclass
class OD3D_FrameSequenceMixin(OD3D_Object):
    sequence_type = None #  OD3D_Sequence

    @property
    def sequence_name(self):
        return self.meta.sequence_name

    @property
    def sequence_name_unique(self):
        return self.meta.sequence_name_unique

    @property
    def sequence(self):
        from dataclasses import fields
        frame_fields = fields(self)
        sequence_fields_names = [field.name for field in fields(self.sequence_type)]
        all_attrs_except_name_unique = {field.name: getattr(self, field.name) for field in frame_fields
                                        if field.name != 'name_unique' and field.name in sequence_fields_names}
        return self.sequence_type(name_unique=self.sequence_name_unique, **all_attrs_except_name_unique)

@dataclass
class OD3D_FrameRaysCenter3dMixin(OD3D_FrameSequenceMixin):
    _rays_center3d = None

    @property
    def rays_center3d(self):
        if self._rays_center3d is None:
            self._rays_center3d = self.sequence.get_sfm_rays_center3d()
        return self._rays_center3d

@dataclass
class OD3D_FrameSubsetMixin(OD3D_Object):
    @property
    def subset(self):
        return self.meta.subset

from od3d.datasets.frame_meta import OD3D_FrameMeta



@dataclass
class OD3D_FrameCamIntr4x4Mixin(OD3D_Frame):
    cam_intr4x4 = None

    def read_cam_intr(self):
        return self.meta.cam_intr4x4.clone()

    def get_cam_intr(self):
        if self.cam_intr4x4 is None:
            self.cam_intr4x4 = self.read_cam_intr()
        return self.cam_intr4x4



@dataclass
class OD3D_FrameTformObjMixin(OD3D_TformObjMixin, OD3D_FrameCamTform4x4ObjMixin, OD3D_Frame):

    def read_cam_tform4x4_obj(self, cam_tform4x4_obj_type=None, tform_obj_type =None):
        cam_tform4x4_obj = super().read_cam_tform4x4_obj(cam_tform4x4_obj_type=cam_tform4x4_obj_type)

        tform_obj = self.sequence.get_tform_obj(tform_obj_type=tform_obj_type)
        if tform_obj is not None:
            cam_tform4x4_obj = tform4x4(cam_tform4x4_obj, inv_tform4x4(tform_obj))

        # note: note alignment of droid slam may include scale, therefore remove this scale.
        # note: projection does not change as we scale the depth z to the object as well
        scale = cam_tform4x4_obj[:3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        cam_tform4x4_obj[:3] = cam_tform4x4_obj[:3] / scale

        if (cam_tform4x4_obj_type is None or cam_tform4x4_obj_type == self.cam_tform4x4_obj_type) and (tform_obj_type == self.tform_obj_type or tform_obj_type is None) :
            self.cam_tform4x4_obj = cam_tform4x4_obj
        return cam_tform4x4_obj
    
    
    def get_cam_tform4x4_obj(self, cam_tform4x4_obj_type = None,  tform_obj_type =None):
        if self.cam_tform4x4_obj is not None and (cam_tform4x4_obj_type is None or self.cam_tform4x4_obj_type == cam_tform4x4_obj_type)  and (tform_obj_type == self.tform_obj_type or tform_obj_type is None):
            cam_tform4x4_obj = self.cam_tform4x4_obj
        else:
            cam_tform4x4_obj = self.read_cam_tform4x4_obj(cam_tform4x4_obj_type=cam_tform4x4_obj_type, tform_obj_type=tform_obj_type)
        return cam_tform4x4_obj


OD3D_FrameClasses = Union[OD3D_Object, OD3D_FrameCategoryMixin, OD3D_FrameCategoriesMixin,
                          OD3D_FrameCamTform4x4ObjMixin, OD3D_CamProj4x4ObjMixin, OD3D_FrameCamIntr4x4Mixin,
                          OD3D_FrameMaskMixin, OD3D_FrameDepthMixin, OD3D_FrameDepthMaskMixin,
                          OD3D_FrameSizeMixin, OD3D_FrameRGBMixin,
                          OD3D_FrameMeshMixin, OD3D_FrameSequenceMixin, OD3D_FrameSubsetMixin,
                          OD3D_FrameBBoxMixin, OD3D_FrameKpts2d3dMixin, OD3D_FrameRGBMaskMixin]