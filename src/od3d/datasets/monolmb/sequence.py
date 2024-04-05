import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass
from od3d.datasets.monolmb.frame import MonoLMB_Frame

from od3d.datasets.sequence import OD3D_SequenceMeshMixin, OD3D_MESH_TYPES, OD3D_PCL_TYPES, OD3D_Sequence, \
    OD3D_SequenceCategoryMixin, OD3D_SequenceSfMMixin, OD3D_SEQUENCE_SFM_TYPES
from od3d.datasets.sequence_meta import OD3D_SequenceMeta, OD3D_SequenceMetaCategoryMixin

from od3d.datasets.object import OD3D_MaskTypeMixin, OD3D_CamTform4x4ObjTypeMixin
from od3d.datasets.monolmb.enum import MAP_CATEGORIES_MONOLMB_TO_OD3D

@dataclass
class MonoLMB_SequenceMeta(OD3D_SequenceMetaCategoryMixin, OD3D_SequenceMeta):

    @staticmethod
    def load_from_raw(category: str, name: str):
        return MonoLMB_SequenceMeta(category=category, name=name)

@dataclass
class MonoLMB_Sequence(OD3D_SequenceMeshMixin, OD3D_SequenceCategoryMixin,
                       OD3D_MaskTypeMixin, OD3D_CamTform4x4ObjTypeMixin, OD3D_Sequence):
    frame_type = MonoLMB_Frame
    map_categories_to_od3d = MAP_CATEGORIES_MONOLMB_TO_OD3D
    meta_type = MonoLMB_SequenceMeta


