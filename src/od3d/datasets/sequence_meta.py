import logging
logger = logging.getLogger(__name__)

from od3d.datasets.frame_meta import OD3D_Meta
from dataclasses import dataclass
from pathlib import Path
from typing import Union

@dataclass
class OD3D_SequenceMeta(OD3D_Meta):

    @classmethod
    def get_rfpath_metas(cls):
        return Path("sequences")

@dataclass
class OD3D_SequenceMetaCategoryMixin(OD3D_SequenceMeta):
    category: str

    @property
    def name_unique(self):
        return f'{self.category}/{super().name_unique}'

    @staticmethod
    def load_from_meta_with_category_and_name(path_meta: Path, category: str, name: str):
        name_unique = f'{category}/{name}'
        return OD3D_SequenceMetaCategoryMixin.load_from_meta_with_name_unique(path_meta=path_meta, name_unique=name_unique)

    @staticmethod
    def get_name_unique_with_category_and_name(category: str, name: str):
        return f'{category}/{name}'

    @staticmethod
    def get_fpath_sequence_meta_with_category_and_name(path_meta: Path, category: str, name: str):
        return path_meta.joinpath(OD3D_SequenceMetaCategoryMixin.get_rfpath_metas(), OD3D_SequenceMetaCategoryMixin.get_name_unique_with_category_and_name(category=category, name=name))

    @staticmethod
    def load_from_raw(category: str, name: str):
        return OD3D_SequenceMetaCategoryMixin(category=category, name=name)


OD3D_SequenceMetaClasses = Union[OD3D_SequenceMeta, OD3D_SequenceMetaCategoryMixin]