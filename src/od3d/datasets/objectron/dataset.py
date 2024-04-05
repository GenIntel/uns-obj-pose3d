import logging

import od3d.io

logger = logging.getLogger(__name__)
from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.frame_meta import OD3D_FrameMeta, OD3D_FrameMetaRGBMixin, \
    OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaMaskMixin, OD3D_FrameMetaSizeMixin
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame
#from od3d.datasets.objectnet3d.enum import OBJECTNET3D_CATEOGORIES
from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
import shutil

class Objectron_FrameMeta(OD3D_FrameMetaMaskMixin, OD3D_FrameMetaRGBMixin, OD3D_FrameMetaSizeMixin,
                          OD3D_FrameMetaCategoryMixin, OD3D_FrameMeta):

    @property
    def name_unique(self):
        return f'{self.category}/{self.name}'

    @staticmethod
    def load_from_raw():
        pass

class Objectron(OD3D_Dataset):
    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List=None,
                 dict_nested_frames: Dict=None,
                 dict_nested_frames_ban: Dict=None,
                 transform=None, index_shift=0, subset_fraction=1.):

        categories = categories if categories is not None else  [] # TODO: OBJECTNET3D_CATEOGORIES.list()
        super().__init__(categories=categories, dict_nested_frames=dict_nested_frames, dict_nested_frames_ban=dict_nested_frames_ban, name=name, modalities=modalities, path_raw=path_raw,
                         path_preprocess=path_preprocess, transform=transform, index_shift=index_shift,
                         subset_fraction=subset_fraction)

    def get_item(self, item):
        frame_meta = Objectron_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.list_frames_unique[item])
        return OD3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                          meta=frame_meta, modalities=self.modalities, categories=self.categories)


    @staticmethod
    def extract_meta(config: DictConfig):
        path_meta = Objectron.get_path_meta(config=config)
        path_raw = Path(config.path_raw)

        if path_meta.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous Objectron")
            shutil.rmtree(path_meta)

        path_meta.mkdir(parents=True, exist_ok=True)

        pass


    @staticmethod
    def setup(config: DictConfig):
        pass