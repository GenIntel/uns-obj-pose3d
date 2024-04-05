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

class ObjaVerse_FrameMeta(OD3D_FrameMetaMaskMixin, OD3D_FrameMetaRGBMixin, OD3D_FrameMetaSizeMixin,
                             OD3D_FrameMetaCategoryMixin, OD3D_FrameMeta):

    @property
    def name_unique(self):
        return f'{self.category}/{self.name}'

    @staticmethod
    def load_from_raw():
        pass

class ObjaVerse(OD3D_Dataset):
    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List=None,
                 dict_nested_frames: Dict=None,
                 dict_nested_frames_ban: Dict=None,
                 transform=None, index_shift=0, subset_fraction=1.):

        categories = categories if categories is not None else  [] # TODO: OBJECTNET3D_CATEOGORIES.list()
        super().__init__(categories=categories, dict_nested_frames=dict_nested_frames, dict_nested_frames_ban=dict_nested_frames_ban, name=name, modalities=modalities, path_raw=path_raw,
                         path_preprocess=path_preprocess, transform=transform, index_shift=index_shift,
                         subset_fraction=subset_fraction)


        # directories
        # images/texture/textureXX.jpg # 0...14
        # images/texture/texture0.png
        # models/
        #   model_normalized.json
        #   model_normalized.mtl
        #   model_normalized.obj
        #   model_normalized.solid.binvox
        #   model_normalized.surface.binvox
    def get_item(self, item):
        frame_meta = ObjaVerse_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.list_frames_unique[item])
        return OD3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                          meta=frame_meta, modalities=self.modalities, categories=self.categories)

    @staticmethod
    def setup(config: DictConfig):
        # logger.info(OmegaConf.to_yaml(config))
        path_raw = Path(config.path_raw)
        if path_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous ObjaVerse")
            shutil.rmtree(path_raw)

        # git config --global credential.helper store
        # huggingface-cli login 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
        from huggingface_hub import login
        login('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', add_to_git_credential=True)

        "https://huggingface.co/datasets/ObjaVerse/ObjaVerseCore-archive/blob/main/ObjaVerseCore.v2.zip"

        # dataset = load_dataset("ObjaVerse/ObjaVerseCore")
        # from datasets import load_dataset
        # --filter=blob:none
        # dataset = load_dataset('ObjaVerse/ObjaVerseCore', cache_dir="/scratch/sommerl/repos/NeMo/ObjaVerseCore")
        # git clone --filter=blob:none https://huggingface.co/datasets/ObjaVerse/ObjaVerseCore
        # https://huggingface.co/datasets/ObjaVerse/ObjaVerseCore/tree/main
        # from datasets import get_dataset_split_names
        # from datasets import load_dataset_builder
        # from datasets import load_dataset
        # from datasets import get_dataset_config_names
        # "ObjaVerse/ObjaVerseCore"  "rotten_tomatoes"
        # ds_builder = load_dataset_builder("ObjaVerse/ObjaVerseCore")
        # ds_builder.info.description
        # ds_builder.info.features
        # get_dataset_config_names("ObjaVerse/ObjaVerseCore")
        # # load_dataset('LOADING_SCRIPT', cache_dir="PATH/TO/MY/CACHE/DIR")

        path_raw.mkdir(parents=True, exist_ok=True)
        od3d.io.run_cmd('pip install opendatalab', logger=logger, live=True)
        od3d.io.run_cmd('please signup and login first with "odl login"', logger=logger, live=True)
        #  -u {config.credentials.opendatalab.username} -p {config.credentials.opendatalab.password}
        od3d.io.run_cmd('odl info   OpenXD-OmniObject3D-New', logger=logger, live=True)  # View dataset metadata
        od3d.io.run_cmd('odl ls     OpenXD-OmniObject3D-New', logger=logger, live=True)  # View a list of dataset files
        od3d.io.run_cmd(f'cd {path_raw} && odl get    OpenXD-OmniObject3D-New', logger=logger, live=True)