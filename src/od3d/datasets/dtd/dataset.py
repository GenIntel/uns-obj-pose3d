import logging
logger = logging.getLogger(__name__)
from od3d.datasets.dataset import OD3D_Dataset
import scipy.io as sio
import numpy as np
from omegaconf import DictConfig
import od3d.io
from pathlib import Path
import scipy.io
import torchvision.io
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame
from od3d.datasets.frame_meta import OD3D_FrameMeta, OD3D_FrameMetaRGBMixin, OD3D_FrameMetaSizeMixin, OD3D_FrameMetaSubsetMixin
from typing import List, Dict
from pathlib import Path
import shutil
from tqdm import tqdm
from dataclasses import dataclass


SUBSETS = [
    "train",
    "val"
]

@dataclass
class DTD_FrameMeta(OD3D_FrameMetaSubsetMixin, OD3D_FrameMetaSizeMixin, OD3D_FrameMetaRGBMixin, OD3D_FrameMeta):

    @property
    def name_unique(self):
        return f'{self.subset}/{self.name}'

    @staticmethod
    def get_name_unique_from_subset_name(subset, name):
        return f'{subset}/{name}'

    @staticmethod
    def load_from_raw(**kwargs):
        print('None')
        return None
        pass

class DTD(OD3D_Dataset):
    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List[str]=None, transform=None, index_shift=0, subset_fraction=1.,
                 dict_nested_frames: Dict=None, dict_nested_frames_ban: Dict=None):
        super().__init__(name=name, modalities=modalities, path_raw=path_raw, path_preprocess=path_preprocess,
                 categories=categories, transform=transform, index_shift=index_shift, subset_fraction=subset_fraction,
                 dict_nested_frames=dict_nested_frames, dict_nested_frames_ban=dict_nested_frames_ban)

    #def __len__(self):
    #    return len(self.rfpaths)
    #def __getitem__(self, item):
    #    fpath = self.path.joinpath(self.rfpaths[item])
    #    return torchvision.io.read_image(str(fpath))

    def get_subset_with_dict_nested_frames(self, dict_nested_frames):
        return DTD(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
                   path_preprocess=self.path_preprocess, categories=self.categories,
                   dict_nested_frames=dict_nested_frames, transform=self.transform, index_shift=self.index_shift)

    def get_item(self, item):
        frame_meta = DTD_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta,
                                                                      name_unique=self.list_frames_unique[item])
        return OD3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                          meta=frame_meta, modalities=self.modalities,
                          categories=self.categories)


    @staticmethod
    def setup(config: DictConfig):
        path_raw = Path(config.path_raw)

        if path_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous EXAMPLE")
            shutil.rmtree(path_raw)


        if Path(config.path_raw).exists():
            logger.info(f"Found Describable Textures Dataset at {config.path_raw}")
        else:
            logger.info(f"Download Describable Textures Dataset at {config.path_raw}")
            fpath = Path(config.path_raw).joinpath("dtd.tar.gz")
            od3d.io.download("https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz", fpath=fpath)
            od3d.io.untar(fpath=fpath, dst=fpath.parent)
            od3d.io.move_dir(src=fpath.parent.joinpath('dtd'), dst=fpath.parent)


    @staticmethod
    def extract_meta(config: DictConfig):
        path_raw = Path(config.path_raw)
        dict_nested_frames = config.get("dict_nested_frames")
        if dict_nested_frames is not None:
            subsets = dict_nested_frames.keys()
        else:
            subsets = SUBSETS

        txtrs_meta = scipy.io.loadmat(path_raw.joinpath("imdb", "imdb.mat"))["images"]
        txtrs_ids = txtrs_meta[0, 0][0][0]
        txtrs_fnames = txtrs_meta[0, 0][1][0]
        txtrs_fnames = np.array([f[0] for f in txtrs_fnames])
        txtrs_subsets = txtrs_meta[0, 0][2][0]
        for j, f, s in tqdm(zip(txtrs_ids, txtrs_fnames, txtrs_subsets)):
            rfpath_rgb = Path("images").joinpath(f)
            fpath_rgb = path_raw.joinpath(rfpath_rgb)
            rgb = torchvision.io.read_image(str(fpath_rgb))
            l_size = [rgb.shape[1], rgb.shape[2]]
            if s == 1 and "train" in subsets:
                subset = "train"
            elif s == 2 and "val" in subsets:
                subset = "val"
            else:
                continue
            name = Path(f).stem
            frame_meta = DTD_FrameMeta(subset=subset, l_size=l_size, rfpath_rgb=rfpath_rgb, name=name)
            frame_meta.save(path_meta=DTD.get_path_meta(config))

    @staticmethod
    def preprocess(self, override: False):
        pass
