import logging
logger = logging.getLogger(__name__)

from pathlib import Path
import torch
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass
from typing import List, Union, Dict
from abc import ABC, abstractmethod
from od3d.data.ext_dicts import unroll_nested_dict, rollup_flattened_dict
import re

@dataclass
class OD3D_Meta(ABC):
    name: str

    # @staticmethod
    # @abstractmethod
    @classmethod
    def load_from_meta_with_rfpath(cls, path_meta: Path, rfpath: Path):
        return cls(**cls.load_omega_conf_with_rfpath(path_meta=path_meta, rfpath=rfpath))
        # raise NotImplementedError
        # each subclass must implement this function with
        #   SUBCLASS(**SUBCLASS.load_omega_conf_with_rfpath(path_meta=path_meta, rfpath=rfpath))

    # @staticmethod
    @classmethod
    def load_from_meta_with_name_unique(cls, path_meta: Path, name_unique: str):
        rfpath = cls.get_rfpath_from_name_unique(name_unique=name_unique)
        return cls.load_from_meta_with_rfpath(path_meta=path_meta, rfpath=rfpath)

    @staticmethod
    def load_from_raw(**kwargs):
        pass

    @property
    def name_unique(self):
        return self.name

    @classmethod
    def get_rfpath_from_name_unique(cls, name_unique: str):
        return cls.get_rfpath_metas().joinpath(Path(f'{name_unique}.yaml'))

    @classmethod
    def get_rfpath_metas(cls):
        raise NotImplementedError

    @classmethod
    def get_path_metas(cls, path_meta: Path):
        return path_meta.joinpath(cls.get_rfpath_metas())

    @staticmethod
    def atoi(text):
        return int(text) if text.isdigit() else text

    @classmethod
    def complete_nested_metas(cls, path_meta: Path, dict_nested_metas: Union[Dict, DictConfig, None], parent_key='', separator='/', dict_nested_metas_ban: Union[Dict, DictConfig, None]=None):
        if dict_nested_metas is None:
            dict_nested_metas = {'': None}
            if dict_nested_metas_ban is not None:
                dict_nested_metas_ban = {'': dict_nested_metas_ban}

        dict_nested_frames_completed = {}

        for key, value in dict_nested_metas.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            if value is None:
                dir_fpaths = list(
                    cls.get_path_metas(path_meta=path_meta).joinpath(new_key).iterdir())
                if dir_fpaths[0].is_dir():
                    if dict_nested_metas_ban is None or key not in dict_nested_metas_ban:
                        dict_nested_frames_completed[key] = \
                            cls.complete_nested_metas(path_meta=path_meta, parent_key=new_key,
                                                      dict_nested_metas={f'{dir_fpath.stem}': None
                                                                          for dir_fpath in dir_fpaths})
                    elif dict_nested_metas_ban[key] is not None:
                        dict_nested_frames_completed[key] = \
                            cls.complete_nested_metas(path_meta=path_meta, parent_key=new_key,
                                                      dict_nested_metas={f'{dir_fpath.stem}': None
                                                                         for dir_fpath in dir_fpaths},
                                                      dict_nested_metas_ban=dict_nested_metas_ban[key])
                    else:
                        pass
                else:
                    if dict_nested_metas_ban is None or key not in dict_nested_metas_ban:
                        dict_nested_frames_completed[key] = [dir_fpath.stem for dir_fpath in sorted(dir_fpaths, key=lambda f: [OD3D_Meta.atoi(val) for val in re.split(r'(\d+)', f.stem)])]
                    elif dict_nested_metas_ban[key] is not None:
                        dict_nested_frames_completed[key] = [dir_fpath.stem for dir_fpath in sorted(dir_fpaths, key=lambda f: [OD3D_Meta.atoi(val) for val in re.split(r'(\d+)', f.stem)]) if dir_fpath.stem not in dict_nested_metas_ban[key]]
                    else:
                        pass
            elif isinstance(value,  Dict) or isinstance(value, DictConfig):
                if dict_nested_metas_ban is None or key not in dict_nested_metas_ban:
                    dict_nested_frames_completed[key] = cls.complete_nested_metas(path_meta=path_meta,
                                                                                  parent_key=new_key,
                                                                                  dict_nested_metas=value)
                elif dict_nested_metas_ban[key] is not None:
                    dict_nested_frames_completed[key] = cls.complete_nested_metas(path_meta=path_meta,
                                                                                  parent_key=new_key,
                                                                                  dict_nested_metas=value,
                                                                                  dict_nested_metas_ban=dict_nested_metas_ban[key])
                else:
                    pass
            else:
                if dict_nested_metas_ban is None or key not in dict_nested_metas_ban:
                    dict_nested_frames_completed[key] = value
                elif dict_nested_metas_ban[key] is not None:
                    dict_nested_frames_completed[key] = [val for val in value if val not in dict_nested_metas_ban[key]]
                else:
                    pass

        if len(dict_nested_frames_completed.keys()) == 1 and '' in dict_nested_frames_completed.keys():
            dict_nested_frames_completed = dict_nested_frames_completed['']
        return dict_nested_frames_completed

    @property
    def rfpath(self):
        return self.get_rfpath_from_name_unique(self.name_unique)

    @staticmethod
    def unroll_nested_metas(dict_nested_meta: Dict, separator='/'):
        dict_frames = unroll_nested_dict(dict_nested_meta, separator=separator)
        list_frames_names_unique = []
        for key, frames_names in dict_frames.items():
            for frame_name in frames_names:
                frame_name_unique = f"{key}{separator}{frame_name}" if key else frame_name
                list_frames_names_unique.append(frame_name_unique)
        return list_frames_names_unique

    @staticmethod
    def rollup_flattened_frames(list_meta_names_unique: List):
        dict_frames = {}
        for frame_name_unique in list_meta_names_unique:
            frame_name_unique_split = frame_name_unique.split('/')
            key = '/'.join(frame_name_unique_split[:-1])
            if key not in dict_frames.keys():
                dict_frames[key] = []
            frame_name = frame_name_unique_split[-1]
            dict_frames[key].append(frame_name)
        return rollup_flattened_dict(flattened_dict=dict_frames)


    def get_fpath(self, path_meta: Path):
        return path_meta.joinpath(self.rfpath)

    @staticmethod
    def load_omega_conf_with_rfpath(path_meta: Path, rfpath: Path):
        fpath_meta = path_meta.joinpath(rfpath)
        if not fpath_meta.exists():
            logger.error(f'Missing meta fpath {fpath_meta}. Preprocess meta before.')
        return OmegaConf.load(fpath_meta)

    def save(self, path_meta):
        frame_meta_fpath = self.get_fpath(path_meta=path_meta)
        frame_meta_config = OmegaConf.structured(self)
        if not frame_meta_fpath.parent.exists():
            frame_meta_fpath.parent.mkdir(parents=True)
        OmegaConf.save(frame_meta_config, frame_meta_fpath, resolve=True)