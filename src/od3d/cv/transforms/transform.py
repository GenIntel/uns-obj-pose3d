import inspect
from pathlib import Path
import od3d.io
from omegaconf import DictConfig
import abc
from enum import Enum
from od3d.io import write_dict_as_yaml
from typing import List, Dict
import torch
import numpy as np
class OD3D_Transform():

    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def get_as_dict(self):
        _dict = {}
        keys = inspect.getfullargspec(self.__init__)[0][1:]
        for key in keys:
            if hasattr(self, key):
                _dict[key] = getattr(self, key)
                if isinstance(_dict[key], torch.Tensor):
                    _dict[key] = _dict[key].detach().cpu().tolist()
                if isinstance(_dict[key], np.ndarray):
                    _dict[key] = _dict[key].tolist()
                if isinstance(_dict[key], List) and len(_dict[key]) > 0 and isinstance(_dict[key][0], OD3D_Transform):
                    _dict[key] = [transform.get_as_dict() for transform in _dict[key]]
                if isinstance(_dict[key], Enum):
                    _dict[key] = str(_dict[key])
        _dict['class_name'] = type(self).__name__
        return _dict

    def save_to_config(self, fpath: Path):
        _dict = self.get_as_dict()
        write_dict_as_yaml(fpath=fpath, _dict=_dict)

    def __init__(self, **kwargs):
        pass

    # @abc.abstractmethod
    def __call__(self, frame):
        pass

    @classmethod
    def create_by_name(cls, name: str):
        config = od3d.io.read_config_intern(rfpath=Path("methods/transform").joinpath(f"{name}.yaml"))
        return cls.create_from_config(config)

    @classmethod
    def create_from_config(cls, config: DictConfig):

        keys = inspect.getfullargspec(cls.subclasses[config.class_name].__init__)[0][1:]
        return cls.subclasses[config.class_name](**dict((key, config.get(key)) for key in keys if config.get(key, None) is not None))


