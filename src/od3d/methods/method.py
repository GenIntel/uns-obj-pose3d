import logging
logger = logging.getLogger(__name__)
import abc
import subprocess
import sys
from od3d.benchmark.results import OD3D_Results
from od3d.datasets.dataset import OD3D_Dataset
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict
import inspect
import od3d.io
from pathlib import Path

class OD3D_Method(abc.ABC):
    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls
    def __init__(self, config: DictConfig, logging_dir):
        self.config = config
        self.logging_dir = logging_dir
    #@abc.abstractmethod
    #def setup(self):
    #    raise NotImplementedError
    @abc.abstractmethod
    def train(self, datasets_train: Dict[str, OD3D_Dataset], datasets_val: Dict[str, OD3D_Dataset]):
        raise NotImplementedError

    @abc.abstractmethod
    def test(self, datasets_test: List[OD3D_Dataset]) -> OD3D_Results:
        raise NotImplementedError

    @classmethod
    def create_by_name(cls, name: str, logging_dir: Path, config: dict = None):

        keys = inspect.getfullargspec(cls.__init__)[0][1:]

        config_loaded = od3d.io.read_config_intern(rfpath=Path("methods").joinpath(f"{name}.yaml"))
        if config is not None:
            OmegaConf.set_struct(config_loaded, False)
            config_loaded = OmegaConf.merge(config_loaded, config)
            OmegaConf.set_struct(config_loaded, True)
        OmegaConf.resolve(config_loaded)
        od3d_method = cls(config_loaded, logging_dir=logging_dir)
        return od3d_method