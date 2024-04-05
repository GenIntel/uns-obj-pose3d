from torch import nn
from omegaconf import DictConfig


class OD3D_Backbone(nn.Module):

    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.freeze = config.get("freeze", False)
        self.transform = None
        self.downsample_rate = None

    pass