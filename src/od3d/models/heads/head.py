from torch import nn
from omegaconf import DictConfig
from typing import List

class OD3D_Head(nn.Module):

    subclasses = {}
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls
    def __init__(self, in_dims: List, in_upsample_scales: List, config: DictConfig):
        super().__init__()
        self.config = config
        self.in_dims = in_dims
        self.in_upsample_scales = in_upsample_scales
        self.normalize = config.normalize
        self.downsample_rate = None

    def forward(self, x):
        return x[-1]