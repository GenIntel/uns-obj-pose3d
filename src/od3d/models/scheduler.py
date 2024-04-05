import logging
logger = logging.getLogger(__name__)
from omegaconf import DictConfig
import torch.nn as nn

class OD3D_Scheduler(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
