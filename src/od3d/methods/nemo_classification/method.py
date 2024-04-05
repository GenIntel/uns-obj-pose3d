from typing import List
from od3d.methods.method import OD3D_Method
from od3d.datasets.dataset import OD3D_Dataset
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


class NeMo_Classification(OD3D_Method):
    def __init__(
            self,
            config: DictConfig,
            logging_dir,
    ):
        super().__init__(config=config, logging_dir=logging_dir)

    def setup(self):
        pass

    def train(self, dataset: OD3D_Dataset, datasets_test: List[OD3D_Dataset]):
        raise NotImplementedError

    def test(self, dataset: OD3D_Dataset, config_inference: DictConfig = None, pose_iterative_refine=True,
             dataset_sub=None):
        # -> return results dict: key: value

        raise NotImplementedError
