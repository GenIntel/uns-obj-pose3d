import logging
logger = logging.getLogger(__name__)
from typing import List
from omegaconf import DictConfig
from od3d.cv.transforms.transform import OD3D_Transform


class SequentialTransform(OD3D_Transform):

    def __init__(self, transforms: List[OD3D_Transform]):
        super().__init__()
        self.transforms: List[OD3D_Transform] = transforms

    def __call__(self, frame):
        for transform in self.transforms:
            if transform is not None:
                frame = transform(frame)
            else:
                logger.warning(f'Transform is None. {self.transforms}')
        return frame

    @classmethod
    def create_from_config(cls, config: DictConfig):
        transforms: List[OD3D_Transform] = []
        for config_transform in config.transforms:
            transforms.append(
                OD3D_Transform.subclasses[config_transform.class_name].create_from_config(config=config_transform))
        return SequentialTransform(transforms)


