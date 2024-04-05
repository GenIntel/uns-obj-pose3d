from od3d.cv.transforms.centerzoom3d.transform import CenterZoom3D
from omegaconf import DictConfig
import torch
from od3d.cv.transforms.transform import OD3D_Transform

class RandomCenterZoom3D(OD3D_Transform):
    def __init__(self, H, W, apply_txtr=False, config:DictConfig = None, scale_min=None, scale_max=None, center_rel_shift_xy_min=[0., 0.], center_rel_shift_xy_max=[0., 0.],
                 scale_with_mask=None, scale_with_dist=None, scale_with_pad=True, center_use_mask=False, scale_selection='shorter'):
        super().__init__()
        self.centerzoom3d = CenterZoom3D(H=H, W=W, scale=None, apply_txtr=apply_txtr, config=config, center_use_mask=center_use_mask, scale_selection=scale_selection)
        self.center_rel_shift_xy_min = torch.Tensor(center_rel_shift_xy_min)
        self.center_rel_shift_xy_max = torch.Tensor(center_rel_shift_xy_max)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_with_mask = scale_with_mask
        self.scale_with_dist = scale_with_dist
        self.scale_with_pad = scale_with_pad


    def __call__(self, frame):
        if self.scale_min is not None and self.scale_max is not None:
            _scale = self.scale_min + torch.rand(1)[0] * (self.scale_max - self.scale_min)
            if self.scale_with_mask is not None:
                self.centerzoom3d.scale_with_mask = _scale
            if self.scale_with_dist is not None:
                self.centerzoom3d.scale_with_dist = _scale
            if self.scale_with_pad is not None:
                self.centerzoom3d.scale_with_pad = _scale
            if self.scale_with_dist is None and self.scale_with_mask is None and self.scale_with_pad is None:
                self.centerzoom3d.scale = _scale

        self.centerzoom3d.center_rel_shift_xy = self.center_rel_shift_xy_min + torch.rand(2) * (self.center_rel_shift_xy_max - self.center_rel_shift_xy_min)
        return self.centerzoom3d(frame)
