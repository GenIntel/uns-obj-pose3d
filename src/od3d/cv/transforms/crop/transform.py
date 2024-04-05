import logging
logger = logging.getLogger(__name__)
from od3d.datasets.frame import OD3D_Frame, OD3D_FRAME_MODALITIES
from od3d.cv.visual.crop import crop
from od3d.cv.transforms.transform import OD3D_Transform

class Crop(OD3D_Transform):

    def __init__(self, H, W):
        super().__init__()
        self.H = H
        self.W = W

    def __call__(self, frame: OD3D_Frame):
        # logger.info(f"Frame name {self.name}")

        #_ = frame.size

        scale = min(self.H / frame.H, self.W / frame.W)
        frame.size[0:1] = self.H
        frame.size[1:2] = self.W

        frame.rgb, cam_crop_tform_cam = crop(img=frame.get_rgb(), H_out=self.H, W_out=self.W, scale=scale)

        return frame