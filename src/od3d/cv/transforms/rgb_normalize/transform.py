from torchvision.transforms.transforms import Normalize
from od3d.cv.transforms.transform import OD3D_Transform

class RGB_Normalize(OD3D_Transform):
    def __init__(self, mean=None, std=None):
        super().__init__()
        self.normalize = Normalize(mean=mean, std=std)

    def __call__(self, frame):
        frame.rgb = self.normalize(frame.get_rgb())
        return frame