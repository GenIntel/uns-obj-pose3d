import logging
logger = logging.getLogger(__name__)
from omegaconf import DictConfig
from od3d.cv.transforms.rgb_uint8_to_float import RGB_UInt8ToFloat
from od3d.cv.transforms.rgb_normalize import RGB_Normalize
from od3d.cv.transforms.sequential import SequentialTransform

import torchvision
from od3d.models.backbones.backbone import OD3D_Backbone
from od3d.models.backbones.resnet_old.backbone_old import NetE2E
from od3d.cv.visual.resize import resize

class ResNetOld(OD3D_Backbone):
    def __init__(
        self,
        config: DictConfig
    ):

        super().__init__(config=config)

        self.transform = SequentialTransform([
                RGB_UInt8ToFloat(),
                RGB_Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


        self.layers_returned = [0] # [1, 2, 3, 4] #  config.layers_returned # choose from [1, 2, 3, 4]
        self.layers_count = len(self.layers_returned)
        self.out_dims = [128]
        self.out_downsample_scales = []

        # self.net = od3d.methods.nemo.backbone_old.ResNetExt(pretrained=True)
        self.net = NetE2E(
            net_type='resnetext',
            local_size=[1, 1],
            output_dimension=128,
            reduce_function=None,
            noise_on_mask=False,
            n_noise_points=5, #config.num_noise,
            pretrain=True,
        )

        self.downsample_rate = self.config.downsample_rate
        self.downsample_rate_resnet = 8

        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, rgb):

        if rgb.dim() == 3:
            C, H, W = rgb.shape
        elif rgb.dim() == 4:
            B, C, H, W = rgb.shape
        else:
            raise NotImplementedError

        H_out = (H // self.downsample_rate)
        W_out = (W // self.downsample_rate)
        H_in = H_out * self.downsample_rate_resnet
        W_in = W_out * self.downsample_rate_resnet

        rgb = resize(rgb, H_out= H_in, W_out=W_in)


        #return self.net.forward_test(resize(rgb, H_out=512, W_out=512))
        #return torch.nn.functional.normalize(self.net.net(rgb), p=2, dim=1)
        return [self.net.forward_test(rgb)]
