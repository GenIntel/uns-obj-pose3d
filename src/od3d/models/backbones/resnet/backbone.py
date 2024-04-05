import logging
logger = logging.getLogger(__name__)
from torch import nn
from omegaconf import DictConfig
import torch
from od3d.cv.transforms.sequential import SequentialTransform
from od3d.cv.transforms.rgb_uint8_to_float import RGB_UInt8ToFloat
from od3d.cv.transforms.rgb_normalize import RGB_Normalize
import torchvision
from od3d.models.backbones.backbone import OD3D_Backbone
from od3d.data.ext_enum import ExtEnum
from od3d.cv.visual.resize import resize

class RESNET50_WEIGHTS(str, ExtEnum):
    IMAGENET1K_V2 = 'imagenet1k_v2'  # torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V2
    IMAGENET1K_V1 = 'imagenet1k_v1'  # torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1
    NONE = 'none'

MAP_RESNET50_WEIGHTS = {
    'imagenet1k_v2': torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V2,
    'imagenet1k_v1': torchvision.models.resnet.ResNet50_Weights.IMAGENET1K_V1,
    'none': None
}

class ResNet(OD3D_Backbone):
    def __init__(
        self,
        config: DictConfig
    ):

        super().__init__(config=config)

        self.transform = SequentialTransform([
                RGB_UInt8ToFloat(),
                RGB_Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.layers_returned = config.layers_returned # choose from [1, 2, 3, 4], start with depeest layer (4)
        self.layers_count = len(self.layers_returned)

        resnet = torchvision.models.resnet50(weights=MAP_RESNET50_WEIGHTS[config.weights])

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layers = nn.ModuleList()
        self.layers.append(resnet.layer1) # 256, downsample: 8
        self.layers.append(resnet.layer2) # 512, downsample: 16
        self.layers.append(resnet.layer3) # 1024, downsample: 32
        self.layers.append(resnet.layer4) # 2048, downsample: 64

        self.out_dims = [self.layers[layer_id - 1][-1].conv3.out_channels for layer_id in self.layers_returned]
        self.out_downsample_scales = [2**(self.layers_returned[i]-self.layers_returned[i+1]) for i in range(self.layers_count - 1)]

        self.downsample_rate = self.config.downsample_rate
        self.downsample_rate_resnet = 2 * (2**self.layers_returned[-1])

        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        if x.dim() == 3:
            C, H, W = x.shape
        elif x.dim() == 4:
            B, C, H, W = x.shape
        else:
            raise NotImplementedError

        H_out = (H // self.downsample_rate)
        W_out = (W // self.downsample_rate)
        H_in = H_out * self.downsample_rate_resnet
        W_in = W_out * self.downsample_rate_resnet

        x = resize(x, H_out= H_in, W_out=W_in)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_layers = []
        x_layers.append(self.layers[0](x))
        x_layers.append(self.layers[1](x_layers[-1]))
        x_layers.append(self.layers[2](x_layers[-1]))
        x_layers.append(self.layers[3](x_layers[-1]))

        x_layers = [x_layers[layer_id - 1] for layer_id in self.layers_returned]

        return x_layers


