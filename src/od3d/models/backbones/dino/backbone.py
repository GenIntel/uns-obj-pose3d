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

from od3d.models.backbones.dino.dinov1 import ViTExtractor # for selecting keys, querys, values

class DINOv2_WEIGHTS(str, ExtEnum):
    DEFAULT = 'default'
    NONE = 'none'

class DINOv2(OD3D_Backbone):
    def __init__(
            self,
            config: DictConfig
    ):

        super().__init__(config=config)

        self.transform = SequentialTransform([
            RGB_UInt8ToFloat(),
            RGB_Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.layers_returned = config.layers_returned # choose from [1, 2, 3, 4]
        self.layers_count = len(self.layers_returned)

        # dino_vits8, dino_vitb8, dino_vits16, dino_vitb16, dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
        self.dinov2 = 'dinov2' in self.config.hub_model
        if self.dinov2:
            self.extractor = torch.hub.load(self.config.hub_repo, self.config.hub_model,
                                        pretrained=self.config.weights == 'default')
            self.out_dims = [self.extractor.embed_dim]
        else: # using keys did not show any improvement
            self.extractor = ViTExtractor(model_type=self.config.hub_model)
            self.out_dims = [self.extractor.model.embed_dim]

        self.out_downsample_scales = []
        self.downsample_rate = self.config.downsample_rate
        import re
        match = re.match(r"dino[v2]*_vit[a-z]*([0-9]+)", self.config.hub_model, re.I)
        if match and len(match.groups()) == 1:
            self.downsample_rate_dino = int(match.groups()[0])
        else:
            msg = f'could not retrieve down sample rate dino from model name {self.config.hub_model}'
            raise Exception(msg)

        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

            # if not self.dinov2:
            #     for param in self.extractor.model.parameters():
            #         param.requires_grad = False

    def forward(self, x):
        if x.dim() == 3:
            C, H, W = x.shape
        elif x.dim() == 4:
            B, C, H, W = x.shape
        else:
            raise NotImplementedError

        H_out = (H // self.downsample_rate)
        W_out = (W // self.downsample_rate)
        H_in = H_out * self.downsample_rate_dino
        W_in = W_out * self.downsample_rate_dino

        x = resize(x, H_out= H_in, W_out=W_in)
        if self.dinov2:
            x = self.extractor.forward_features(x)["x_norm_patchtokens"]  # # 'x_norm_patchtokens', 'x_prenorm'
        else:
            #x = self.extractor.get_intermediate_layers(x, n=12)[9]  # maximum 12 layers, zsp uses 9
            #x = x[:, 1:] # remove cls token

            x = self.extractor.extract_descriptors(batch=x, layer=9, facet='key', bin=False, include_cls=False)
            # note: key layer 9 outperforms layer 9

        x = x.reshape(-1, H_out, W_out, self.out_dims[-1]).permute(0, 3, 1, 2)
        x_layers = [x]
        return x_layers
