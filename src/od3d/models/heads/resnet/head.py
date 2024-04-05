import logging
logger = logging.getLogger(__name__)
from od3d.models.heads.head import OD3D_Head
from omegaconf import DictConfig
import torch
import torchvision.models.resnet
from torchvision.models.resnet import Bottleneck, BasicBlock
import torch.nn as nn
from typing import List
from od3d.data.ext_enum import ExtEnum

class RESNET_CONV_BLOCK_TYPES(str, ExtEnum):
    BOTTLENECK = 'bottleneck'
    BASIC = 'basic'


def get_block(block_type: RESNET_CONV_BLOCK_TYPES, in_dim: int, out_dim: int, stride: int, pre_upsampling: float = 1.):
    moudule_list = nn.ModuleList()
    if pre_upsampling != 1.:
        moudule_list.append(nn.Upsample(scale_factor=pre_upsampling))

    if block_type == RESNET_CONV_BLOCK_TYPES.BASIC:
        moudule_list.append(BasicBlock(inplanes=in_dim, planes=out_dim, stride=stride,
                                        downsample=nn.Sequential(
                                            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(out_dim))))

    elif block_type == RESNET_CONV_BLOCK_TYPES.BOTTLENECK:
        moudule_list.append(Bottleneck(inplanes=in_dim, planes=out_dim // 4, stride=stride,
                          downsample=nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False),
                                                   nn.BatchNorm2d(out_dim))))
    else:
        logger.error(f'Unknown block type {block_type}.')
        raise NotImplementedError

    return nn.Sequential(*moudule_list)

class ResNet(OD3D_Head):
    def __init__(
        self,
        in_dims: List,
        in_upsample_scales: List,
        config: DictConfig,
    ):
        super().__init__(in_dims=in_dims, in_upsample_scales=in_upsample_scales, config=config)

        self.upsample_conv_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.block_type: RESNET_CONV_BLOCK_TYPES = config.block_type
        self.pad_zero = config.get('pad_zero', False)
        self.pad_width = 1

        self.in_upsampled_dim = config.get("in_upsampled_dim", self.in_dims[-1])
        assert len(self.in_upsample_scales) == len(self.in_dims) - 1

        for i in range(len(self.in_dims) - 1):
            #downsample_channels = nn.Sequential(
            #    nn.Conv2d(self.in_dims[i] + self.in_dims[i + 1], self.in_dims[i + 1], kernel_size=1, stride=1, bias=False),
            #    nn.BatchNorm2d(self.in_dims[i + 1]))
            #self.upsample_conv_blocks.append(Bottleneck(inplanes=self.in_dims[i] + self.in_dims[i + 1], planes=self.in_dims[i + 1] // 4, downsample=downsample_channels))
            if self.in_upsampled_dim != self.in_dims[-1]:
                if i == 0:
                    self.upsample_conv_blocks.append(get_block(block_type=self.block_type,
                                                               in_dim=self.in_dims[i] + self.in_dims[i + 1],
                                                               out_dim=self.in_upsampled_dim,
                                                               stride=1))
                else:
                    self.upsample_conv_blocks.append(get_block(block_type=self.block_type,
                                                               in_dim=self.in_upsampled_dim + self.in_dims[i + 1],
                                                               out_dim=self.in_upsampled_dim,
                                                               stride=1))
            else:
                self.upsample_conv_blocks.append(get_block(block_type=self.block_type,
                                                           in_dim=self.in_dims[i] + self.in_dims[i + 1],
                                                           out_dim=self.in_dims[i + 1],
                                                           stride=1))

            self.upsample.append(nn.Upsample(scale_factor=self.in_upsample_scales[i]))

        self.conv_blocks = nn.ModuleList()
        self.conv_blocks_out_dims = config.conv_blocks.out_dims
        self.conv_blocks_count = len(self.conv_blocks_out_dims)
        self.conv_blocks_strides = config.conv_blocks.strides
        if self.conv_blocks_count > 0:
            self.conv_blocks_in_dims = [self.in_upsampled_dim] + [config.conv_blocks.out_dims[i] for i in range(self.conv_blocks_count - 1)]
        else:
            self.conv_blocks_in_dims = []
        self.conv_blocks_pre_upsampling = config.conv_blocks.pre_upsampling
        assert len(self.conv_blocks_in_dims) == len(self.conv_blocks_out_dims)
        assert len(self.conv_blocks_out_dims) == len(self.conv_blocks_strides)
        assert len(self.conv_blocks_in_dims) == 0 or self.conv_blocks_in_dims[0] == self.in_upsampled_dim
        assert len(self.conv_blocks_out_dims) == len(self.conv_blocks_pre_upsampling)

        self.conv_block_scaling = [ self.conv_blocks_strides[i] / self.conv_blocks_pre_upsampling[i] for i in range(len(self.conv_blocks_strides))]
        #self.conv_blocks = nn.Sequential(*[Bottleneck(inplanes=self.conv_blocks_in_dims[i],
        #                                              planes=self.conv_blocks_out_dims[i] // 4,
        #                                              stride=self.conv_blocks_strides[i],
        #                                              downsample=nn.Sequential(
        #                                                    nn.Conv2d(self.conv_blocks_in_dims[i], self.conv_blocks_out_dims[i], kernel_size=1, stride=1, bias=False),
        #                                                    nn.BatchNorm2d(self.conv_blocks_out_dims[i])))
        #                                   for i in range(self.conv_blocks_count)])

        self.conv_blocks = nn.Sequential(*[get_block(block_type=self.block_type,
                                                     in_dim=self.conv_blocks_in_dims[i],
                                                     out_dim=self.conv_blocks_out_dims[i],
                                                     stride=self.conv_blocks_strides[i],
                                                     pre_upsampling=self.conv_blocks_pre_upsampling[i])
                                           for i in range(self.conv_blocks_count)])


        if config.fully_connected.out_dim is not None:
            self.out_dim = config.fully_connected.out_dim
        elif len(config.conv_blocks.out_dims) > 0:
            self.out_dim = config.conv_blocks.out_dims[-1]
        else:
            self.out_dim = self.in_dims[-1]

        if config.fully_connected.out_dim is not None:
            self.fc_enabled = True
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear_in_dim = self.conv_blocks_out_dims[-1]
            self.fc = nn.Linear(self.linear_in_dim, self.out_dim)
            self.downsample_rate = 1
        else:
            from operator import mul
            from functools import reduce
            self.downsample_rate = reduce(mul, [1] + self.conv_block_scaling, 1)
            self.fc_enabled = False

    def forward(self, x):
        if len(self.in_dims) == 1:
            x_res = x[0]
        else:
            x_res = None
            for i in range(len(self.in_dims) - 1):
                if i == 0:
                    x_low = x[i]
                else:
                    x_low = x_res
                x_res = self.upsample_conv_blocks[i](torch.cat([x[i+1], self.upsample[i](x_low)], dim=1))

        if self.pad_zero:

            x_res = torch.nn.functional.pad(x_res, (self.pad_width, self.pad_width, self.pad_width, self.pad_width), mode='constant', value=0)

        x_res = self.conv_blocks(x_res)

        if self.pad_zero:
            pad_width = int((1 / self.downsample_rate) * self.pad_width)
            x_res = x_res[:, :, pad_width:-pad_width, pad_width:-pad_width]

        if self.fc_enabled:
            x_res = self.avgpool(x_res)
            x_res = torch.flatten(x_res, 1)
            x_res = self.fc(x_res)

        if self.normalize:
            x_res = torch.nn.functional.normalize(x_res, p=2, dim=1)

        return x_res