import logging
logger = logging.getLogger(__name__)
from od3d.models.heads.head import OD3D_Head
import torch
from timm.models.vision_transformer import Block
from typing import List, Optional, Type
from omegaconf import DictConfig
from torch import nn
from timm.layers import Mlp, LayerType
from functools import partial

class ViT(OD3D_Head):
    def __init__(
        self,
        in_dims: List,
        in_upsample_scales: List,
        config: DictConfig,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm: bool = False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
    ):
        super().__init__(in_dims=in_dims, in_upsample_scales=in_upsample_scales, config=config,)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)  # dinov2 uses this LayerNorm, prob not important
        act_layer = nn.GELU

        self.blocks_dim = config.blocks.get('dim', 128) #384, # 384 | 384*2
        self.blocks_depth = config.blocks.get('depth', 2) # ,
        self.blocks_num_heads = config.blocks.get('num_heads', 6) # 6, # 6 | 12
        self.downsample_rate = 1.

        self.linear_dim_map = nn.Linear(self.in_dims[-1], self.blocks_dim)
        self.norm_dim_map = norm_layer(self.in_dims[-1])


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.blocks_depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=self.blocks_dim,
                num_heads=self.blocks_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(self.blocks_depth)])

        if config.fully_connected.out_dim is not None:
            self.out_dim = config.fully_connected.out_dim
        else:
            self.out_dim = self.blocks_dim

        if config.fully_connected.out_dim is not None:
            self.fc_enabled = True
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear_in_dim = self.blocks_dim
            self.fc = nn.Linear(self.linear_in_dim, self.out_dim)
        else:
            self.fc_enabled = False

    def forward(self, x):
        if len(x) > 1:
            logger.warning('ViT head only process last feature map.')
        x_res = x[-1] # BxCxHxW
        B, C, H, W = x_res.shape[:]
        x_res = x_res.reshape(B, C, -1).permute(0, 2, 1) # B x N x C

        x_res = self.linear_dim_map(self.norm_dim_map(x_res))

        x_res = self.blocks(x_res)

        x_res = x_res.permute(0, 2, 1).reshape(B, -1, H, W)

        if self.fc_enabled:
            x_res = self.avgpool(x_res)
            x_res = torch.flatten(x_res, 1)
            x_res = self.fc(x_res)

        if self.normalize:
            x_res = torch.nn.functional.normalize(x_res, p=2, dim=1)

        return x_res