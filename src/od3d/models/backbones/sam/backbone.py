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
from segment_anything import SamPredictor, sam_model_registry
from pathlib import Path
from od3d.io import download


class SAM(OD3D_Backbone):
    def __init__(
        self,
        config: DictConfig
    ):

        super().__init__(config=config)

        self.transform = SequentialTransform([
            # RGB_UInt8ToFloat(),
            # RGB_Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        device = "cuda"

        if not Path(config.sam_checkpoint).exists():
            logger.info('SAM checkpoint not found, downloading from the official source')
            download(url=f'https://dl.fbaipublicfiles.com/segment_anything/{Path(config.sam_checkpoint).name}',
                     fpath=Path(config.sam_checkpoint))

        sam = sam_model_registry[config.model_type](checkpoint=config.sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.out_downsample_scales = []
        self.downsample_rate = 1
        self.out_dims = [1]

    def forward(self, x, points_xy):
        batch_size = x.shape[0]
        input_box_width_half = x.shape[3] / 2.5 # total 2/3 image width
        input_box_height_half = x.shape[2] / 2.5 # total 2/3 image height
        input_box_size_half = min(input_box_width_half, input_box_height_half)
        input_box_width_half = input_box_size_half
        input_box_height_half = input_box_size_half
        masks, scores, logits = [], [], []
        for b in range(batch_size):
            x_b = x[b]
            self.predictor.set_image(x_b.permute(1, 2, 0).contiguous().detach().cpu().numpy())

            import numpy as np
            cx = points_xy[b:b+1].detach().cpu().numpy()[0, 0]
            cy = points_xy[b:b+1].detach().cpu().numpy()[0, 1]
            input_point = np.array([[cx, cy]]) # np.array [[cx, cy]]
            input_point = None

            x1 = max(cx - input_box_width_half, 0)
            x2 = min(cx + input_box_width_half, x.shape[3] - 1)
            y1 = max(cy - input_box_height_half, 0)
            y2 = min(cy + input_box_height_half, x.shape[2] - 1)
            input_box = np.array([x1, y1, x2, y2]) # np.array [x1, y1, x2, y2]
            input_label = np.array([1]) # 1: foreground, 0: background

            masks_b, scores_b, logits_b = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=True,
            )
            masks.append(torch.from_numpy(masks_b))
            scores.append(torch.from_numpy(scores_b))
            logits.append(torch.from_numpy(logits_b))

        masks = torch.stack(masks, dim=0)
        scores = torch.stack(scores, dim=0)
        logits = torch.stack(logits, dim=0)
        #from od3d.cv.visual.resize import resize
        #masks = resize(masks, H_out=x.shape[2] , W_out=x.shape[3], mode='nearest')
        return masks, scores, logits
