from pathlib import Path
from PIL import Image
import numpy as np
from typing import List
import torch
import cv2
import os
from tqdm import tqdm

from od3d.cv.visual.blend import rgb_to_range01
def save_gif(imgs: List[torch.Tensor], fpath: Path):
    fpath.parent.mkdir(parents=True, exist_ok=True)

    for img_id in range(len(imgs)):
        imgs[img_id] = Image.fromarray(np.uint8(imgs[img_id].permute(1, 2, 0).detach().cpu().numpy() * 255))
    imgs[0].save(fpath, format="GIF", append_images=imgs,
                 save_all=True, duration=100, loop=0)

def save_video(imgs: List[torch.Tensor], fpath: Path, fps=10):
    height, width = imgs[0].shape[1:]
    fpath.parent.mkdir(parents=True, exist_ok=True)
    vwriter = create_vwriter(fpath=fpath, width=width, height=height, fps=fps)
    for img in tqdm(imgs):
        vwriter.write(tensor_to_cv_img(img))

def create_vwriter(fpath, width, height, fps=10):
    # cv2.VideoWriter_fourcc(*'VP80') -> avilable, browser compatible, .webm, ignore error msg
    # cv2.VideoWriter_fourcc(*'VP90') -> avilable, browser compatible, .webm, ignore error msg
    # cv2.VideoWriter_fourcc(*'X264') -> not available, browser compatible, .mkv
    # cv2.VideoWriter_fourcc(*'MP4V') -> available, not browser compatible, .mp4
    fpath = Path(fpath)
    fpath = fpath.with_suffix(".webm")
    fourcc = cv2.VideoWriter_fourcc(*'VP90')
    vwriter = cv2.VideoWriter(
        str(fpath),
        fourcc,
        fps,
        (width, height),
    )
    return vwriter

def tensor_to_cv_img(x_in):
    # x_in : CxHxW float32
    # x_out : HxWxC uint8
    x_in = rgb_to_range01(x_in)
    x_in = x_in * 1.0
    x_in = torch.clamp(x_in, min=0.0, max=1.0)
    x_out = (x_in.permute(1, 2, 0).cpu().detach().numpy() * 255.0).astype(np.uint8)
    x_out = x_out[:, :, ::-1]
    return x_out