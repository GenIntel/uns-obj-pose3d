import torch

def rgb_to_range01(rgb):
    if (rgb < 0).any() or (rgb > 1).any():
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    return rgb
def blend_rgb(rgb1, rgb2, alpha1=0.5, alpha2=0.5):
    if rgb1.dtype == torch.bool or rgb1.dtype == torch.float:
        rgb1 = rgb_to_range01(rgb1)
        rgb1 = rgb1 * 255.
    if rgb2.dtype == torch.bool or rgb2.dtype == torch.float:
        rgb2 = rgb_to_range01(rgb2)
        rgb2 = rgb2 * 255.
    return (alpha1 * rgb1 + alpha2 * rgb2).clamp(0, 255).to(torch.uint8)