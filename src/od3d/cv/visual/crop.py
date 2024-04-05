import logging
logger = logging.getLogger(__name__)
import math
import torch
from od3d.cv.visual.resize import resize
from od3d.cv.visual.show import show_img


def crop_white_border_from_img(img, crop_width=True, crop_height=True, resize_to_orig=False, white_pad=0):
    """
    Args:
        img: 3xHxW
    Returns:
        img: 3

    """
    img_mask = ~(img == 1.).all(dim=0)

    H, W = img_mask.shape
    # Find the coordinates of non-zero pixels
    non_zero_coords = torch.nonzero(img_mask)

    if non_zero_coords.numel() == 0:
        # If there are no non-zero pixels, return the input tensor as is
        return img

    # Calculate the minimum and maximum coordinates for each dimension
    min_coords = torch.min(non_zero_coords, dim=0).values
    max_coords = torch.max(non_zero_coords, dim=0).values

    if not crop_width:
        min_coords[1] = 0
        max_coords[1] = W-1

    if not crop_height:
        min_coords[0] = 0
        max_coords[0] = H-1

    # Crop the input tensor using the calculated bounding box
    cropped_img = img[:,
                     min_coords[0]: max_coords[0] + 1,
                     min_coords[1]: max_coords[1] + 1,
                     ]

    if white_pad > 0:
        cropped_img = torch.nn.functional.pad(cropped_img, [white_pad, white_pad, white_pad, white_pad], value=1.)

    if resize_to_orig:
        H_cropped, W_cropped = cropped_img.shape[-2:]
        H_scale = H / H_cropped
        W_scale = W / W_cropped
        scale = min(H_scale, W_scale)
        cropped_img = resize(cropped_img, scale_factor=scale)
        H_cropped, W_cropped = cropped_img.shape[-2:]
        cropped_img_padded = torch.ones_like(img)
        cropped_img_padded[:, :H_cropped, :W_cropped] = cropped_img[:, :H_cropped, :W_cropped]
        cropped_img = cropped_img_padded
    return cropped_img

def crop(img, H_out, W_out, center=None, scale=1., ctx=None, mode="bilinear"):
    device = img.device
    dtype = img.dtype
    img_in_shape = img.shape[1:]

    if center is None:
        center = [img_in_shape[1] // 2, img_in_shape[0] // 2]

    if isinstance(scale, float):
        H_scale = scale
        W_scale = scale
    elif isinstance(scale, torch.Tensor):
        if scale.numel() == 1:
            H_scale = scale.item()
            W_scale = scale.item()
        elif scale.numel() == 2:
            W_scale = scale[0]
            H_scale = scale[1]
        else:
            msg = f'Unexpected number of elements in scale tensor {scale}.'
            raise Exception(msg)
    else:
        msg = f'Unknown scale type {scale}.'
        raise Exception(msg)

    scale_WH = torch.Tensor([W_scale, H_scale])
    scale_avg = (W_scale + H_scale) / 2.

    bbox_in_shape_xhalf = 1. * (W_out / W_scale) / 2.
    bbox_in_shape_yhalf = 1. * (H_out / H_scale) / 2.

    #  x0", "y0", "x1", "y1"
    bbox_in = torch.LongTensor([math.floor(center[0] - bbox_in_shape_xhalf),
                                math.floor(center[1] - bbox_in_shape_yhalf),
                                math.ceil(center[0] + bbox_in_shape_xhalf),
                                math.ceil(center[1] + bbox_in_shape_yhalf)]).to(device)
    # x-, x+, y-, y+
    pad_in = torch.LongTensor([max(-bbox_in[0], 0), max(bbox_in[2] - img_in_shape[1], 0), max(-bbox_in[1], 0),
              max(bbox_in[3] - img_in_shape[0], 0)]).to(device)


    # two options:
    # a) first crop then resize (preferred if scale > 1. -> pad on lower-resolution image)
    if scale_avg >= 1.:
        # img = torch.nn.functional.pad(img, pad=pad)
        img_padded = torch.zeros(
            size=img.shape[:-2] + torch.Size([img.shape[-2] + pad_in[2] + pad_in[3], img.shape[-1] + pad_in[0] + pad_in[1]]),
            dtype=img.dtype, device=device)
        pad_x_upper = -pad_in[1] if pad_in[1] > 0 else None
        pad_y_upper = -pad_in[3] if pad_in[3] > 0 else None
        img_padded[:, pad_in[2]:pad_y_upper, pad_in[0]:pad_x_upper] = img
        img_cropped = img_padded[:, bbox_in[1] + pad_in[2]: bbox_in[3] + pad_in[2], bbox_in[0] + pad_in[0]:bbox_in[2] + pad_in[0]]
        img_out = resize(img_cropped, H_out=H_out, W_out=W_out, mode=mode)
        #logger.info(f'scale >= 1. out size: ({img_out.shape[1]}, {img_out.shape[2]})')

    # b) first resize then crop (preferred if scale < 1. -> pad on lower-resolution image)
    else:
        img_res = resize(img, scale_factor=scale, mode=mode)
        bbox_in_res = ((bbox_in.reshape(2, 2) * scale_WH[None, ]).flatten()).to(torch.long)
        bbox_in_res[3] = H_out + bbox_in_res[1]
        bbox_in_res[2] = W_out + bbox_in_res[0]
        pad_in_res = ((pad_in.reshape(2, 2) * scale_WH[:, None]).flatten()).to(torch.long)
        img_padded = torch.zeros(
            size=img_res.shape[:-2] + torch.Size(
                [max(bbox_in_res[3], img_res.shape[1]) + pad_in_res[2] + 1, max(bbox_in_res[2], img_res.shape[2]) + pad_in_res[0] + 1]),
            dtype=img_res.dtype, device=device)
        img_padded[:, pad_in_res[2]:pad_in_res[2]+img_res.shape[1], pad_in_res[0]:pad_in_res[0]+img_res.shape[2]] = img_res
        img_out = img_padded[:, bbox_in_res[1] + pad_in_res[2]: bbox_in_res[3] + pad_in_res[2],
                      bbox_in_res[0] + pad_in_res[0]:bbox_in_res[2] + pad_in_res[0]]

    if ctx is not None:
        bbox_out = torch.LongTensor([math.ceil(pad_in[0] * W_scale),
                                     math.ceil(pad_in[2] * H_scale),
                                     math.floor(W_out - 1 - pad_in[1] * W_scale),
                                     math.floor(H_out - 1 - pad_in[3] * H_scale)]).to(device)
        ctx = resize(ctx, H_out=H_out, W_out=W_out, mode=mode)
        ctx[:, bbox_out[1]:bbox_out[3], bbox_out[0]:bbox_out[2]] = img_out[:, bbox_out[1]: bbox_out[3], bbox_out[0]:bbox_out[2]]
        img_out = ctx

    cam_crop_tform_cam = torch.Tensor([[W_scale, 0., -bbox_in[0] * W_scale, 0.],
                                       [0., H_scale, -bbox_in[1] * H_scale, 0.],
                                       [0., 0., 1., 0.],
                                       [0., 0., 0., 1.]]).to(device=device, dtype=torch.float)
    return img_out, cam_crop_tform_cam
