import torch
from typing import List

def get_sub_dims(dims: List[int], dims_banned: List[int]):
    sub_dims = [dim for dim in torch.arange(len(dims)).tolist() if dim not in dims_banned]
    return sub_dims

def get_sub_shape(shape: torch.Size, dims_banned: List[int]):
    sub_shape = [shape[dim] for dim in torch.arange(len(shape)).tolist() if dim not in dims_banned]
    return torch.Size(sub_shape)

def mask_from_pxl2d(pxl2d: torch.Tensor, dim_pxl: int, dim_pts: int, H: int, W: int):
    """
    Args:
        pxl2d (torch.Tensor): ...x2x... 2d pixel information
        dim_pxl (int): Dimension of the 2D pixel information
        dim_pts (int): Dimension of the points belonging to one mask
        H (int): Output height.
        W (int): Output width.

    Returns:
        mask (torch.Tensor): ..xHxW mask which contains ones at 2d pixels.
    """

    device = pxl2d.device
    shape_mask = get_sub_shape(pxl2d.shape, dims_banned=[dim_pxl, dim_pts]) + torch.Size([H, W])
    mask = torch.zeros(size=shape_mask, dtype=torch.bool, device=device)

    pxl2d_x = torch.index_select(pxl2d, index=torch.LongTensor([0]).to(device=device), dim=dim_pxl).to(dtype=torch.long)[..., 0]
    pxl2d_y = torch.index_select(pxl2d, index=torch.LongTensor([1]).to(device=device), dim=dim_pxl).to(dtype=torch.long)[..., 0]

    pxl2d_has_zero = (pxl2d == 0).all(dim=dim_pxl, keepdim=True).all(dim=dim_pts, keepdim=True)
    pxl2d_has_zero = pxl2d_has_zero.squeeze(dim=(dim_pts, dim_pxl))

    pxl2d_mask_out_of_bounds = (pxl2d_x < 0) + (pxl2d_x > W-1) + (pxl2d_y < 0) + (pxl2d_y > H-1)
    pxl1d = pxl2d_y * W + pxl2d_x
    pxl1d[pxl2d_mask_out_of_bounds] = 0
    mask = mask.reshape(*mask.shape[:-2], -1)
    mask = torch.scatter(dim=2, input=mask, index=pxl1d.permute(1, 2, 0), src=torch.ones(size=pxl1d.permute(1, 2, 0).shape, dtype=torch.bool).to(device=device))

    mask = mask.reshape(*mask.shape[:-1], H, W)

    mask[:, :, 0, 0] = pxl2d_has_zero

    return mask