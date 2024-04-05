import torch

def substract_pxl2d(x: torch.Tensor):
    """

    Args:
        x: ...xHxWx2
    Returns:
        x_sub_pxl2d: ...xHxWx2

    """

    pxl2d = get_pxl2d_like(x)
    return x - pxl2d

def get_pxl2d_like(x: torch.Tensor):
    """

    Args:
        x: ...xHxWxC
    Returns:
        x_sub_pxl2d: ...xHxWx2

    """
    H, W = x.shape[-3:-1]
    pxl2d = get_pxl2d(H, W, device=x.device, dtype=x.dtype)
    return pxl2d[(None,) * (x.dim() - 3)].expand(*(x.shape[:-1] + torch.Size([2,])))

def get_pxl2d(H, W, dtype, device, B=None):
    grid_x, grid_y = torch.meshgrid(
        [
            torch.arange(0., W, dtype=dtype, device=device),
            torch.arange(0., H, dtype=dtype, device=device),
        ], indexing='xy'
    )
    grid_xy = torch.stack((grid_x, grid_y), dim=-1)

    if B is not None:
        grid_xy = grid_xy[None].expand(B, H, W, 2)
    return grid_xy

def pxl2d_2_pxl2d_normalized(grid_xy, H=None, W=None):
    # ensure normalize pxlcoords is no inplace
    grid_xy = grid_xy.clone()

    if grid_xy.dims() < 4:
        no_batch = True
        grid_xy = grid_xy[None,]
    else:
        no_batch = False

    if H is None or W is None:
        B, C, H, W = grid_xy.shape

    grid_xy[:, 0] = grid_xy[:, 0] / (W - 1.0) * 2.0 - 1.0
    grid_xy[:, 1] = grid_xy[:, 1] / (H - 1.0) * 2.0 - 1.0

    if no_batch:
        grid_xy = grid_xy[0]

    return grid_xy

