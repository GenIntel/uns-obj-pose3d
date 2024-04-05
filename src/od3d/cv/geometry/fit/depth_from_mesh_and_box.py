import logging
logger = logging.getLogger(__name__)
import torch
from od3d.cv.geometry.transform import proj3d2d_broadcast, tform4x4
from od3d.cv.geometry.mesh import Meshes
from od3d.cv.geometry.grid import get_pxl2d


def depth_from_mesh_and_box(b_cams_multiview_intr4x4, b_cams_multiview_tform4x4_obj, meshes, labels, mask, downsample_rate, multiview=False):
    """
    Args:
        b_cams_multiview_intr4x4(torch.Tensor): BxCx4x4/Bx4x4
        b_cams_multiview_tform4x4_obj(torch.Tensor): BxCx4x4
        meshes(Meshes): M Meshes
        labels(torch.LongTensor): B labels in range [0, M-1]
        mask(torch.Tensor): BxCxHxW
    Returns:
        depth(torch.Tensor): BxC
    """

    # fx X / (Z s) + cx = x -> (x_proj - cx) / s = (x_box - cx) -> s = (x_proj - cx) / (x_box - cx)
    # fy Y / (Z s) + cy = y -> (y_proj - cy) / s = (y_box - cy) -> s = (y_proj - cy) / (y_box - cy)
    # Z' = Z s

    device = mask.device
    B, C = b_cams_multiview_tform4x4_obj.shape[:2]
    H = mask.shape[-2] * downsample_rate
    W = mask.shape[-1] * downsample_rate
    imgs_sizes = torch.tensor([H, W], dtype=torch.float32, device=device)
    cx = b_cams_multiview_intr4x4[..., 0, 2][:, None]
    cy = b_cams_multiview_intr4x4[..., 1, 2][:, None]

    mesh_verts2d, mesh_verts2d_vsbl = meshes.verts2d(cams_tform4x4_obj=b_cams_multiview_tform4x4_obj,
                                                     cams_intr4x4=b_cams_multiview_intr4x4[:, None,], mesh_ids=labels,
                                                     imgs_sizes=imgs_sizes, down_sample_rate=downsample_rate,
                                                     broadcast_batch_and_cams=True)


    # get largest x and y from vert2d with BxCxVx2 and mask with BxCxV
    mesh_verts2d_masked_mask = mesh_verts2d_vsbl[..., None].repeat(1, 1, 1, 2)
    mesh_verts2d_masked = torch.full_like(mesh_verts2d, float('-inf'))
    mesh_verts2d_masked[mesh_verts2d_masked_mask] = mesh_verts2d[mesh_verts2d_masked_mask]
    mesh_verts2d_x_max = (mesh_verts2d_masked[..., 0].max(dim=-1).values * downsample_rate).clamp(0, W-1)
    mesh_verts2d_y_max = (mesh_verts2d_masked[..., 1].max(dim=-1).values * downsample_rate).clamp(0, H-1)
    mesh_verts2d_masked = torch.full_like(mesh_verts2d, float('+inf'))
    mesh_verts2d_masked[mesh_verts2d_masked_mask] = mesh_verts2d[mesh_verts2d_masked_mask]
    mesh_verts2d_x_min = (mesh_verts2d_masked[..., 0].min(dim=-1).values * downsample_rate).clamp(0, W-1)
    mesh_verts2d_y_min = (mesh_verts2d_masked[..., 1].min(dim=-1).values * downsample_rate).clamp(0, H-1)


    mask_pxl2d = get_pxl2d(H=mask.shape[-2], W=mask.shape[-1], dtype=float, device=device, B=B)[:, None]
    mask_pxl2d_masked_mask = mask[..., None].repeat(1, 1, 1, 1, 2)
    mask_pxl2d_masked = torch.full_like(mask_pxl2d, float('-inf'))
    mask_pxl2d_masked[mask_pxl2d_masked_mask] = mask_pxl2d[mask_pxl2d_masked_mask]
    mask_verts2d_x_max = (mask_pxl2d_masked[..., 0].flatten(2).max(dim=-1).values * downsample_rate).clamp(0, W-1)
    mask_verts2d_y_max = (mask_pxl2d_masked[..., 1].flatten(2).max(dim=-1).values * downsample_rate).clamp(0, H-1)
    mask_pxl2d_masked = torch.full_like(mask_pxl2d, float('+inf'))
    mask_pxl2d_masked[mask_pxl2d_masked_mask] = mask_pxl2d[mask_pxl2d_masked_mask]
    mask_verts2d_x_min = (mask_pxl2d_masked[..., 0].flatten(2).min(dim=-1).values * downsample_rate).clamp(0, W-1)
    mask_verts2d_y_min = (mask_pxl2d_masked[..., 1].flatten(2).min(dim=-1).values * downsample_rate).clamp(0, H-1)

    # B x C x 4
    scales = torch.stack([
        (mesh_verts2d_x_max - cx) / (mask_verts2d_x_max - cx),
        (mesh_verts2d_x_min - cx) / (mask_verts2d_x_min - cx),
        (mesh_verts2d_y_max - cy) / (mask_verts2d_y_max - cy),
        (mesh_verts2d_y_min - cy) / (mask_verts2d_y_min - cy)],
        dim=-1)

    if multiview:
        scale = scales.permute(1, 0, 2).flatten(1).median(dim=-1).values[None,].repeat(B, 1)
    else:
        scale = scales.flatten(2).median(dim=-1).values

    if (scale < 0.01).any() or (scale > 100.).any():
        scale[scale < 0.01] = 1.
        scale[scale > 100.] = 1.
        logger.warning('setting scale of <0.01 or >100. to 1.')

    depth = b_cams_multiview_tform4x4_obj[:, :, 2, 3] * scale

    return depth