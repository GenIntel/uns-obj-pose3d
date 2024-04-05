from logging import getLogger
logger = getLogger(__file__)

import torch
from od3d.cv.geometry.transform import reproj2d3d_broadcast, so3_exp_map, rot3x3

from od3d.cv.select import batched_index_select


def fit_plane(pts: torch.Tensor, pts_ids: torch.Tensor):
    """
    Args:
        pts (torch.Tensor): ...xNxF
        pts_ids (torch.Tensor): ...xPxS
    Returns:
        planes (torch.Tensor): ...xPxM
    """

    # ...xPxSxF
    pts_sampled = batched_index_select(index=pts_ids.flatten(-2), input=pts).view(pts_ids.shape + (-1,))

    proposed_shape = pts_sampled.shape[:-2]
    proposed_planes_axis_1 = pts_sampled[..., 1, :] - pts_sampled[..., 0, :]
    proposed_planes_axis_2 = pts_sampled[..., 2, :] - pts_sampled[..., 0, :]
    proposed_planes_axis_z = torch.cross(proposed_planes_axis_1, proposed_planes_axis_2, dim=-1)
    proposed_planes_axis_z = proposed_planes_axis_z / proposed_planes_axis_z.norm(dim=-1, keepdim=True)
    proposed_planes_signed_dist = torch.einsum('pf,pf->p', pts_sampled[..., 0, :].view(-1, 3),
                                               proposed_planes_axis_z.view(-1, 3)).view(proposed_shape + (1,))
    proposed_planes4d = torch.cat([proposed_planes_axis_z, proposed_planes_signed_dist], dim=-1)
    return proposed_planes4d


def score_plane4d_fit(pts: torch.Tensor, plane4d: torch.Tensor, plane_dist_thresh: float,
                      cams_traj: torch.Tensor, pts_on_plane_weight: float = 1.3):
    """
    Args:
        pts (torch.Tensor): ...xNxF
        planes (torch.Tensor): ...xPxM
        cams_traj ( torch.Tensor): ...xCx3
    Returns:
        scores (torch.Tensor): ...xP
    """
    N, F = pts.shape[-2:]
    batch_shape = pts.shape[:-2]
    batch_count = batch_shape.numel()

    proposed_planes_axis_z = plane4d[..., :3]
    proposed_planes_signed_dist = plane4d[..., 3:]
    signed_dists_pts3d_to_proposed_planes = torch.einsum('bnc,bpc->bpn', pts.view(batch_count, -1, F),
                                                         proposed_planes_axis_z.view(batch_count, -1,
                                                                                     F)) - proposed_planes_signed_dist.view(
        batch_count, -1, 1)
    signed_dists_cams_traj_to_proposed_planes = torch.einsum('bnc,bpc->bpn', cams_traj.view(batch_count, -1, F),
                                                             proposed_planes_axis_z.view(batch_count, -1,
                                                                                         F)) - proposed_planes_signed_dist.view(
        batch_count, -1, 1)

    signed_dists_pos_perc = (signed_dists_pts3d_to_proposed_planes > plane_dist_thresh).sum(dim=-1) / N
    signed_dists_thresh_perc = (signed_dists_pts3d_to_proposed_planes.abs() < plane_dist_thresh).sum(dim=-1) / N
    scores = signed_dists_pos_perc + signed_dists_thresh_perc * pts_on_plane_weight
    scores[(signed_dists_cams_traj_to_proposed_planes < 0.).any(dim=-1)] = 0.

    scores = scores.view(batch_shape + (-1,))
    return scores