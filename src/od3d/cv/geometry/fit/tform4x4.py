import logging
logger = logging.getLogger(__name__)
from od3d.cv.select import batched_index_select
import torch
from od3d.cv.select import batched_indexMD_select
from pytorch3d.ops.points_alignment import corresponding_points_alignment
from od3d.cv.geometry.transform import transf3d_broadcast


def fit_tform4x4_with_matches3d3d(pts: torch.Tensor, pts_ref: torch.Tensor, estimate_scale=True):
    """
    Args:
        pts (torch.Tensor): ...xNxF
        pts_ref (torch.Tensor): ...xNxF
    Returns:
        ref_tform4x4 (torch.Tensor): ...xNx4x4
    """
    N, F = pts.shape[-2:]
    device = pts.device
    S = corresponding_points_alignment(pts.view(-1, N, F), pts_ref.view(-1, N, F), weights=None,
                                       estimate_scale=estimate_scale)

    pts_ref_tform4x4_pts = torch.zeros(size=(pts.shape[:-2].numel(), 4, 4)).to(device=device)
    pts_ref_tform4x4_pts[..., 3, 3] = 1.
    pts_ref_tform4x4_pts[..., :3, :3] = S.R.permute(0, 2, 1) * S.s[..., None, None]
    pts_ref_tform4x4_pts[..., :3, 3] = S.T

    pts_ref_tform4x4_pts = pts_ref_tform4x4_pts.reshape(*pts.shape[:-2], 4, 4)
    return pts_ref_tform4x4_pts

def fit_tform4x4(pts: torch.Tensor, pts_ids: torch.LongTensor, pts_ref: torch.Tensor, dist_ref: torch.Tensor):
    """
    Args:
        pts (torch.Tensor): ...xNxF
        pts_ids (torch.Tensor): ...xPxS
        pts_ref (torch.Tensor): ...xRxF
        dist_ref (torch.Tensor): ...xNxR
    Returns:
        tform4x4 (torch.Tensor): ...xPx4x4
    """
    batch_dims = pts.shape[:-2]
    N, F = pts.shape[-2:]
    R = pts_ref.shape[-2]
    P, S = pts_ids.shape[-2:]
    device=pts.device

    # ...xPxSxF
    pts_sampled = batched_index_select(index=pts_ids.flatten(-2), input=pts).view(pts_ids.shape + (-1,))

    # ...xPxSxR
    dist_sampled = batched_index_select(index=pts_ids.flatten(-2), input=dist_ref).view(pts_ids.shape + (-1,))
    # nearest neighbor correspondences
    pts_ref_ids = dist_sampled.argmin(dim=-1)
    dist_sampled_ref, pts_ref_ids = dist_sampled.min(dim=-1)

    dist_sampled_ref_inf_mask = dist_sampled_ref == torch.inf
    pts_ref_ids[dist_sampled_ref_inf_mask] = (torch.rand(size=(dist_sampled_ref_inf_mask.sum(),)) * R).to(dtype=int, device=device)

    # random correspondences, worse results
    #pts_ref_sample_probs = torch.ones(size=batch_dims + (P, R)).to(device=device)
    #pts_ref_ids = torch.multinomial(pts_ref_sample_probs.view(-1, R), num_samples=S).view(batch_dims + (P, S))

    pts_ref_sampled = batched_index_select(index=pts_ref_ids.flatten(-2), input=pts_ref).view(pts_ref_ids.shape + (-1,))

    pts_ref_tform4x4_pts = fit_tform4x4_with_matches3d3d(pts=pts_sampled, pts_ref=pts_ref_sampled, estimate_scale=True)

    # problem of zeros is ill posed problem if from all 4 sampled points the nearest neighbor is the same.
    mask_pts_ref_tform4x4_pts_zeros = pts_ref_tform4x4_pts[:, :3, :3].flatten(1).sum(dim=-1) == 0.
    pts_ref_tform4x4_pts[mask_pts_ref_tform4x4_pts_zeros, :3, :3] = torch.eye(3)[None,].expand(
        mask_pts_ref_tform4x4_pts_zeros.sum(), 3, 3).to(device=device)

    ## end single batch dimension
    pts_ref_tform4x4_pts = pts_ref_tform4x4_pts.view(pts_sampled.shape[:-2] + (4, 4))

    return pts_ref_tform4x4_pts


def score_tform4x4_fit(pts: torch.Tensor, tform4x4: torch.Tensor, pts_ref: torch.Tensor, dist_app_ref: torch.Tensor,
                       return_dists=False, return_weights=False,
                       geo_cyclic_weight_temp=1., app_cyclic_weight_temp=1., dist_app_weight=0.5, score_perc=1.):
    """
    Args:
        pts (torch.Tensor): ...xNxF
        tform4x4 (torch.Tensor): ...xPx4x4
        pts_ref (torch.Tensor): ...xRxF
        dist_app_ref (torch.Tensor): ...xNxR
    Returns:
        scores (torch.Tensor): ...xP
    """
    N, F = pts.shape[-2:]
    R = dist_app_ref.shape[-1]
    P = tform4x4.shape[-3]
    device = pts.device

    proposal_tform_pts = transf3d_broadcast(pts3d=pts[None,], transf4x4=tform4x4[:, None])

    norm_p = 2

    # PxNxR
    # dist_ref_geometry = (proposal_tform_pts[:, :, None] - pts_ref[None, None,]).norm(dim=-1)
    dist_ref_geometry = torch.cdist(proposal_tform_pts, pts_ref[None,], p=norm_p)  #
    dist_ref_geo_max = torch.cdist(pts_ref[None,], pts_ref[None,], p=norm_p).max().detach()  #
    dist_src_geo_max = torch.cdist(pts[None,], pts[None,], p=norm_p).max().detach()  #
    dist_ref_geometry = (dist_ref_geometry.clone() / (dist_ref_geo_max))

    argmin_ref_from_src = dist_app_ref.argmin(dim=-1)  # N,
    argmin_src_from_ref = dist_app_ref.argmin(dim=-2)  # R,
    src_cyclic_dist = (pts - pts[argmin_src_from_ref[argmin_ref_from_src]]).norm(dim=-1, p=norm_p).detach() / dist_src_geo_max #  N,
    ref_cyclic_dist = (pts_ref - pts_ref[argmin_ref_from_src[argmin_src_from_ref]]).norm(dim=-1, p=norm_p).detach() / dist_ref_geo_max # R,

    src_cyclic_dist[batched_index_select(input=dist_app_ref, index=argmin_ref_from_src[..., None], dim=1).isinf()[:, 0]] = torch.inf
    ref_cyclic_dist[batched_index_select(input=dist_app_ref.T, index=argmin_src_from_ref[..., None], dim=1).isinf()[:, 0]] = torch.inf

    # not used anymore due to uni-directional cycle dist
    #cyclic_dist_avg = (src_cyclic_dist[:, None] + ref_cyclic_dist[None,]) / 2. # NxR,
    #cyclic_dist_avg = cyclic_dist_avg[None,].expand(*dist_ref_geometry.shape).clone().detach() # PxNxR

    src_cyclic_dist_mask = ~src_cyclic_dist.isinf()
    ref_cyclic_dist_mask = ~ref_cyclic_dist.isinf()

    if app_cyclic_weight_temp is None or geo_cyclic_weight_temp is None:
        cyclic_weight_temp = (src_cyclic_dist[src_cyclic_dist_mask].sum(dim=-1) + ref_cyclic_dist[ref_cyclic_dist_mask].sum(dim=-1)) / (src_cyclic_dist_mask.sum(dim=-1) + ref_cyclic_dist_mask.sum(dim=-1) + 1e-6)
        if app_cyclic_weight_temp is None:
            app_cyclic_weight_temp = cyclic_weight_temp
        if geo_cyclic_weight_temp is None:
            geo_cyclic_weight_temp = cyclic_weight_temp

    # PxNxR
    argmin_ref_from_src = argmin_ref_from_src[None,].expand(P, N)
    argmin_src_from_ref = argmin_src_from_ref[None,].expand(P, R)

    # PxN
    proposal_tform_pts_nn_geo_ref_id = dist_ref_geometry.argmin(dim=-1)
    proposal_tform_pts_geo_id = torch.arange(proposal_tform_pts_nn_geo_ref_id.shape[-1]).view(1, -1).\
        expand(proposal_tform_pts_nn_geo_ref_id.shape).to(device=device)

    # PxR
    proposal_tform_pts_ref_nn_geo_pts_id = dist_ref_geometry.argmin(dim=-2)
    proposal_tform_pts_ref_geo_id = torch.arange(proposal_tform_pts_ref_nn_geo_pts_id.shape[-1]).view(1, -1).\
        expand(proposal_tform_pts_ref_nn_geo_pts_id.shape).to(device=device)

    # PxN
    proposal_tform_pts_nn_app_ref_id = argmin_ref_from_src
    proposal_tform_pts_app_id = torch.arange(proposal_tform_pts_nn_app_ref_id.shape[-1]).view(1, -1).\
        expand(proposal_tform_pts_nn_app_ref_id.shape).to(device=device)

    # PxR
    proposal_tform_pts_ref_nn_app_pts_id = argmin_src_from_ref
    proposal_tform_pts_ref_app_id = torch.arange(proposal_tform_pts_ref_nn_app_pts_id.shape[-1]).view(1, -1).\
        expand(proposal_tform_pts_ref_nn_app_pts_id.shape).to(device=device)

    # Px2N
    proposal_tform_pts_nn_ref_id = torch.cat([proposal_tform_pts_nn_geo_ref_id, proposal_tform_pts_nn_app_ref_id], dim=-1)
    proposal_tform_pts_id = torch.cat([proposal_tform_pts_geo_id, proposal_tform_pts_app_id], dim=-1)
    # Px2R
    proposal_tform_pts_ref_nn_pts_id = torch.cat([proposal_tform_pts_ref_nn_geo_pts_id, proposal_tform_pts_ref_nn_app_pts_id], dim=-1)
    proposal_tform_pts_ref_id = torch.cat([proposal_tform_pts_ref_geo_id, proposal_tform_pts_ref_app_id], dim=-1)

    # forward: Px2Nx2
    proposal_tform_pts_nn_ref_id_2D = torch.stack([proposal_tform_pts_id, proposal_tform_pts_nn_ref_id], dim=-1)
    proposal_tform_pts_nn_ref_id_2D = proposal_tform_pts_nn_ref_id_2D.clone()
    assert proposal_tform_pts_nn_ref_id_2D.shape == torch.Size([P, 2*N, 2])

    # backward: Px2Rx2
    proposal_tform_pts_ref_nn_pts_id_2D = torch.stack([proposal_tform_pts_ref_id, proposal_tform_pts_ref_nn_pts_id], dim=-1)
    proposal_tform_pts_ref_nn_pts_id_2D = proposal_tform_pts_ref_nn_pts_id_2D.flip(dims=[-1])
    proposal_tform_pts_ref_nn_pts_id_2D = proposal_tform_pts_ref_nn_pts_id_2D.clone()

    # forward+backward nn
    proposal_pts_nn_id_2D = torch.cat([proposal_tform_pts_nn_ref_id_2D, proposal_tform_pts_ref_nn_pts_id_2D], dim=1)
    proposal_dist_ref = batched_indexMD_select(indexMD=proposal_pts_nn_id_2D, inputMD=dist_ref_geometry)

    # not used anymore due to uni-directional cycle dist
    #proposal_cyclic_dist_avg_ref = batched_indexMD_select(indexMD=proposal_pts_nn_id_2D, inputMD=cyclic_dist_avg)

    # forward + backward nn
    proposal_cyclic_dist_avg_ref = torch.cat([batched_index_select(index=proposal_tform_pts_nn_ref_id_2D[:, :, 0:1], input=src_cyclic_dist[None, None].repeat(*(proposal_tform_pts_nn_ref_id_2D.shape[:2] + (1,))), dim=2),
                                              batched_index_select(index=proposal_tform_pts_ref_nn_pts_id_2D[:, :, 1:2], input=ref_cyclic_dist[None, None].repeat(*(proposal_tform_pts_ref_nn_pts_id_2D.shape[:2] + (1,))), dim=2)], dim=1)[:, :, 0]

    proposal_dist_ref_geometry = torch.cat([proposal_dist_ref[:, :N], proposal_dist_ref[:, 2*N:2*N+R]], dim=-1)
    proposal_dist_ref_appear = torch.cat([proposal_dist_ref[:, N:2*N], proposal_dist_ref[:, 2*N+R:]], dim=-1)

    proposal_cyclic_dist_ref_geometry = torch.cat([proposal_cyclic_dist_avg_ref[:, :N], proposal_cyclic_dist_avg_ref[:, 2*N:2*N+R]], dim=-1)
    proposal_cyclic_dist_ref_appear = torch.cat([proposal_cyclic_dist_avg_ref[:, N:2*N], proposal_cyclic_dist_avg_ref[:, 2*N+R:]], dim=-1)

    proposal_dist_ref_geometry_weight = torch.exp(-((proposal_cyclic_dist_ref_geometry / geo_cyclic_weight_temp)))
    proposal_dist_ref_geometry_weight = proposal_dist_ref_geometry_weight / proposal_dist_ref_geometry_weight.mean(dim=-1, keepdim=True)
    proposal_dist_ref_geometry_weight = proposal_dist_ref_geometry_weight.nan_to_num(1.)
    proposal_dist_ref_appear_weight = torch.exp(- ((proposal_cyclic_dist_ref_appear / app_cyclic_weight_temp)))
    proposal_dist_ref_appear_weight = proposal_dist_ref_appear_weight / proposal_dist_ref_appear_weight.mean(dim=-1, keepdim=True)
    proposal_dist_ref_appear_weight = proposal_dist_ref_appear_weight.nan_to_num(1.)

    proposal_dist_ref_geometry *= proposal_dist_ref_geometry_weight
    proposal_dist_ref_appear *= proposal_dist_ref_appear_weight

    proposal_scores_pointwise = -((1. - dist_app_weight) * proposal_dist_ref_geometry + dist_app_weight * proposal_dist_ref_appear)

    proposal_scores = proposal_scores_pointwise.mean(dim=-1)

    proposal_dist_ref_geo_avg = proposal_dist_ref_geometry.mean(dim=-1)
    proposal_dist_ref_appear_avg = proposal_dist_ref_appear.mean(dim=-1)

    returns = (proposal_scores, )
    if return_dists:
        returns += (proposal_dist_ref_geo_avg, proposal_dist_ref_appear_avg, )
    if return_weights:
        returns += (proposal_dist_ref_geometry_weight, proposal_dist_ref_appear_weight, )
    if len(returns) == 1:
        returns = returns[0]
    return returns