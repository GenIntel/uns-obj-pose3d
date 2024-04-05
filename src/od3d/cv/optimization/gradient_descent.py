import torch
from od3d.cv.geometry.transform import se3_exp_map
from od3d.cv.geometry.transform import tform4x4, tform4x4_broadcast
from od3d.cv.visual.show import show_scene

def gradient_descent_se3(pts, models, score_func, pts_dist=None, steps=10, beta0=0.4, beta1=0.6, lr=2e-5, reg_weight=1., pts_weight=0.5, arap_weight=0.05, arap_geo_std=0.02, dims_detached=[], return_pts_offset=False):
    """
    Args:
        pts (torch.Tensor): ...xNxF
        models (torch.Tensor): ...xPx4x4
        score_func: return scores for multiple fitted models, in: ...xNxF, ...xPxM -> ...xP
        pts_affinity: ...xNxN
        pts_dist: ...xNxN
        dims_detached: [0, 1, 2] # 0-5, transl: 0, 1, 2, rot: 3, 4, 5
        #  2e-2, 2e-5
    Returns:
        models: returns fitted models
    """
    device = pts.device
    batch_dims = pts.shape[:-2]
    batch_dims_count = len(batch_dims)

    pts_offset = torch.nn.Parameter(torch.zeros(size=pts.shape).to(device=device), requires_grad=True)
    obj_tform6_tmp = torch.nn.Parameter(torch.zeros(size=models.shape[:-2] + (6, )).to(device=device), requires_grad=True)

    optim_inference = torch.optim.Adam(
        params=[obj_tform6_tmp, pts_offset],
        lr=lr,
        betas=(beta0, beta1),
    )

    models = tform4x4_broadcast(models.detach(), se3_exp_map(obj_tform6_tmp))

    # ...xP
    for s in range(steps):
        # none is required for proposals which are not
        scores = score_func(pts + pts_weight * pts_offset, models[..., None, :, :])[..., 0]
        pairwise_dist = torch.cdist(pts, pts, p=2).detach()
        weights_arap = torch.exp(- (pairwise_dist / pairwise_dist.max()) ** 2 / (arap_geo_std**2) )
        weights_arap.fill_diagonal_(0.) # remove self-connections
        weights_arap = weights_arap / weights_arap.mean()
        weights_arap = weights_arap.nan_to_num(1.)
        pairwise_dist_with_offset = torch.cdist(pts + pts_weight * pts_offset, pts + pts_weight * pts_offset, p=2)
        scores_arap = -(weights_arap * ((pairwise_dist - pairwise_dist_with_offset).abs() / pairwise_dist.max())).flatten(-2).mean()
        obj_tform6_tmp.data[..., dims_detached] = 0.
        models = tform4x4_broadcast(models.detach(), se3_exp_map(obj_tform6_tmp.detach()))
        obj_tform6_tmp.data[..., :] = 0.
        models = tform4x4(models.detach(), se3_exp_map(obj_tform6_tmp))
        loss = (-scores).sum() + arap_weight * (-scores_arap) + reg_weight * (pts_offset * pts_weight).norm(dim=-1).mean()
        loss.backward()
        optim_inference.step()
        optim_inference.zero_grad()

        # show_scene(pts3d = [ pts + pts_weight * pts_offset ], pts3d_colors= [1. - weights_arap[0, :, None].clamp(0, 1).repeat(1, 3)])
    pts_offset = pts_weight * pts_offset.detach()
    models = tform4x4(models.detach(), se3_exp_map(obj_tform6_tmp.detach()))

    if return_pts_offset:
        return models, pts_offset
    else:
        return models
