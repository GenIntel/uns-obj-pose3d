import logging
logger = logging.getLogger(__name__)
import torch
from od3d.cv.geometry.transform import rot3x3, transf4x4_from_rot3x3, tform4x4, se3_exp_map, transf3d_broadcast, transf4x4_to_rot4x4_without_scale
from od3d.cv.geometry.primitives import Cuboids

def fit_cuboid_to_pts3d(pts3d, size=None, optimize_transl=True, optimize_rot=False, vertices_max_count=1000, q=0.98,
                        optimize_steps=100, force_symmetric=True, tform_obj_label=None):
    """
    Args:
        pts3d (torch.Tensor): Nx3
        size (float): sets the size of the maximum cuboid length
        optimize_rot (bool): indicates whether transformation is optimized w.r.t. rotation
        optimize_transl (bool): indicates whether transformation is optimized w.r.t. translation
        q (float): quantile indicates the percentile of points which should lie inside each dimension

    Returns:
        cuboid (Cuboids): single mesh with the cuboid
        cuboid_tform4x4_obj (torch.Tensor): transformation from object coordinate system to cuboid
    """

    dtype = pts3d.dtype
    device = pts3d.device

    normalize_scale = (pts3d - pts3d.mean(dim=0)).abs().max()
    pts3d = pts3d.clone() / normalize_scale

    if tform_obj_label is not None:
        cuboid_tform4x4_obj = transf4x4_to_rot4x4_without_scale(tform_obj_label)
    else:
        cuboid_tform4x4_obj = torch.eye(4).to(dtype=dtype, device=device)
    tmp_tform6_cuboid = torch.zeros(6).to(dtype=dtype, device=device)

    # using min max
    # _, cuboid_pts3d_ids_min = pts3d.min(dim=0)
    # _, cuboid_pts3d_ids_max = pts3d.max(dim=0)
    # cuboid_pts3d_limits = pts3d[torch.cat([cuboid_pts3d_ids_min, cuboid_pts3d_ids_max], dim=0)]
    #
    # # using maximum ensures centering.
    # cuboids_vol = (max(abs(cuboid_pts3d_limits[3, 0]), abs(cuboid_pts3d_limits[0, 0])) * 2) * \
    #               (max(abs(cuboid_pts3d_limits[4, 1]), abs(cuboid_pts3d_limits[1, 1])) * 2) * \
    #               (max(abs(cuboid_pts3d_limits[5, 2]), abs(cuboid_pts3d_limits[2, 2])) * 2)

    # # using quantiles
    cuboid_pts3d_limits = torch.cat(
        [pts3d.quantile(q=(1.0 - q) / 2., dim=0), pts3d.quantile(q=1.0 - (1.0 - q) / 2., dim=0)], dim=0)

    if optimize_transl:
        pts3d_mean = (cuboid_pts3d_limits[3:6] + cuboid_pts3d_limits[0:3]) / 2.
        tmp_tform6_cuboid[:3] = -pts3d_mean

    if optimize_steps == 0:
        cuboid_tform4x4_obj = tform4x4(se3_exp_map(tmp_tform6_cuboid.detach()), cuboid_tform4x4_obj.detach())

    for i in range(optimize_steps):
        if not optimize_rot:
            tmp_tform6_cuboid.data[3:] = 0.
        if not optimize_transl:
            tmp_tform6_cuboid.data[:3] = 0.

        cuboid_tform4x4_obj = tform4x4(se3_exp_map(tmp_tform6_cuboid.detach()), cuboid_tform4x4_obj.detach())

        tmp_tform6_cuboid = torch.nn.Parameter(torch.zeros(6).to(device=pts3d.device),
                                               requires_grad=True)
        optimizer = torch.optim.SGD(params=[tmp_tform6_cuboid], lr=0.01, momentum=0.)

        cuboid_tform4x4_obj = tform4x4(se3_exp_map(tmp_tform6_cuboid), cuboid_tform4x4_obj)

        cuboid_pts3d = transf3d_broadcast(pts3d=pts3d, transf4x4=cuboid_tform4x4_obj)

        # logger.info(cuboid_tform4x4_obj[:3, 3])

        # using min max
        # _, cuboid_pts3d_ids_min = cuboid_pts3d.min(dim=0)
        # _, cuboid_pts3d_ids_max = cuboid_pts3d.max(dim=0)
        # cuboid_pts3d_limits = cuboid_pts3d[torch.cat([cuboid_pts3d_ids_min, cuboid_pts3d_ids_max], dim=0)]
        #
        # # using maximum ensures centering.
        # cuboids_vol = (max(abs(cuboid_pts3d_limits[3, 0]), abs(cuboid_pts3d_limits[0, 0])) * 2) * \
        #               (max(abs(cuboid_pts3d_limits[4, 1]), abs(cuboid_pts3d_limits[1, 1])) * 2) * \
        #               (max(abs(cuboid_pts3d_limits[5, 2]), abs(cuboid_pts3d_limits[2, 2])) * 2)

        # # using quantiles
        cuboid_pts3d_limits = torch.cat([cuboid_pts3d.quantile(q=(1.0 - q) / 2., dim=0), cuboid_pts3d.quantile(q=1.0 - (1.0 - q) / 2., dim=0)], dim=0)

        # using maximum ensures centering.
        cuboids_vol = (max(abs(cuboid_pts3d_limits[3]), abs(cuboid_pts3d_limits[0])) * 2) * \
                      (max(abs(cuboid_pts3d_limits[4]), abs(cuboid_pts3d_limits[1])) * 2) * \
                      (max(abs(cuboid_pts3d_limits[5]), abs(cuboid_pts3d_limits[2])) * 2)

        # symmetric_vol = (cuboid_pts3d_limits[3]-cuboid_pts3d_limits[0]) * (cuboid_pts3d_limits[4] - cuboid_pts3d_limits[1]) * (cuboid_pts3d_limits[5] - cuboid_pts3d_limits[2])
        loss = cuboids_vol / cuboids_vol.detach()

        loss.backward()
        logger.info(f'Volume {cuboids_vol}')
        optimizer.step()

    # placing cuboid on ground
    # cuboid_tform4x4_obj[2, 3] -= cuboid_pts3d_limits[2, 2]

    cuboid_tform4x4_obj = cuboid_tform4x4_obj.detach()

    pts3d = pts3d * normalize_scale
    cuboid_tform4x4_obj[:3, 3] *= normalize_scale

    cuboid_pts3d = transf3d_broadcast(pts3d=pts3d, transf4x4=cuboid_tform4x4_obj).detach()

    # using min max
    # _, cuboid_pts3d_ids_min = cuboid_pts3d.min(dim=0)
    # _, cuboid_pts3d_ids_max = cuboid_pts3d.max(dim=0)
    # cuboid_pts3d_limits = cuboid_pts3d[torch.cat([cuboid_pts3d_ids_min, cuboid_pts3d_ids_max], dim=0)]
    #
    # # 1 x 2 x 3
    # cuboid_pts3d_limits = torch.stack(
    #     [cuboid_pts3d_limits[0, 0], cuboid_pts3d_limits[3, 0], cuboid_pts3d_limits[1, 1], cuboid_pts3d_limits[4, 1],
    #      cuboid_pts3d_limits[2, 2], cuboid_pts3d_limits[5, 2]]).reshape(3, 2).T.reshape(1, 2, 3)

    # using quantiles
    # # 1 x 2 x 3
    cuboid_pts3d_limits = torch.cat([cuboid_pts3d.quantile(q=(1.0 - q) / 2., dim=0), cuboid_pts3d.quantile(q=1. - (1.0 - q) / 2., dim=0)], dim=0).reshape(1, 2, 3)

    if force_symmetric:
        logger.info(cuboid_pts3d_limits)
        cuboid_pts3d_limits = cuboid_pts3d_limits.abs().max(dim=1, keepdim=True).values.expand(1, 2, 3).clone()
        cuboid_pts3d_limits[:, 0, :] *= -1

    if size is not None:
        cuboid_size = (cuboid_pts3d_limits[0, 1] - cuboid_pts3d_limits[0, 0]).max()
        scale = size / cuboid_size

        cuboid_pts3d *= scale
        cuboid_pts3d_limits *= scale
        cuboid_tform4x4_obj[:3] *= scale


    cuboid = Cuboids.create_dense_from_limits(limits=cuboid_pts3d_limits, verts_count=vertices_max_count, device=device)

    logger.info(size)
    logger.info(cuboid_pts3d_limits)

    #from od3d.cv.visual.show import show_scene
    #show_scene(meshes=cuboid, pts3d=[cuboid_pts3d], meshes_add_translation=False, pts3d_add_translation=False)

    return cuboid, cuboid_tform4x4_obj


