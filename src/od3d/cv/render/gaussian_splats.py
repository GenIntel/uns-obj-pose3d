import logging
logger = logging.getLogger(__name__)
import open3d as o3d
import numpy as np
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


# image_height: int
# image_width: int
# tanfovx: float
# tanfovy: float
# bg: torch.Tensor
# scale_modifier: float
# viewmatrix: torch.Tensor
# projmatrix: torch.Tensor
# sh_degree: int
# campos: torch.Tensor
# prefiltered: bool
# debug: bool

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def getProjectionMatrixFromIntrinsics(znear, zfar, fx, fy, W, H, cx, cy):
    #  fx * 2, 0, 0, 0,
    #  0, fy * 2, 0, 0,
    #  0, 0, -(zfar + znear) / (zrange), -(2 * zfar * znear) / (zrange),
    #  0, 0, 1, 0,
    P = torch.zeros(4, 4)
    P[0, 0] = 2 * fx / W
    P[1, 1] = 2 * fy / H
    P[0, 2] = 2 * (cx / W) - 1. # or -1.
    P[1, 2] = 2 * (cy / H) - 1. # or -1.
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[3, 2] = 1.0
    P[2, 3] = -(2 * zfar * znear) / (zfar - znear)
    # zfar and znear are apparently irrelvant https://github.com/graphdeco-inria/gaussian-splatting/issues/399
    return P

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    #  fx * w_ncds, 0, 0, 0,
    #  0, fy * h_ncds, 0, 0,
    #  0, 0, zfar / (zrange), -zfar * znear / (zrange),
    #  0, 0, 1, 0,

    # z' = - (z * (zfar+znear) - 2 * zfar * znear) / zrange =

    # z' = (z * zfar - znear * zfar) / zrange
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left) # fx * screen_width
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left) # 0
    P[1, 2] = (top + bottom) / (top - bottom) # 0
    P[3, 2] = z_sign # 1
    P[2, 2] = z_sign * zfar / (zfar - znear) #
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def render_gaussians(
        cams_tform4x4_obj: torch.Tensor,
        cams_intr4x4: torch.Tensor,
        imgs_size: torch.Tensor,
        pts3d: torch.Tensor,
        pts3d_mask: torch.Tensor,
        feats: torch.Tensor,
        pts3d_size_rel_to_neighbor_dist: float=0.5,
        opacity: float=0.1,
        z_far: float=10000.,
        z_near: float=0.01,
        feats_dim_base=32,
):
    """
    Args:
        cams_tform4x4_obj (torch.Tensor): B x 4 x 4,
        cams_intr4x4 (torch.Tensor): B x 4 x 4,
        imgs_size (torch.Tensor): 2, (height, width)
        pts3d (torch.Tensor): B x N x 3
        pts3d_mask (torch.Tensor): B x N
        feats (torch.Tensor): B x N x F
        opacity (float)
        z_far (float)
        z_near (float)
        feats_dim_base (int)
    Returns:
        img (torch.Tensor): B x F x H x W
    """
    # logger.info(pts3d.shape)
    device = pts3d.device
    B = pts3d.shape[0]
    F = feats.shape[-1]
    N_max = pts3d.shape[1]
    from od3d.cv.geometry.transform import transf3d_broadcast
    pts3d = transf3d_broadcast(pts3d=pts3d, transf4x4=cams_tform4x4_obj[:, None])
    pts3d_mask = pts3d_mask & (pts3d[:, :, 2] >= z_near) & (pts3d[:, :, 2] <= z_far) # otherwise illegal access memory

    feats_splits = int(math.ceil(F / feats_dim_base))

    image_height, image_width = imgs_size
    image_height = int(image_height)
    image_width = int(image_width)

    bg = torch.zeros((feats_dim_base,)).to(device)
    # means2d_buffer = torch.zeros_like(pts3d[0]) + 0  # torch.zeros((N, 2)).to(device) # N x 2
    # opacities_buffer = torch.ones((N_max, 1)).to(device) * opacity  # N x 1
    # rotations_buffer = torch.zeros((N_max, 4)).to(device)  # N x 4 (quaternion) 1 0 0 0
    # rotations_buffer[:, 0] = 1.
    # feats_buffer = torch.zeros(size=(N_max, feats_dim_base)).to(device)

    # rendered_imgs = torch.zeros((B, F, image_height, image_width)).to(device)
    # rendered_imgs = torch.nn.Parameter(torch.zeros((B, F, image_height, image_width)).to(device), requires_grad=True)

    rendered_imgs = []

    for b in range(B):
        rendered_imgs_b = []

        N = int(pts3d_mask[b].sum())

        if N == 0:
            rendered_imgs.append(torch.zeros((F, image_height, image_width)).to(device))
            continue

        pts3d_b = pts3d[b, pts3d_mask[b]].clone()
        pts3d_dists = torch.cdist(pts3d_b.clone().detach(), pts3d_b.clone().detach())
        pts3d_dists.fill_diagonal_(torch.inf)
        pts3d_size_b = pts3d_dists.min(dim=-1).values[:, None].mean(dim=0, keepdim=True).expand(N, 3) * pts3d_size_rel_to_neighbor_dist
        pts3d_size_b = pts3d_size_b.clamp(1e-5, 1e+5) # otherwise illegal access memory

        feats_b = feats[b, pts3d_mask[b]]
        # means2d_b = means2d_buffer[pts3d_mask[b]]
        # opacities_b = opacities_buffer[pts3d_mask[b]]
        # rotations_b = rotations_buffer[pts3d_mask[b]]
        # feats_b_f = feats_buffer[pts3d_mask[b]]

        #logger.info(pts3d_b.isinf().sum())
        #logger.info(pts3d_b.isnan().sum())
        # logger.info(pts3d_b)

        # careful inplace operation
        #pts3d_b_cloned = pts3d_b.clone().detach()
        #pts3d_b_cloned[pts3d_b_cloned[:, 2] < z_far] = pts3d_b[pts3d_b[:, 2] < z_far]
        #pts3d_b = pts3d_b_cloned

        fx = cams_intr4x4[b, 0, 0]
        cx = cams_intr4x4[b, 0, 2]
        fy = cams_intr4x4[b, 1, 1]
        cy = cams_intr4x4[b, 1, 2]

        fovx = focal2fov(fx, image_width)
        fovy = focal2fov(fy, image_height)
        tanfovx = math.tan(fovx * 0.5)
        tanfovy = math.tan(fovy * 0.5)

        # viewmatrix = cams_tform4x4_obj[b].T
        viewmatrix = torch.eye(4).to(device)
        campos = viewmatrix.inverse()[3, :3]

        #projmatrix = getProjectionMatrix(znear=z_near, zfar=z_far, fovX=fovx, fovY=fovy).to(device).T
        projmatrix = getProjectionMatrixFromIntrinsics(znear=z_near, zfar=z_far, fx=fx, fy=fy, W=image_width, H=image_height, cx=cx, cy=cy).to(device).T
        fullprojmatrix = (viewmatrix.unsqueeze(0).bmm(projmatrix.unsqueeze(0))).squeeze(0)
        raster_settings = GaussianRasterizationSettings(image_width=image_width,
                                                        image_height=image_height,
                                                        tanfovx=tanfovx,
                                                        tanfovy=tanfovy,
                                                        bg=bg,
                                                        scale_modifier=1.0,
                                                        viewmatrix=viewmatrix,
                                                        projmatrix=fullprojmatrix,
                                                        sh_degree=0,
                                                        campos=campos,
                                                        prefiltered=False,
                                                        debug=True)


        means2d_b = torch.zeros_like(pts3d_b) + 0  # torch.zeros((N, 2)).to(device) # N x 2
        opacities_b = opacity * torch.ones((N, 1)).to(device)  # N x 1

        rotations_b = torch.zeros((N, 4)).to(device)  # N x 4 (quaternion) 1 0 0 0
        rotations_b[:, 0] = 1.

        for f in range(feats_splits):
            F_f = min((f+1)*feats_dim_base, F) - f*feats_dim_base

            # t = torch.cuda.get_device_properties(0).total_memory
            # r = torch.cuda.memory_reserved(0)
            # a = torch.cuda.memory_allocated(0)
            # free_gpu = r - a  # free inside reserved
            # logger.info(f'{F_f}, {b}')

            feats_b_f = torch.zeros(size=(N, feats_dim_base)).to(device)
            feats_b_f[:, :F_f] = feats_b[:, f*feats_dim_base:f*feats_dim_base + F_f]

            # colors_precomp = None
            # shs = RGB2SH(rgb).to(device)[:, None] # N x (SH-DEG + 1)**2 x 3
            colors_precomp = feats_b_f  # N x F
            shs = None
            scales = pts3d_size_b #  torch.ones((N, 3)).to(device) * pts3d_size  # N x 3

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            color, radii = rasterizer(
                means3D=pts3d_b,
                means2D=means2d_b,
                shs=shs,
                opacities=opacities_b,
                scales=scales,
                rotations=rotations_b,
                cov3D_precomp=None,
                colors_precomp=colors_precomp,
            )

            # from od3d.cv.visual.show import show_img
            # show_img(color[:3])

            # rendered_imgs[b, f*feats_dim_base:f*feats_dim_base+F_f] = color[:F_f]
            # rendered_imgs[b, f*feats_dim_base:f*feats_dim_base+F_f].copy_(color[:F_f])

            rendered_imgs_b.append(color[:F_f])
        rendered_imgs.append(torch.cat(rendered_imgs_b, dim=0))
    rendered_imgs = torch.stack(rendered_imgs, dim=0)

    return rendered_imgs

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))