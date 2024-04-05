import logging
logger = logging.getLogger(__name__)

import torch
from pytorch3d.renderer.cameras import look_at_view_transform, look_at_rotation
import math
from pytorch3d.transforms import axis_angle_to_matrix, rotation_6d_to_matrix, matrix_to_rotation_6d
import pytorch3d.transforms

def transf4x4_to_rot4x4_without_scale(transf4x4):
    # note: note alignment of droid slam may include scale, therefore remove this scale.
    # note: projection does not change as we scale the depth z to the object as well
    rot4x4 = transf4x4.clone()
    scale = rot4x4[:3, :3].norm(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
    rot4x4[:3] = rot4x4[:3] / scale
    rot4x4[:3, 3] = 0.
    return rot4x4

def so3_exp_map(so3_log:torch.Tensor):

    so3_log_shape = so3_log.shape
    so3_3x3 = pytorch3d.transforms.so3_exp_map(so3_log.reshape(-1, 3))
    so3_3x3 = so3_3x3.reshape(so3_log_shape[:-1] + torch.Size([3, 3]))

    return so3_3x3

def so3_exp_map_tform4x4(so3_log:torch.Tensor):
    so3_3x3 = so3_exp_map(so3_log)
    return transf4x4_from_rot3x3(so3_3x3)

def rot6d_to_rot3x3(rot6d: torch.Tensor):
    rot6d_shape = rot6d.shape
    _rot3x3 = rotation_6d_to_matrix(rot6d.reshape(-1, 6))
    return _rot3x3.reshape(rot6d_shape[:-1] + torch.Size([3, 3]))

def so3_log_map(so3_3x3: torch.Tensor):
    so3_exp_shape = so3_3x3.shape

    so3_log = pytorch3d.transforms.so3_log_map(so3_3x3.reshape(-1, 3, 3))

    so3_log = so3_log.reshape(so3_exp_shape[:-2] + torch.Size([3]))

    return so3_log

def rot3x3_to_rot6d(rot3x3: torch.Tensor):
    rot3x3_shape = rot3x3.shape
    _rot6d = matrix_to_rotation_6d(rot3x3.reshape(-1, 3, 3))
    return _rot6d.reshape(rot3x3_shape[:-2] + torch.Size([6]))

def se3_log_map(se3_exp: torch.Tensor):
    """
    Args:
        se3_exp (torch.Tensor): ...x4x4
    Returns:
        se3_log (torch.Tensor): ...x4x4
    """

    se3_exp_shape = se3_exp.shape

    se3_log = pytorch3d.transforms.se3_log_map(se3_exp.reshape(-1, 4, 4).permute(0, 2, 1))

    se3_log = se3_log.reshape(se3_exp_shape[:-2] + torch.Size([6]))

    return se3_log

def se3_exp_map(se3_log: torch.Tensor):
    """
        Args:
            se3_log (torch.Tensor): ...x6 (transl :3, rot 3:6)
        Returns:
            se3_4x4 (torch.Tensor): ...x4x4
    """
    se3_log_shape = se3_log.shape

    se3_4x4 = pytorch3d.transforms.se3_exp_map(se3_log.reshape(-1, 6)).permute(0, 2, 1)

    se3_4x4 = se3_4x4.reshape(se3_log_shape[:-1] + torch.Size([4, 4]))

    return se3_4x4


def rot3x3_from_normal(normal: torch.Tensor):
    """Rotation matrix for rotating normal to vector [0.,-1.,0.]
        Args:
            normal (torch.Tensor): ...x3

        Returns:
            rot3x3 (torch.Tensor): ...x3x3
    """
    return rot3x3_from_two_vectors(a=normal, b=torch.Tensor([0., -1., 0.]).to(device=normal.device, dtype=normal.dtype).expand(normal.shape[:-1] + torch.Size([3])))


def transf4x4_from_normal(normal: torch.Tensor):
    """Transformation matrix for rotating vector [0.,-1.,0.] onto normal
        Args:
            normal (torch.Tensor): ...x3

        Returns:
            transf4x4 (torch.Tensor): ...x4x4
    """

    return transf4x4_from_rot3x3(rot3x3_from_normal(normal))

def rot3x3_from_two_vectors(a: torch.Tensor, b: torch.Tensor):
    """Rotation matrix for rotating vector a onto vector b
        Args:
            a (torch.Tensor): ...x3
            b (torch.Tensor): ...x3

        Returns:
            rot3x3 (torch.Tensor): ...x3x3

    """



    v = torch.cross(a / a.norm(dim=-1, keepdim=True), b / b.norm(dim=-1, keepdim=True), dim=-1)

    if a.dim() == 1:
        v = v[None, ]
    rot3x3 = so3_exp_map(v)  # .transpose(-1, -2)

    if a.dim() == 1:
        rot3x3 = rot3x3[0]
    return rot3x3


def inv_tform4x4(a_tform4x4_b):
    scale = torch.linalg.norm(a_tform4x4_b[..., :3, :3], dim=-1, keepdim=True)
    scale_avg = scale.mean(dim=-2, keepdim=True)
    if ((scale - scale_avg).abs() > 1e-5).any():
        scale_sel = scale[((scale - scale_avg).abs() > 1e-5)]
        logger.warning(f'Scale is not constant over all dimensions {scale_sel}')

    b_rot3x3_a = a_tform4x4_b[..., :3, :3].transpose(-1, -2) / (scale_avg ** 2 + 1e-5)
    a_rot3x3_b_origin = a_tform4x4_b[..., :3, 3]
    b_transl3_0 = -rot3d(pts3d=a_rot3x3_b_origin, rot3x3=b_rot3x3_a)
    return transf4x4_from_rot3x3_and_transl3(transl3=b_transl3_0, rot3x3=b_rot3x3_a)

def tform4x4(tform1_4x4, tform2_4x4):
    return torch.bmm(tform1_4x4.reshape(-1, 4, 4), tform2_4x4.reshape(-1, 4, 4)).reshape(tform1_4x4.shape)

def tform4x4_broadcast(a_tform4x4_b, b_tform4x4_c):
    shape_first_dims = torch.broadcast_shapes(a_tform4x4_b.shape[:-2], b_tform4x4_c.shape[:-2])
    a_tform4x4_c = tform4x4(a_tform4x4_b.expand(*shape_first_dims, 4, 4), b_tform4x4_c.expand(*shape_first_dims, 4, 4))
    return a_tform4x4_c
def rot3x3(rot1_3x3, rot2_3x3):
    return torch.bmm(rot1_3x3.reshape(-1, 3, 3), rot2_3x3.reshape(-1, 3, 3)).reshape(rot1_3x3.shape)

def rot3x3_broadcast(a_rot3x3_b, b_rot3x3_c):
    shape_first_dims = torch.broadcast_shapes(a_rot3x3_b.shape[:-2], b_rot3x3_c.shape[:-2])
    a_rot3x3_c = rot3x3(a_rot3x3_b.expand(*shape_first_dims, 3, 3), b_rot3x3_c.expand(*shape_first_dims, 3, 3))
    return a_rot3x3_c


def transf4x4_from_pos_and_theta(pos, theta):
    in_shape = theta.shape
    dtype = theta.dtype
    device = theta.device
    zeros = torch.zeros(size=in_shape, dtype=dtype, device=device).reshape(-1)
    ones = torch.ones(size=in_shape, dtype=dtype, device=device).reshape(-1)
    camrot_pos_rot3x3_obj = look_at_rotation(pos, up=((0, 0, 1),), device=pos.device).transpose(-1, -2) # pytorch3d convention to store rotation matrix transposed
    camrot_pos_rot3x3_obj[..., 0:2, :] = -camrot_pos_rot3x3_obj[..., 0:2, :] # pytorch3d convention to have negative x,y axis

    camrot_theta_rot3x3_camrot_pos = torch.stack([
        torch.cos(theta), -torch.sin(theta), zeros, torch.sin(theta), torch.cos(theta), zeros, zeros, zeros, ones
    ], dim=-1).reshape(-1, 3, 3).transpose(-1, -2) # transpose is required because theta is usually given in -z axis instead of +z

    camrot_rot3x3_obj = torch.bmm(camrot_theta_rot3x3_camrot_pos, camrot_pos_rot3x3_obj)
    camrot_transl3_obj = rot3d(pts3d=-pos, rot3x3=camrot_rot3x3_obj)
    camrot_tform4x4_obj = transf4x4_from_rot3x3_and_transl3(camrot_rot3x3_obj, camrot_transl3_obj)

    return camrot_tform4x4_obj

    #dist = pos.norm(dim=-1)
    #azim = torch.atan(pos[..., 0] / -pos[..., 1]) % math.pi + math.pi * (pos[..., 0] < 0) # torch.atan(pos[..., 0] / pos[..., 2])  % math.pi + math.pi * (pos[..., 0] < 0)
    #elev = torch.asin(pos[..., 2] / dist)
    #return transf4x4_from_spherical(azim=azim, elev=elev, theta=theta, dist=dist)


def get_spherical_uniform_tform4x4(azim_min=-math.pi, azim_max=math.pi, azim_steps=5, elev_min=-math.pi/2, elev_max=math.pi/2, elev_steps=5, theta_min=-math.pi/2, theta_max=math.pi/2, theta_steps=5, device='cpu'):
    azim = torch.linspace(start=azim_min, end=azim_max, steps=azim_steps).to(device=device)  # 12
    elev = torch.linspace(start=elev_min, end=elev_max, steps=elev_steps).to(device=device)  # start=-torch.pi / 6, end=torch.pi / 3, steps=4
    theta = torch.linspace(start=theta_min, end=theta_max, steps=theta_steps).to(device=device)  # -torch.pi / 6, end=torch.pi / 6, steps=3

    # dist = torch.linspace(start=eval(config_sample.uniform.dist.min), end=eval(config_sample.uniform.dist.max), steps=config_sample.uniform.dist.steps).to(
    #    device=self.device)
    dist = torch.linspace(start=1., end=1., steps=1).to(device=device)

    azim_shape = azim.shape
    elev_shape = elev.shape
    theta_shape = theta.shape
    dist_shape = dist.shape
    in_shape = azim_shape + elev_shape + theta_shape + dist_shape
    azim = azim[:, None, None, None].expand(in_shape).reshape(-1)
    elev = elev[None, :, None, None].expand(in_shape).reshape(-1)
    theta = theta[None, None, :, None].expand(in_shape).reshape(-1)
    dist = dist[None, None, None, :].expand(in_shape).reshape(-1)
    cams_multiview_tform4x4_cuboid = transf4x4_from_spherical(azim=azim, elev=elev, theta=theta, dist=dist)

    return cams_multiview_tform4x4_cuboid

def get_cam_tform4x4_obj_for_viewpoints_count(viewpoints_count=1, dist: float=1., device=None, dtype=None, spiral=False):

    if not spiral:
        if viewpoints_count == 1:
            # front:
            azim = torch.Tensor([0.])
            elev = torch.Tensor([0.])
            theta = torch.Tensor([0.])
        elif viewpoints_count == 2:
            # front, top
            azim = torch.Tensor([0., 0.])
            elev = torch.Tensor([0., math.pi / 2. - 0.01])
            theta = torch.Tensor([0., 0.])
        elif viewpoints_count == 3:
            # front, top, right
            azim = torch.Tensor([0., 0., math.pi / 2.])
            elev = torch.Tensor([0., math.pi / 2. - 0.01 , 0.])
            theta = torch.Tensor([0., 0., 0.])
        elif viewpoints_count == 4:
            # front, top, right, bottom
            azim = torch.Tensor([0., 0., math.pi / 2., 0.])
            elev = torch.Tensor([0., math.pi / 2. - 0.01 , 0., -math.pi/2. + 0.01])
            theta = torch.Tensor([0., 0., 0., 0.])
        else:
            viewpoints_count_sqrt = math.ceil(math.sqrt(viewpoints_count))
            range_max = 1. - 1./ viewpoints_count_sqrt
            azim = torch.linspace(-math.pi * range_max, math.pi * range_max, viewpoints_count_sqrt)
            elev = torch.linspace(+math.pi / 2. * range_max, -math.pi / 2. * range_max, viewpoints_count_sqrt)
            azim = azim.repeat(viewpoints_count_sqrt)[:viewpoints_count]
            elev = elev.repeat_interleave(viewpoints_count_sqrt)[:viewpoints_count]
            theta = torch.zeros_like(elev)
    else:
        azim = torch.linspace(-math.pi, math.pi, viewpoints_count)
        viewpoints_count_first = viewpoints_count // 2
        viewpoints_count_second = viewpoints_count - viewpoints_count_first
        elev = torch.cat([
            torch.linspace(-math.pi / 2 * 0.4, math.pi / 2 * 0.6, viewpoints_count_first),
            torch.linspace(math.pi / 2 * 0.6, -math.pi / 2 * 0.4, viewpoints_count_second)], dim=0)
        elev = elev
        theta = torch.zeros_like(elev)

    if dist == 0.:
        cam_tform4x4_obj = transf4x4_from_spherical(azim=azim, elev=elev, theta=theta, dist=1.)
        cam_tform4x4_obj[:, :3, 3] = 0.
    else:
        cam_tform4x4_obj = transf4x4_from_spherical(azim=azim, elev=elev, theta=theta, dist=dist)

    if dtype is not None:
        cam_tform4x4_obj = cam_tform4x4_obj.to(dtype=dtype)

    if device is not None:
        cam_tform4x4_obj = cam_tform4x4_obj.to(device=device)

    return cam_tform4x4_obj

def transf4x4_from_spherical(azim, elev, theta, dist):
    # camera center
    obj_transl3_cam = torch.stack([
        dist * torch.cos(elev) * torch.sin(azim),
        -dist * torch.cos(elev) * torch.cos(azim),
        dist * torch.sin(elev)
    ], dim=-1)

    return transf4x4_from_pos_and_theta(obj_transl3_cam, theta)
    """

    cam_transl3_obj = -obj_transl3_cam

    azim = -azim
    elev = -(torch.pi / 2 - elev)

    # rotation matrix
    camazim_tform_cam = torch.stack([
        torch.cos(azim), -torch.sin(azim), zeros, torch.sin(azim), torch.cos(azim), zeros, zeros, zeros, ones
    ], dim=-1).reshape(-1, 3, 3) # .permute(2, 0, 1)

    camelev_tform_camazim = torch.stack([
        ones, zeros, zeros, zeros, torch.cos(elev), -torch.sin(elev), zeros, torch.sin(elev), torch.cos(elev)
    ], dim=-1).reshape(-1, 3, 3) # .permute(2, 0, 1)

    camtheta_tform_camelev = torch.stack([
        torch.cos(theta), -torch.sin(theta), zeros, torch.sin(theta), torch.cos(theta), zeros, zeros, zeros, ones
    ], dim=-1).reshape(-1, 3, 3) # .permute(2, 0, 1)

    camrot_rot3x3_cam = torch.bmm(camtheta_tform_camelev, torch.bmm(camelev_tform_camazim, camazim_tform_cam))

    camrot_transl3_obj = rot3d(pts3d=cam_transl3_obj, rot3x3=camrot_rot3x3_cam)

    camrot_tform4x4_obj = transf4x4_from_rot3x3_and_transl3(camrot_rot3x3_cam, camrot_transl3_obj)

    camrot_tform4x4_obj[:, 1:3, :] = -camrot_tform4x4_obj[:, 1:3, :]

    return camrot_tform4x4_obj
    
    """

    #camrot_tform_obj = np.hstack((camrot_tform_cam, np.dot(camrot_tform_cam, cam_transl3_obj)))
    #camrot_tform_obj = np.vstack((camrot_tform_obj, [0, 0, 0, 1]))

    # T =
    # dist, elev, azim
    #R, t = look_at_view_transform(dist, elev=elev_s, azim=azim_s, degrees=False)

    # return 0.


@torch.jit.script
def transf4x4_from_rot3x3(rot3x3):
    transf4x4 = torch.zeros(rot3x3.shape[:-2] + torch.Size([4, 4]), device=rot3x3.device, dtype=rot3x3.dtype)
    transf4x4[..., :3, :3] = rot3x3
    transf4x4[..., 3, 3] = 1.
    return transf4x4

@torch.jit.script
def transf4x4_from_rot3x3_and_transl3(rot3x3, transl3):
    transf4x4 = transf4x4_from_rot3x3(rot3x3)
    transf4x4[..., :3, 3] = transl3
    return transf4x4

def tform4x4_from_transl3d(transl3d: torch.Tensor):
    """
    Args:
        transl3d (torch.Tensor): ...x3
    Returns:
        tform4x4 (torch.Tensor): ...x4x4
    """
    a_tform4x4_b = torch.eye(4)[(None,)* (len(transl3d.shape) - 1)].expand(transl3d.shape[:-1] + torch.Size([4, 4])).to(device=transl3d.device, dtype=transl3d.dtype)
    a_tform4x4_b[..., :3, 3] = transl3d
    return a_tform4x4_b


def rot2d(pts2d, rot2x2):
    pts2d_shape_in = pts2d.shape
    pts2d_rot = torch.bmm(rot2x2.reshape(-1, 2, 2), pts2d.reshape(-1, 2, 1)).reshape(pts2d_shape_in)
    return pts2d_rot

@torch.jit.script
def rot3d(pts3d, rot3x3):
    pts3d_shape_in = pts3d.shape
    pts3d_rot = torch.bmm(rot3x3.reshape(-1, 3, 3), pts3d.reshape(-1, 3, 1)).reshape(pts3d_shape_in)
    return pts3d_rot

def rot3d_broadcast(pts3d, rot3x3):
    shape_first_dims = torch.broadcast_shapes(pts3d.shape[:-1], rot3x3.shape[:-2])
    return rot3d(pts3d.expand(*shape_first_dims, 3), rot3x3.expand(*shape_first_dims, 3, 3))


def proj3d2d_broadcast(pts3d, proj4x4):
    shape_first_dims = torch.broadcast_shapes(pts3d.shape[:-1], proj4x4.shape[:-2])
    return proj3d2d(pts3d.expand(*shape_first_dims, 3), proj4x4.expand(*shape_first_dims, 4, 4))


def pts3d_to_pts4d(pts3d):
    device = pts3d.device
    dtype = pts3d.dtype
    ones1d = torch.ones(size=list(pts3d.shape[:-1]) + [1]).to(device=device, dtype=dtype)
    pts4d = torch.concatenate([pts3d, ones1d], dim=-1)
    return pts4d

def pts2d_to_pts3d(pts2d):
    device = pts2d.device
    dtype = pts2d.dtype
    ones1d = torch.ones(size=list(pts2d.shape[:-1]) + [1]).to(device=device, dtype=dtype)
    pts3d = torch.concatenate([pts2d, ones1d], dim=-1)
    return pts3d

def pts2d_to_pts4d(pts2d):
    device = pts2d.device
    dtype = pts2d.dtype
    ones2d = torch.ones(size=list(pts2d.shape[:-1]) + [2]).to(device=device, dtype=dtype)
    pts4d = torch.concatenate([pts2d, ones2d], dim=-1)
    pts4d = pts4d.reshape(list(pts4d.shape) + [1])
    return pts4d

def proj3d2d_origin(proj4x4):
    device = proj4x4.device
    dtype = proj4x4.dtype
    pts3d = torch.zeros(size=proj4x4.shape[:-2] + torch.Size([3,]), device=device, dtype=dtype)
    return proj3d2d(pts3d=pts3d, proj4x4=proj4x4)


def add_homog_dim(pts, dim):
    device = pts.device
    dtype = pts.dtype
    if dim == -1:
        dim = pts.dim() - 1
    ones1d = torch.ones(size=list(pts.shape[:dim]) + [1] + list(pts.shape[dim+1:])).to(device=device, dtype=dtype)
    return torch.cat([pts, ones1d], dim=dim)



def proj3d2d(pts3d, proj4x4):
    device = pts3d.device
    dtype = pts3d.dtype
    ones1d = torch.ones(size=list(pts3d.shape[:-1]) + [1]).to(device=device, dtype=dtype)
    pts4d = torch.concatenate([pts3d, ones1d], dim=-1)
    pts4d = pts4d.reshape(list(pts4d.shape) + [1])
    pts4d_transf = torch.bmm(proj4x4.reshape(-1, 4, 4), pts4d.reshape(-1, 4, 1)).reshape(pts4d.shape)
    dim_coords3d = pts4d_transf.dim() - 2
    pts3d_transf = pts4d_transf.index_select(dim=dim_coords3d, index=torch.LongTensor([0, 1, 2]).to(device=device))
    pts2d_transf = pts3d_transf.index_select(dim=dim_coords3d, index=torch.LongTensor([0, 1]).to(device=device))
    ptsZ_transf = pts3d_transf.index_select(dim=dim_coords3d, index=torch.LongTensor([2]).to(device=device))
    pts2d_transf_proj = pts2d_transf / ptsZ_transf
    pts2d_transf_proj = pts2d_transf_proj.squeeze(dim=-1)
    return pts2d_transf_proj

def reproj2d3d_broadcast(pxl2d, proj4x4_inv):
    shape_first_dims = torch.broadcast_shapes(pxl2d.shape[:-1], proj4x4_inv.shape[:-2])
    return reproj2d3d(pxl2d.expand(*shape_first_dims, 2), proj4x4_inv.expand(*shape_first_dims, 4, 4))

def reproj2d3d(pxl2d, proj4x4_inv):
    device = pxl2d.device
    pts4d = pts2d_to_pts4d(pxl2d)
    pts4d_reproj = torch.bmm(proj4x4_inv.reshape(-1, 4, 4), pts4d.reshape(-1, 4, 1)).reshape(pts4d.shape)
    dim_coords2d = pts4d_reproj.dim() - 2
    pts3d_reproj = pts4d_reproj.index_select(dim=dim_coords2d, index=torch.LongTensor([0, 1, 2]).to(device=device))
    pts3d_reproj = pts3d_reproj.squeeze(dim=-1)
    return pts3d_reproj

def depth2pts3d_grid(depth, cam_intr4x4):
    #  depth: ...x1xHxW
    #  cam_intr: ...x4x4
    device = cam_intr4x4.device
    dtype = cam_intr4x4.dtype
    H, W = depth.shape[-2:]
    pxl2d = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'), dim=-1).to(device=device, dtype=dtype)
    # pxl2d[(None, ) * (cam_intr4x4.dim() - 2)] # legacy code, do not use for not broadcasting cam intr...
    pts3d_homog = reproj2d3d_broadcast(pxl2d, proj4x4_inv=cam_intr4x4[..., None, None, :, :].inverse()).transpose(-2, -1).transpose(-3, -2)
    shape_first_dims = torch.broadcast_shapes(depth.shape[:-3], cam_intr4x4.shape[:-2])
    pts3d = pts3d_homog.expand(*shape_first_dims, 3, H, W) * depth.expand(*shape_first_dims, 1, H, W)
    return pts3d

from od3d.cv.differentiation.gradient import calc_batch_gradients
def depth2normals_grid(depth, cam_intr4x4, shift=1):
    # depth: ...x1xHxW
    # cam_intr: ...x4x4
    device = depth.device
    dtype = depth.dtype

    fx = cam_intr4x4[..., 0, 0]
    fy = cam_intr4x4[..., 1, 1]

    dz_dpx, dz_dpy = calc_batch_gradients(depth, pad_zeros=True, shift=shift)
    dpx_dx = fx[..., None, None, None] / (shift * depth)
    dpy_dy = fy[..., None, None, None] / (shift * depth)
    dz_dx = dz_dpx * dpx_dx
    dz_dy = dz_dpy * dpy_dy
    dz_dx = dz_dx.nan_to_num(0., neginf=0., posinf=0.)
    dz_dy = dz_dy.nan_to_num(0., neginf=0., posinf=0.)
    normals = torch.zeros(size=depth.shape[:-3] + (3,) + depth.shape[-2:]).to(dtype=dtype, device=device)
    normals[..., 2, :, :] = -1.
    normals[..., 0:1, :, :] = dz_dx
    normals[..., 1:2, :, :] = dz_dy
    normals = torch.nn.functional.normalize(normals, dim=-3)
    return normals

    # pxl2d = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'), dim=-1).to(device=device, dtype=dtype)
    # # pxl2d[(None, ) * (cam_intr4x4.dim() - 2)] # legacy code, do not use for not broadcasting cam intr...
    # pts3d_homog = reproj2d3d_broadcast(pxl2d, proj4x4_inv=cam_intr4x4[..., None, None, :, :].inverse()).transpose(-2, -1).transpose(-3, -2)
    # shape_first_dims = torch.broadcast_shapes(depth.shape[:-3], cam_intr4x4.shape[:-2])
    # pts3d = pts3d_homog.expand(*shape_first_dims, 3, H, W) * depth.expand(*shape_first_dims, 1, H, W)
    # return pts3d


def transf3d_normal_broadcast(normals3d, transf4x4):
    shape_first_dims = torch.broadcast_shapes(normals3d.shape[:-1], transf4x4.shape[:-2])
    transf4x4_zero_transl = transf4x4.clone()
    transf4x4_zero_transl[..., :3, 3] = 0.
    return transf3d(normals3d.expand(*shape_first_dims, 3), transf4x4_zero_transl.expand(*shape_first_dims, 4, 4))

def transf3d_broadcast(pts3d, transf4x4):
    shape_first_dims = torch.broadcast_shapes(pts3d.shape[:-1], transf4x4.shape[:-2])
    return transf3d(pts3d.expand(*shape_first_dims, 3), transf4x4.expand(*shape_first_dims, 4, 4))

def cam_intr_4_to_4x4(cam_intr4):
    """
    Args:
        cam_intr4: ...x4, [fx, fy, cx, cy]
    Returns:
        cam_intr_4x4: ...x4x4, [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    """
    cam_intr4x4 = torch.eye(4).expand(cam_intr4.shape[:-1] + (4, 4)).clone()
    cam_intr4x4[..., 0, 0] = cam_intr4[..., 0]
    cam_intr4x4[..., 1, 1] = cam_intr4[..., 1]
    cam_intr4x4[..., 0, 2] = cam_intr4[..., 2]
    cam_intr4x4[..., 1, 2] = cam_intr4[..., 3]
    return cam_intr4x4
def transf3d(pts3d, transf4x4):
    """
    Args:
        pts3d (torch.Tensor): ...x3
        transf4x4 (torch.Tensor): ...x4x4

    Returns:
        pts3d_transf (torch.Tensor): ...x3
    """
    device = pts3d.device
    dtype = pts3d.dtype
    ones1d = torch.ones(size=list(pts3d.shape[:-1]) + [1]).to(device=device, dtype=dtype)
    pts4d = torch.concatenate([pts3d, ones1d], dim=-1)
    pts4d = pts4d.reshape(list(pts4d.shape) + [1])
    pts4d_transf = torch.bmm(transf4x4.reshape(-1, 4, 4), pts4d.reshape(-1, 4, 1)).reshape(pts4d.shape)
    dim_coords3d = pts4d_transf.dim() - 2
    pts3d_transf = pts4d_transf.index_select(dim=dim_coords3d, index=torch.LongTensor([0, 1, 2]).to(device=device))
    pts3d_transf = pts3d_transf.squeeze(dim=-1)
    return pts3d_transf


def transf2d(pts2d, transf3x3):
    """
    Args:
        pts2d (torch.Tensor): ...x2
        transf3x3 (torch.Tensor): ...x3x3

    Returns:
        pts2d_transf (torch.Tensor): ...x2
    """
    device = pts2d.device
    dtype = pts2d.dtype
    ones1d = torch.ones(size=list(pts2d.shape[:-1]) + [1]).to(device=device, dtype=dtype)
    pts3d = torch.concatenate([pts2d, ones1d], dim=-1)
    pts3d = pts3d.reshape(list(pts3d.shape) + [1])
    pts2d_transf = torch.bmm(transf3x3.reshape(-1, 3, 3), pts3d.reshape(-1, 3, 1)).reshape(pts3d.shape)
    dim_coords2d = pts2d_transf.dim() - 2
    pts2d_transf = pts2d_transf.index_select(dim=dim_coords2d, index=torch.LongTensor([0, 1]).to(device=device))
    pts2d_transf = pts2d_transf.squeeze(dim=-1)
    return pts2d_transf


def plane4d_to_tform4x4(plane4d: torch.Tensor):
    """
    Args:
        plane4d (torch.Tensor): ...x4, first 3 dimensions are axis, last dimension offset.
    Returns:
        plan3d_tform_pts (torch.Tensor): ...x4x4

    """
    device = plane4d.device
    plane3d_tform4x4_obj = torch.eye(4).to(device=device)
    top_axis = plane4d[:3] / plane4d[:3].norm()
    x = top_axis[0]
    y = top_axis[1]
    z = top_axis[2]
    if x != 0 or y != 0:
        left_axis = torch.Tensor([-y, x, 0.]).to(device=device)
        left_axis = left_axis / left_axis.norm()
        back_axis = torch.Tensor([-x * z, -y * z, x * x + y * y]).to(device=device)
        back_axis = back_axis / back_axis.norm()
    else:
        left_axis = torch.Tensor([1., 0., 0.], device=device)
        back_axis = torch.Tensor([0., 1., 0.], device=device)
    plane3d_tform4x4_obj[0, :3] = left_axis
    plane3d_tform4x4_obj[1, :3] = back_axis
    plane3d_tform4x4_obj[2, :3] = top_axis
    plane3d_tform4x4_obj[2, 3] = -plane4d[3]

    return plane3d_tform4x4_obj

