from logging import getLogger
logger = getLogger(__file__)
from od3d.cv.geometry.transform import inv_tform4x4

import torch

def fit_rays_center3d(cams_tform4x4_obj: torch.Tensor):
    """
    Args:
        cams_tform4x4_obj (torch.Tensor): Cx4x4

    Returns:
        center3d (torch.Tensor): 3,

    """

    device=cams_tform4x4_obj.device
    obj_cams_rays6d = torch.cat([inv_tform4x4(cams_tform4x4_obj)[:, :3, 3], cams_tform4x4_obj[:, 2, :3]], dim=-1).to(
        device=device)
    obj_cams_rays_start = obj_cams_rays6d[:, :3]
    obj_cams_rays_dir = obj_cams_rays6d[:, 3:]
    #obj_cams_rays_end = obj_cams_rays_start + obj_cams_rays_dir * scene_size

    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    S = (torch.eye(3).to(device=device)[None,] - obj_cams_rays_dir[:, None, :] * obj_cams_rays_dir[:, :, None]).sum(
        dim=0)
    C = torch.einsum('BXY,BY->BX', (torch.eye(3).to(device=device)[None,] - (
                obj_cams_rays_dir[:, None, :] * obj_cams_rays_dir[:, :, None])), obj_cams_rays_start).sum(dim=0)
    center3d = torch.linalg.solve(A=S, B=C)

    return center3d