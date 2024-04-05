import logging
logger = logging.getLogger(__name__)
import torch
from od3d.cv.geometry.transform import rot3x3, transf4x4_from_rot3x3

def axis_rot3x3_obj_from_pts3d(axis_pts3d: torch.Tensor):
    """
    Args:
        axis_pts3d (torch.Tensor): 3x2x3
    Returns:
        axis_rot3x3_obj (torch.Tensor): 3x3
    """
    obj_axis3d = axis_pts3d[:6].reshape(3, 2, 3)[:, 0] - axis_pts3d[:6].reshape(3, 2, 3)[:, 1]
    logger.info(f'Axis \n{obj_axis3d}')

    obj_axis3d = torch.nn.functional.normalize(obj_axis3d, dim=-1)
    logger.info(f'Axis Normalized \n{obj_axis3d}')
    U, S, V = torch.linalg.svd(obj_axis3d)
    axis3d_rot3x3_obj = rot3x3(U, rot3x3(torch.diag(S.sign()), V))
    return axis3d_rot3x3_obj


def axis_tform4x4_obj_from_pts3d(axis_pts3d: torch.Tensor):
    """
    Args:
        axis_pts3d (torch.Tensor): 3x2x3
    Returns:
        axis_tform4x4_obj (torch.Tensor): 4x4
    """

    axis_rot3x3_obj = axis_rot3x3_obj_from_pts3d(axis_pts3d=axis_pts3d)
    axis_tform4x4_obj = transf4x4_from_rot3x3(rot3x3=axis_rot3x3_obj)
    return axis_tform4x4_obj