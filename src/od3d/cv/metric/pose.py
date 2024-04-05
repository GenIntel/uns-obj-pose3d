import logging
logger = logging.getLogger(__name__)
import torch
import pytorch3d
import pytorch3d.transforms
from od3d.cv.geometry.transform import rot3x3, inv_tform4x4
import math

def get_pose_diff_in_rad(pred_tform4x4: torch.Tensor, gt_tform4x4: torch.Tensor):
    """
    Args:
        pred_tform4x4 (torch.Tensor): ...x4x4
        gt_tform4x4 (torch.Tensor): ...x4x4
    Returns:
        rot_diff_in_rad (torch.Tensor): ...
    """

    pred_ref_tform_src_scaled = pred_tform4x4.clone()
    pred_ref_tform_src_scaled[..., :3, :3] /= torch.linalg.norm(pred_ref_tform_src_scaled[..., :3, :3], dim=-1,
                                                           keepdim=True)

    gt_ref_tform_src_scaled = gt_tform4x4.clone()
    gt_ref_tform_src_scaled[..., :3, :3] /= torch.linalg.norm(gt_ref_tform_src_scaled[..., :3, :3], dim=-1,
                                                         keepdim=True)
    diff_rot3x3 = rot3x3(inv_tform4x4(gt_ref_tform_src_scaled)[..., :3, :3], pred_ref_tform_src_scaled[..., :3, :3])[..., :3, :3]

    try:
        diff_so3_log = pytorch3d.transforms.so3_log_map(diff_rot3x3.reshape(-1, 3, 3)).reshape(*diff_rot3x3.shape[:-2], 3)
        diff_rot_angle_rad = torch.norm(diff_so3_log, dim=-1)
    except ValueError:
        logger.warning(
            f'Cannot calculate deviation in rotation angle due to rot3x3 trace being too small, setting deviation to PI.')
        diff_rot_angle_rad = torch.ones_like(diff_rot3x3[..., 0, 0]) * math.pi

    if not torch.isfinite(diff_rot_angle_rad).all():
        logger.warning(f'Nan or Inf in diff {diff_rot_angle_rad}')
        # toytruck rotation 0 0 0 0 0 0 00 0 -> nan for :  ref_mesh_id: 1, src_mesh_id: (4,)
        diff_rot_angle_rad[~diff_rot_angle_rad.isfinite()] = math.pi


    return diff_rot_angle_rad