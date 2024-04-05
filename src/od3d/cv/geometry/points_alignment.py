from od3d.cv.geometry.transform import transf3d_broadcast, transf4x4_from_rot3x3_and_transl3
from pytorch3d.ops.points_alignment import iterative_closest_point
from od3d.cv.geometry.transform import rot3d
import scipy.linalg
import torch


def get_pca_tform_world(pts3d):

    origin = pts3d.mean(dim=-2)
    U, S, V = torch.pca_lowrank(pts3d)
    pca_tform_world = transf4x4_from_rot3x3_and_transl3(rot3x3=V.T, transl3=-rot3d(rot3x3=V.T, pts3d=origin))

    return pca_tform_world

def icp(frame1_pts3d_1, frame2_pts3d_2):
    """
    Args:
        frame1_pts3d_1 (torch.Tensor): size N1x3
        frame2_pts3d_2 (torch.Tensor): size N2x3

    Returns:
        frame2_tform_pts3d_1 (torch.Tensor): size 4x4
    """

    icp_sol = iterative_closest_point(X=frame1_pts3d_1[None,].cuda(), Y=frame2_pts3d_2[None,].cuda(),
                                      estimate_scale=False, verbose=True, max_iterations=300)
    frame2_tform_frame1 = \
    transf4x4_from_rot3x3_and_transl3(rot3x3=icp_sol.RTs.R.transpose(-1, -2), transl3=icp_sol.RTs.T)[0].cpu()

    return frame2_tform_frame1

def orthogonal_procrustes(frame1_pts3d_1, frame2_pts3d_2):
    """
    Args:
        frame2_pts3d_2 (torch.Tensor): size Nx3
        frame2_pts3d_2 (torch.Tensor): size Nx3

    Returns:
        frame2_tform_pts3d_1 (torch.Tensor): size 4x4
    """

    frame2_transl_frame1 = frame2_pts3d_2.mean(dim=0) - frame1_pts3d_1.mean(dim=0)
    frame2_rot3x3_frame1, res = scipy.linalg.orthogonal_procrustes(frame2_pts3d_2, frame1_pts3d_1 + frame2_transl_frame1, check_finite=False)
    frame2_rot3x3_frame1 = torch.from_numpy(frame2_rot3x3_frame1).to(device=frame1_pts3d_1.device, dtype=frame1_pts3d_1.dtype)
    frame2_transl_frame1 = rot3d(rot3x3=frame2_rot3x3_frame1.T, pts3d=frame2_transl_frame1)
    frame2_tform_frame1 = transf4x4_from_rot3x3_and_transl3(rot3x3=frame2_rot3x3_frame1, transl3=frame2_transl_frame1)

    return frame2_tform_frame1