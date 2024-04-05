import logging
logger = logging.getLogger(__name__)
import torch
from od3d.cv.geometry.downsample import voxel_downsampling, random_sampling
from od3d.cv.geometry.transform import proj3d2d_broadcast, transf3d_broadcast
from od3d.cv.visual.sample import sample_pxl2d_pts

def get_pcl_clean_with_masks(pcl, masks, cams_tform4x4_obj, cams_intr4x4, pts3d_prob_thresh=0.6,
                             pts3d_count_min=10, pts3d_max_count=20000, return_mask=False):
    """
    Args:

    Returns:

    """
    # cams_proj4x4_obj = tform4x4(cams_intr4x4, cams_tform4x4_obj)
    pcl_sampled, mask_pcl = random_sampling(pcl, pts3d_max_count=pts3d_max_count, return_mask=True)

    pts3d_prob = torch.zeros(size=(pcl_sampled.shape[0], 1), device=pcl.device, dtype=pcl.dtype)
    cam_tform_pts3d = transf3d_broadcast(pts3d=pcl_sampled, transf4x4=cams_tform4x4_obj[:, None])
    pxl2d = proj3d2d_broadcast(pts3d=cam_tform_pts3d, proj4x4=cams_intr4x4[:, None])
    pts3d_prob += ((cam_tform_pts3d[:, :, 2:] > 0.) * sample_pxl2d_pts(masks, pxl2d=pxl2d, padding_mode='value',
                                                                       padding_value=0.)).sum(dim=0)
    pts3d_prob = pts3d_prob / len(masks)
    mask_pcl[mask_pcl == True] *= pts3d_prob[..., 0] > pts3d_prob_thresh
    pcl_clean = pcl[mask_pcl]

    if len(pcl_clean) < pts3d_count_min:
        logger.warning(f'less than {pts3d_count_min} in cleaned pcl')
        pcl_clean, mask_pcl = random_sampling(pcl, pts3d_max_count=pts3d_count_min, return_mask=True)

    if not return_mask:
        return pcl_clean
    else:
        return pcl_clean, mask_pcl