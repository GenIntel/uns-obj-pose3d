import logging
logger = logging.getLogger(__name__)
from pathlib import Path
import torch
import os
from od3d.io import run_cmd
import open3d
from od3d.cv.geometry.transform import tform4x4_from_transl3d, tform4x4_broadcast, inv_tform4x4, transf3d_broadcast, depth2pts3d_grid, depth2normals_grid
from od3d.cv.visual.resize import resize
from od3d.cv.io import write_pts3d_with_colors_and_normals
import re
import numpy as np


def atoi(text):
    return int(text) if text.isdigit() else text

def sorted_image_list(image_list):
    return sorted(image_list, key=lambda f: [atoi(val) for val in re.split(r'(\d+)', Path(f).stem)])

def run_droid_slam(path_rgbs:Path, path_out_root:Path, rpath_out:Path, cam_intr4x4: torch.Tensor, path_masks=None,
                   pts3d_count_min = 10, pts3d_max_count = 20000, pts3d_prob_thresh = 0.6,
                   pcl_fname='pcl.ply', rays_center3d_fname='rays_center3d.pt', cam_tform_obj_dname='cam_tform4x4_obj'):

    stride = "1"
    image_tag = "limpbot/droid-slam:v1"
    path_out = path_out_root.joinpath(rpath_out)
    fpath_out_pcl = path_out.joinpath(pcl_fname)
    fpath_out_center3d = path_out.joinpath(rays_center3d_fname)
    path_out_cam_tform_obj = path_out.joinpath(cam_tform_obj_dname)

    fx = cam_intr4x4[0][0]
    fy = cam_intr4x4[1][1]
    cx = cam_intr4x4[0][2]
    cy = cam_intr4x4[1][2]

    if not path_out.exists():
        path_out.mkdir(parents=True, exist_ok=True)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if len(cuda_visible_devices) == 0:
        cuda_visible_devices = 'all'
    else:
        cuda_visible_devices = f'{cuda_visible_devices}'

    torch.cuda.empty_cache()
    run_cmd(cmd=f'echo "{fx} {fy} {cx} {cy}" > {path_out_root}/{rpath_out}/calib.txt', logger=logger)
    run_cmd(
        cmd=f'docker run --user=$(id -u):$(id -g) --gpus device={cuda_visible_devices} -e RPATH_OUT={rpath_out} -e STRIDE={stride} -v {path_rgbs}:/home/appuser/in -v {path_out_root}:/home/appuser/DROID-SLAM/reconstructions/out -t {image_tag}',
        logger=logger, live=True)

    device = 'cuda'
    dtype = torch.float
    cams_tform4x4_obj = torch.load(path_out.joinpath('traj_est.pt')).to(device=device)
    #partial_disps_up = torch.load(path_out.joinpath('disps_up.pt'))[:, None].to(device=device)
    #partial_masks_up = torch.load(path_out.joinpath('masks_up.pt'))[:, None].to(device=device)
    partial_tstamps = torch.load(path_out.joinpath('tstamps.pt')).to(dtype=torch.long, device=device)
    partial_intr4 = torch.load(path_out.joinpath('intr.pt')).to(dtype=torch.long, device=device)[0]
    partial_cams_intr4x4 = torch.eye(4).to(device)
    partial_cams_intr4x4[0, 0] = partial_intr4[0]
    partial_cams_intr4x4[1, 1] = partial_intr4[1]
    partial_cams_intr4x4[0, 2] = partial_intr4[2]
    partial_cams_intr4x4[1, 2] = partial_intr4[3]
    partial_cams_intr4x4 = torch.stack([partial_cams_intr4x4.clone() for _ in range(len(partial_tstamps))], dim=0).to(device=device)


    partial_disps_up = torch.load(path_out.joinpath('disps.pt'))[:, None].to(device=device)
    partial_masks_up = torch.load(path_out.joinpath('masks.pt'))[:, None].to(device=device)

    fpaths_rgbs = sorted_image_list(list(path_rgbs.iterdir()))
    from od3d.cv.io import read_image
    first_rgb = read_image(fpaths_rgbs[0])
    h0, w0 = first_rgb.shape[1:]
    h1, w1 = partial_disps_up.shape[2:]

    partial_cams_tform4x4_obj = cams_tform4x4_obj[partial_tstamps]
    #frames = self.get_frames(frames_ids=partial_tstamps)
    #partial_disps_up = resize(partial_disps_up, mode="nearest_v2", H_out=h0, W_out=w0)
    #partial_masks_up = resize(partial_masks_up, mode="nearest_v2", H_out=h0, W_out=w0)
    # resize_w0 = w1 / w0
    # resize_h0 = h1 / h0
    # partial_cams_intr4x4 = torch.stack([cam_intr4x4 for _ in range(len(partial_cams_tform4x4_obj))], dim=0).to(device=device)
    # partial_cams_intr4x4[:, 0] *= resize_w0
    # partial_cams_intr4x4[:, 1] *= resize_h0

    partial_depths = 1. / partial_disps_up
    partial_depths = partial_depths.nan_to_num(0., posinf=0., neginf=0.).to(torch.float)

    rgbs = torch.stack([read_image(fpath_rgb) for fpath_rgb in fpaths_rgbs], dim=0).to(device=device)
    partial_rgb = resize(rgbs[partial_tstamps], H_out=h1, W_out=w1, align_corners=True)

    spts = 1

    partial_depths_valid = (partial_disps_up > partial_disps_up.flatten(2).mean(dim=-1)[
        ..., None, None]) * partial_depths.isfinite() * (partial_depths > 0.) * partial_masks_up
    partial_depths[~partial_depths_valid] = 0.

    # masks_up = ((count_up >= 2) & (disps_up > .5 * disps_up.mean(dim=[1, 2], keepdim=True)))

    # note: this operation is highly compute intensive
    pts3d = \
       transf3d_broadcast(
           depth2pts3d_grid(partial_depths[::spts], partial_cams_intr4x4[::spts]).permute(0, 2, 3, 1),
           inv_tform4x4(partial_cams_tform4x4_obj[::spts])[:, None, None])[
           partial_depths_valid[::spts][..., 0, :, :]]

    # pts3d = []
    # bs = 1
    # for b in range(0, len(partial_depths), bs):
    #     pts3d_b = \
    #         transf3d_broadcast(
    #             depth2pts3d_grid(partial_depths[b:b+bs], partial_cams_intr4x4[b:b+bs]).permute(0, 2, 3, 1),
    #             inv_tform4x4(partial_cams_tform4x4_obj[b:b+bs])[:, None, None])[
    #             partial_depths_valid[b:b+bs][..., 0, :, :]]
    #     pts3d.append(pts3d_b)
    # pts3d = torch.cat(pts3d, dim=0)

    pts3d_colors = partial_rgb[::spts].permute(0, 2, 3, 1)[partial_depths_valid[::spts][..., 0, :, :]] / 255.

    # note: this operation is highly compute intensive
    pts3d_normals = transf3d_broadcast(
        depth2normals_grid(partial_depths[::spts], partial_cams_intr4x4[::spts], shift=w0 // 200).permute(0, 2, 3,
                                                                                                          1),
        inv_tform4x4(partial_cams_tform4x4_obj[::spts])[:, None, None])[partial_depths_valid[::spts][..., 0, :, :]]
    # pts3d_normals = []
    # for b in range(0, len(partial_depths), bs):
    #     pts3d_normals_b = transf3d_broadcast(
    #         depth2normals_grid(partial_depths[b:b+bs], partial_cams_intr4x4[b:b+bs], shift=max(1, int(0.02 * w1))).permute(0, 2, 3,
    #                                                                                                           1),
    #         inv_tform4x4(partial_cams_tform4x4_obj[b:b+bs])[:, None, None])[partial_depths_valid[b:b+bs][..., 0, :, :]]
    #     pts3d_normals.append(pts3d_normals_b)
    # pts3d_normals = torch.cat(pts3d_normals, dim=0)

    ## DEBUG BLOCK START
    # o3d_pcl = open3d.geometry.PointCloud()
    # o3d_pcl.points = open3d.utility.Vector3dVector(pts3d.detach().cpu().numpy())
    # o3d_pcl.normals = open3d.utility.Vector3dVector(pts3d_normals.detach().cpu().numpy())  # invalidate existing normals
    # o3d_pcl.colors = open3d.utility.Vector3dVector(pts3d_colors.detach().cpu().numpy())
    # open3d.visualization.draw_geometries([o3d_pcl])
    ## DEBUG BLOCK END

    #if path_masks is not None:
    #    masks = torch.stack([read_image(fpath_mask) for fpath_mask in sorted_image_list(list(path_masks.iterdir()))], dim=0).to(device=device)[partial_tstamps]
    #else:
    #    masks = torch.ones_like(rgbs[:, :1]).to(device=device)
    #
    #cams_intr4x4 = torch.stack([cam_intr4x4 for _ in range(len(cams_tform4x4_obj))], dim=0).to(device=device)
    # pts3d_obj, pts3d_obj_mask = get_pcl_clean_with_masks(pcl=pts3d, masks=masks,
    #                                                      cams_intr4x4=cams_intr4x4,
    #                                                      cams_tform4x4_obj=cams_tform4x4_obj,
    #                                                      pts3d_prob_thresh=pts3d_prob_thresh,
    #                                                      pts3d_max_count=pts3d_max_count,
    #                                                      pts3d_count_min=pts3d_count_min,
    #                                                      return_mask=True)

    # pts3d_clean, pts3d_clean_mask = self.get_pcl_clean_with_focus_point_and_plane_removal(
    #     pts3d=pts3d, pts3d_colors=pts3d_colors, cams_tform4x4_obj=cams_tform4x4_obj,
    #     pts3d_count_min=pts3d_count_min, return_mask=True
    # )

    pts3d_obj = pts3d
    pts3d_colors_obj = pts3d_colors # [pts3d_obj_mask]
    pts3d_normals_obj = pts3d_normals # [pts3d_obj_mask]

    ## DEBUG BLOCK START
    # scams = 30
    # frames = self.get_frames()
    # rgb = torch.stack([frame.rgb for frame in frames], dim=0).to(device=device)
    # cams_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=device)
    # show_scene(pts3d=[pts3d_clean], pts3d_colors=[pts3d_colors_clean], cams_tform4x4_world=cams_tform4x4_obj[::scams], cams_intr4x4=cams_intr4x4[::scams], cams_imgs=rgb[::scams])
    # o3d_pcl = open3d.geometry.PointCloud()
    # o3d_pcl.points = open3d.utility.Vector3dVector(pts3d_clean.detach().cpu().numpy())
    # o3d_pcl.normals = open3d.utility.Vector3dVector(
    #     pts3d_normals_clean.detach().cpu().numpy())  # invalidate existing normals
    # o3d_pcl.colors = open3d.utility.Vector3dVector(pts3d_colors_clean.detach().cpu().numpy())
    # open3d.visualization.draw_geometries([o3d_pcl])
    ## DEBUG BLOCK END


    center3d = pts3d_obj.mean(axis=0)
    center3d_tform4x4_obj = tform4x4_from_transl3d(-center3d).to(device=device)

    center3d_pts3d_clean = transf3d_broadcast(pts3d_obj, center3d_tform4x4_obj)

    ## DEBUG BLOCK START
    o3d_pcl = open3d.geometry.PointCloud()
    o3d_pcl.points = open3d.utility.Vector3dVector(pts3d_obj.detach().cpu().numpy())
    if (pts3d_normals_obj == 0.).all():
        o3d_pcl.estimate_normals()
        pts3d_normals_obj = torch.from_numpy(np.array(o3d_pcl.normals)).to(device=pts3d_normals_obj.device,
                                                                             dtype=pts3d_normals_obj.dtype)
    else:
        o3d_pcl.normals = open3d.utility.Vector3dVector(
            pts3d_normals_obj.detach().cpu().numpy())  # invalidate existing normals
    o3d_pcl.colors = open3d.utility.Vector3dVector(pts3d_colors_obj.detach().cpu().numpy())
    # open3d.visualization.draw_geometries([o3d_pcl])
    ## DEBUG BLOCK END

    fpath_out_pcl.parent.mkdir(parents=True, exist_ok=True)
    write_pts3d_with_colors_and_normals(fpath=fpath_out_pcl,
                                        pts3d=center3d_pts3d_clean.detach().cpu(),
                                        pts3d_colors=pts3d_colors_obj.detach().cpu(),
                                        pts3d_normals=pts3d_normals_obj.detach().cpu())

    cams_tform4x4_obj_centered = tform4x4_broadcast(cams_tform4x4_obj, inv_tform4x4(center3d_tform4x4_obj))
    # save transformation
    for i, cam_tform4x4_obj in enumerate(cams_tform4x4_obj_centered):
        frame_name = fpaths_rgbs[i].stem
        fpath_out_cam_tform_obj = path_out_cam_tform_obj.joinpath(f'{frame_name}.pt')
        fpath_out_cam_tform_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj=cam_tform4x4_obj.detach().cpu(), f=fpath_out_cam_tform_obj)

    from od3d.cv.geometry.fit.rays_center3d import fit_rays_center3d
    center3d = fit_rays_center3d(cams_tform4x4_obj=cams_tform4x4_obj_centered)
    torch.save(center3d.detach().cpu(), f=fpath_out_center3d)