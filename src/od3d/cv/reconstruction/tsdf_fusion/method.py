# import time
#
# import open3d as o3d
# import open3d.core as o3c
# from tqdm import tqdm
#
# from common import load_rgbd_file_names, load_depth_file_names, load_intrinsic, load_extrinsics, get_default_dataset
# from config import ConfigParser
#
#
# def integrate(depth_file_names, color_file_names, depth_intrinsic,
#               color_intrinsic, extrinsics, config):
#     n_files = len(depth_file_names)
#     device = o3d.core.Device('CUDA:0') # 'CUDA:0' , 'CPU
#
#     if config.integrate_color:
#         vbg = o3d.t.geometry.VoxelBlockGrid(
#             attr_names=('tsdf', 'weight', 'color'),
#             attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
#             attr_channels=((1), (1), (3)),
#             voxel_size=3.0 / 512,
#             block_resolution=16,
#             block_count=50000,
#             device=device)
#     else:
#         vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight'),
#                                             attr_dtypes=(o3c.float32,
#                                                          o3c.float32),
#                                             attr_channels=((1), (1)),
#                                             voxel_size=3.0 / 512,
#                                             block_resolution=16,
#                                             block_count=50000,
#                                             device=device)
#
#     start = time.time()
#     for i in tqdm(range(n_files)):
#         depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
#         extrinsic = extrinsics[i]
#
#         frustum_block_coords = vbg.compute_unique_block_coordinates(
#             depth, depth_intrinsic, extrinsic, config.depth_scale,
#             config.depth_max)
#
#         if config.integrate_color:
#             color = o3d.t.io.read_image(color_file_names[i]).to(device)
#             vbg.integrate(frustum_block_coords, depth, color, depth_intrinsic,
#                           color_intrinsic, extrinsic, config.depth_scale,
#                           config.depth_max)
#         else:
#             vbg.integrate(frustum_block_coords, depth, depth_intrinsic,
#                           extrinsic, config.depth_scale, config.depth_max)
#         dt = time.time() - start
#     print('Finished integrating {} frames in {} seconds'.format(n_files, dt))
#
#
#     pcd = vbg.extract_point_cloud()
#     o3d.visualization.draw([pcd])
#
#     mesh = vbg.extract_triangle_mesh()
#     o3d.visualization.draw([mesh.to_legacy()])
#
#     return vbg

import logging
logger = logging.getLogger(__name__)

from od3d.cv.reconstruction.tsdf_fusion.fusion import TSDFVolume, get_view_frustum, pcwrite, meshwrite
import numpy as np
import cv2
import time
import torch

def tsdf_fusion(depth, cam_tform4x4_obj, cam_intr4x4, voxel_size=0.02, rgb=None, fpath_pcl=None, fpath_mesh=None, obs_weight=1.):
    """
    Args:
        depth (torch.Tensor): Kx1xHxW, depth values of 0. are flagged as invalid
        cam_tform4x4_obj (torch.Tensor): Kx4x4
        cam_intr4x4 (torch.Tensor): Kx4x4
        rgb (torch.Tensor): Kx3xHxW
        voxel_size (float)
        fpath_mesh (Path)
        fpath_pcl (Path)
    Returns:
        verts (torch.Tensor):
        faces (torch.Tensor):
        norms (torch.Tensor):
        colors (torch.Tensor):
    """
    device = depth.device
    dtype = depth.dtype
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    logger.info("Estimating voxel volume bounds...")
    K, _, H, W = depth.shape
    if rgb is None:
        rgb = torch.zeros((K, 3, H, W))
    #cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
    vol_bnds = np.zeros((3,2))
    for i in range(K):

        # Read depth image and camera pose

        depth_im = depth[i, 0].detach().cpu().numpy()

        # depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
        #depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        #depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
        # depth_im = depth[i].detach().cpu().numpy()

        # cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))  # 4x4 rigid transformation matrix
        cam_pose = cam_tform4x4_obj[i].detach().cpu().numpy()
        cam_intr = cam_intr4x4[i].detach().cpu().numpy()
        # Compute camera view frustum and extend convex hull
        view_frust_pts = get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
    # ======================================================================================================== #

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    logger.info("Initializing voxel volume...")
    tsdf_vol = TSDFVolume(vol_bnds, voxel_size=voxel_size)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i in range(K):
        logger.info("Fusing frame %d/%d"%(i+1, K))

        # Read RGB-D image and camera pose
        color_image = rgb[i].permute(1, 2, 0).detach().cpu().numpy()
        depth_im = depth[i, 0].detach().cpu().numpy()
        depth_im[depth_im == 0.] = -1. #  np.finfo(depth_im.dtype).max

        #color_image = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg"%(i)), cv2.COLOR_BGR2RGB)
        #depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
        #depth_im /= 1000.
        #depth_im[depth_im == 65.535] = 0
        #cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))

        cam_pose = cam_tform4x4_obj[i].detach().cpu().numpy()
        cam_intr = cam_intr4x4[i].detach().cpu().numpy()

        # Integrate observation into voxel volume (assume color aligned with depth)
        tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=obs_weight)

        fps = K / (time.time() - t0_elapse)
        logger.info("Average FPS: {:.2f}".format(fps))

    logger.info('extracting mesh from tsdf...')
    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    if fpath_mesh is not None:
        logger.info("Saving mesh to mesh.ply...")
        meshwrite(str(fpath_mesh), verts, faces, norms, colors)

    if fpath_pcl is not None:
        # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
        logger.info("Saving point cloud to pc.ply...")
        point_cloud = tsdf_vol.get_point_cloud()
        pcwrite(str(fpath_pcl), point_cloud)

    verts = torch.from_numpy(verts.copy()).to(device=device, dtype=dtype)
    faces = torch.from_numpy(faces.copy()).to(device=device, dtype=dtype)
    norms = torch.from_numpy(norms.copy()).to(device=device, dtype=dtype)
    colors = torch.from_numpy(colors.copy()).to(device=device, dtype=dtype)

    return verts, faces, norms, colors