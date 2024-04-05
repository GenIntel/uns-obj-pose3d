from pathlib import Path
import numpy as np
import torch

from od3d.io import run_cmd
from od3d.io import read_config_extern

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

STRIDE=1

SEQ="216_22800_47486"
SEQ="216_22827_48422"
SEQ="216_22841_48461"
SEQ="340_35306_64677"

SEQS = [
'206_21810_45890',
'106_12650_23736',
'206_21805_45881',
'216_22836_48452',
'216_22808_47498',
]

for SEQ in SEQS:
    OD3D_DIR=Path("/scratch/sommerl/repos/NeMo")
    DROID_SLAM_DIR=OD3D_DIR.joinpath("third_party/DROID-SLAM")
    CO3D_PREPROCESS_PATH=Path("/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess")

    IMAGE_DIR=Path(f"/misc/lmbraid19/sommerl/datasets/CO3D/car/{SEQ}/images")
    META_PATH=list(Path("/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/meta/frames/car").joinpath(SEQ).iterdir())[0]
    frame_meta_config = read_config_extern(META_PATH)
    fx = frame_meta_config.l_cam_intr4x4[0][0]
    fy = frame_meta_config.l_cam_intr4x4[1][1]
    cx = frame_meta_config.l_cam_intr4x4[0][2]
    cy = frame_meta_config.l_cam_intr4x4[1][2]

    WEIGHTS_PATH=OD3D_DIR.joinpath("third_party/weights/droid.pth")
    OUT_RPATH=Path("car").joinpath(SEQ)
    OUT_DROID_PATH=DROID_SLAM_DIR.joinpath("reconstructions", OUT_RPATH)
    OUT_CO3D_PREPROCESS_PATH=CO3D_PREPROCESS_PATH.joinpath("droid_slam", OUT_RPATH)

    # run droid_slam
    # run_cmd(cmd='echo "994.8076782226562 994.8076782226562 468.5 265.0" > calib.txt', logger=logger)
    # run_cmd(cmd='echo "332.63812255859375 332.63812255859375 460.5 260.5" > calib.txt', logger=logger)
    # run_cmd(cmd='echo "435.08123779296875 435.08123779296875 424.5 240.5" > calib.txt', logger=logger)
    # run_cmd(cmd=f'echo "1942.6253662109375 1941.634765625 1000.0 485.0" > calib.txt', logger=logger)
    run_cmd(cmd=f'echo "{fx} {fy} {cx} {cy}" > calib.txt', logger=logger)

    cmd=f'mkdir -p {OUT_CO3D_PREPROCESS_PATH} && mv calib.txt {OUT_CO3D_PREPROCESS_PATH} && cd {DROID_SLAM_DIR} && python demo.py --imagedir={IMAGE_DIR} --calib={OUT_CO3D_PREPROCESS_PATH}/calib.txt --reconstruction_path {OUT_RPATH} --weights {WEIGHTS_PATH} --stride {STRIDE} --disable_vis && mv {OUT_DROID_PATH}/* {OUT_CO3D_PREPROCESS_PATH}'
    run_cmd(cmd=cmd, logger=logger)

    logger.info(f'{OUT_CO3D_PREPROCESS_PATH}')

"""
# retrieve pointcloud
rec_path = CO3D_PREPROCESS_PATH.joinpath('droid_slam', 'car', SEQ)
images = np.load(rec_path.joinpath('images.npy')) # 74x3x304x632 (from 970x2000)
disps = np.load(rec_path.joinpath('disps.npy'))   # 74x304x632
poses = np.load(rec_path.joinpath('poses.npy'))   # 74x7
intrinsics = np.load(rec_path.joinpath('intrinsics.npy'))   # 74x4 (all the same for rescaled)
tstamps = np.load(rec_path.joinpath('tstamps.npy'))   # 74x4 (all the same for rescaled)

images = torch.from_numpy(images)
disps = torch.from_numpy(disps)
poses = torch.from_numpy(poses)
intrinsics = torch.from_numpy(intrinsics)
tstamps = torch.from_numpy(tstamps)

depths = 1./disps
from od3d.cv.geometry.transform import depth2pts3d_grid, cam_intr_4_to_4x4, transf4x4_from_rot3x3_and_transl3, transf3d_broadcast, inv_tform4x4

from pytorch3d.transforms import axis_angle_to_matrix, quaternion_to_axis_angle, quaternion_to_matrix

cam_intr4x4 = cam_intr_4_to_4x4(cam_intr4=intrinsics)

device = cam_intr4x4.device
dtype = cam_intr4x4.dtype
H, W = depths.shape[-2:]
pxl2d = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'), dim=-1).to(device=device,
                                                                                                dtype=dtype)
cams_tform_pts3d = depth2pts3d_grid(depth=depths, cam_intr4x4=cam_intr4x4)
#quats = torch.stack([poses[:, 6], poses[:, 3], poses[:, 4], poses[:, 5]], dim=1)

# export CUDA_HOME=/scratch/sommerl/cudas/cuda-11.7
# pip install git+https://github.com/princeton-vl/lietorch.git
# pip install git+https://github.com/princeton-vl/DROID-SLAM.git
from lietorch import SO3
## rots = SO3.InitFromVec(quats).matrix()
cams_tform4x4_cam1 = transf4x4_from_rot3x3_and_transl3(rot3x3=SO3.InitFromVec(poses[:, 3:7]).matrix()[:, :3, :3], transl3=poses[:, :3])
cam1_tform4x4_cams = inv_tform4x4(cams_tform4x4_cam1)
cam1_tform_pts3d = transf3d_broadcast(transf4x4=cam1_tform4x4_cams[:, None, None], pts3d=cams_tform_pts3d.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

# ((count >= 2)
masks = (disps > .5*disps.mean(dim=[1,2], keepdim=True))
cam1_tform_pts3d = cam1_tform_pts3d.permute(0, 2, 3, 1)[masks]
cam1_tform_pts3d = cam1_tform_pts3d[::100, :]

from od3d.cv.visual.show import show_pcl_via_open3d
show_pcl_via_open3d(pts3d=cam1_tform_pts3d.reshape(-1, 3))

# estimate ground plane

# filter poincloud with groundplane connected samples (or try normalized cut on distances first)

# --image_size
# calib.txt  disps.npy  images.npy  intrinsics.npy  poses.npy  tstamps.npy
#  h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
#  w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
#  image = cv2.resize(image, (w1, h1))
#  image = image[:h1-h1%8, :w1-w1%8]


"""