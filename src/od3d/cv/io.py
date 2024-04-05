from pathlib import Path
from PIL import Image
from torchvision import transforms
from pytorch3d.io import load_ply
from pytorch3d.io import save_ply
import numpy as np
import torch
import wandb
import open3d as o3d
import numpy as np
from typing import List
import cv2

def get_default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_pts3d_colors(fpath: Path):
    pcd = o3d.io.read_point_cloud(str(fpath))
    return torch.from_numpy(np.asarray(pcd.colors)).to(torch.float)

def read_pts3d(fpath: Path):
    pcd = o3d.io.read_point_cloud(str(fpath))
    return torch.from_numpy(np.asarray(pcd.points)).to(torch.float)

def read_pts3d_with_colors_and_normals(fpath: Path, device='cpu'):
    pcd = o3d.io.read_point_cloud(str(fpath))
    pts3d = torch.from_numpy(np.asarray(pcd.points)).to(dtype=torch.float, device=device)
    pts3d_colors = torch.from_numpy(np.asarray(pcd.colors)).to(dtype=torch.float, device=device)
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pts3d_normals = torch.from_numpy(np.asarray(pcd.normals)).to(dtype=torch.float, device=device)
    return pts3d, pts3d_colors, pts3d_normals

def write_pts3d_with_colors(pts3d: torch.Tensor, pts3d_colors: torch.Tensor, fpath: Path):
    pcd = o3d.geometry.PointCloud()

    # Set the point cloud data
    pcd.points = o3d.utility.Vector3dVector(pts3d.detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(pts3d_colors.detach().cpu().numpy())

    o3d.io.write_point_cloud(filename=str(fpath), pointcloud=pcd)

def write_pts3d_with_colors_and_normals(pts3d: torch.Tensor, pts3d_colors: torch.Tensor, pts3d_normals: torch.Tensor, fpath: Path):
    fpath.parent.mkdir(parents=True, exist_ok=True)

    pcd = o3d.geometry.PointCloud()

    # Set the point cloud data
    pcd.points = o3d.utility.Vector3dVector(pts3d.detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(pts3d_colors.detach().cpu().numpy())
    pcd.normals = o3d.utility.Vector3dVector(pts3d_normals.detach().cpu().numpy())
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.io.write_point_cloud(filename=str(fpath), pointcloud=pcd)


def read_co3d_depth_image(path: Path):
    img = Image.open(path)

    img = (
        np.frombuffer(np.array(img, dtype=np.uint16), dtype=np.float16)
        .astype(np.float32)
        .reshape((img.size[1], img.size[0]))
    )
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    img = transform(img)
    return img

def read_depth_image(path: Path):
    depth = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH )
    depth = torch.from_numpy(depth / 1000.)[None,]
    return depth

def write_depth_image(img: torch.Tensor, path: Path):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    if img.dim() == 3:
        depth = img[0]
    else:
        depth = img
    depth = depth.clone().detach().cpu()
    depth = depth.clamp(0, 65535.)
    depth = depth * 1000
    cv2.imwrite(str(path), depth.detach().cpu().numpy().astype(np.uint16))


def write_mask_image(img: torch.Tensor, path: Path):
    transform = transforms.Compose([
        transforms.ToPILImage()
    ])
    img = transform((img * 255).to(torch.uint8))

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def read_image(path: Path):
    img = Image.open(path)

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # Convert the PIL image to Torch tensor
    img = transform(img)
    return img

def image_as_wandb_image(img, caption="Caption Blub"):
    img = wandb.Image(
        img.permute(1, 2, 0).detach().cpu().numpy(),
        caption=caption
    )
    return img


def extract_frames_from_video(fpath_video: Path, path_frames: Path, fps=5):

    #vidcap.set(cv2.CAP_PROP_FPS, fps)
    # delete all files in directory path_frmaes
    from od3d.io import rm_dir
    rm_dir(path_frames)
    path_frames.mkdir(parents=True, exist_ok=True)


    vidcap = cv2.VideoCapture(str(fpath_video))
    vid_fps = vidcap.get(cv2.CAP_PROP_FPS)

    #if not path_frames.exists():

    success, image = vidcap.read()
    count_cap = 0
    count_store = 0
    while success:
        if count_cap % int(vid_fps / fps) == 0:
            count_store += 1
            cv2.imwrite(str(path_frames.joinpath(f"frame_{count_store}.jpg")), image)  # save frame as JPEG file
        #print('Read a new frame: ', success)
        success, image = vidcap.read()
        count_cap += 1
        cv2.waitKey(1)


def write_webm_videos_side_by_side(out_fpath: Path, in_fpaths=List[Path], W=1280,
                                   padding_size=10, padding_color=(0.89, 0.89, 0.89)):
    from moviepy.editor import VideoFileClip, clips_array, ColorClip

    # Load the webm videos
    video_clips = [VideoFileClip(str(fpath)) for fpath in in_fpaths]

    # Set the desired width and padding size

    # Resize videos to have the same height (keeping the aspect ratio)
    for i in range(len(video_clips)):
        video_clips[i] = video_clips[i].resize(height=(W / 2) * video_clips[i].size[1] / video_clips[i].size[0])
    #video1 = video1.resize(height=(final_width / 2) * video1.size[1] / video1.size[0])
    #video2 = video2.resize(height=(final_width / 2) * video2.size[1] / video2.size[0])

    # Create white-gray padding
    padding = ColorClip((padding_size, video_clips[0].h),
                        color=(padding_color[0] * 255, padding_color[1] * 255, padding_color[2] * 255)).set_duration(video_clips[0].duration)

    # Combine videos and padding side by side
    video_clips_with_pad = []#  = [ for video in video_clips]
    for i, video in enumerate(video_clips):
        video_clips_with_pad.append(video)
        if i < len(video_clips) - 1:
            video_clips_with_pad.append(padding)
    final_clip = clips_array([video_clips_with_pad])

    # Write the combined video to a file
    final_clip.write_videofile(str(out_fpath), codec="libvpx", bitrate="5000k")

    # Close the video clips
    for i in range(len(video_clips)):
        video_clips[i].close()
