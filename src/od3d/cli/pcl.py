import logging
logger = logging.getLogger(__name__)
import typer
from od3d.cv.io import load_ply
from od3d.cv.geometry.mesh import Meshes
from od3d.cv.visual.show import show_pcl, show_open3d_pcl
import od3d.cv.visual.show as show
app = typer.Typer()

@app.command()
def show_shapenet():
    from pytorch3d.renderer import (
        OpenGLPerspectiveCameras,
        PointLights,
        RasterizationSettings,
        TexturesVertex,
        look_at_view_transform,
    )
    import torch

    from pytorch3d.structures import Meshes

    device = 'cuda'
    # Rendering settings.
    R, T = look_at_view_transform(1.0, 1.0, 90)
    cameras = OpenGLPerspectiveCameras(R=R, T=T, device=device)
    raster_settings = RasterizationSettings(image_size=512)
    lights = PointLights(location=torch.tensor([0.0, 1.0, -2.0], device=device)[None], device=device)

    Meshes()

@app.command()
def show_mesh():
    logging.basicConfig(level=logging.INFO)

    from od3d.cv.geometry.mesh import Meshes

    sequence_name = '354_37645_70054'  #
    # co3dv2:  354_37645_70054 372_41153_81941 270_28771_56661 136_15656_31168 373_41715_83384
    fpaths_meshes = [
        #'/misc/lmbraid19/sommerl/datasets/PASCAL3D_Preprocess/mesh/cuboid250/bicycle/mesh.ply',
        f'/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/mesh/cuboid500/meta_mask/meta/bicycle/{sequence_name}/mesh.ply',
        f'/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/mesh/alphawrap500/meta_mask/meta/bicycle/{sequence_name}/mesh.ply',

        #'/misc/lmbraid19/sommerl/datasets/CO3Dv1_Preprocess/mesh/alpha500/meta_mask/meta/bicycle/397_49943_98337/mesh.ply'

    ]
    fpaths_meshes_tform_obj = [
        #None,
        f'/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/tform_obj/label3d_cuboid/meta_mask/meta/bicycle/{sequence_name}/tform_obj.pt',
        f'/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/tform_obj/label3d_cuboid/meta_mask/meta/bicycle/{sequence_name}/tform_obj.pt',

        #'/misc/lmbraid19/sommerl/datasets/CO3Dv1_Preprocess/tform_obj/label3d_zsp_cuboid/meta_mask/meta/bicycle/397_49943_98337/tform_obj.pt'
        #
    ]

    logger.info(fpaths_meshes_tform_obj)
    meshes = Meshes.load_from_files(fpaths_meshes=fpaths_meshes, fpaths_meshes_tforms=fpaths_meshes_tform_obj)

    import torch
    tform_obj = torch.load(f'/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/tform_obj/label3d_cuboid/meta_mask/meta/bicycle/{sequence_name}/tform_obj.pt')
    tform_obj = torch.load(f'/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/tform_obj/label3d/meta_mask/meta/bicycle/{sequence_name}/tform_obj.pt')

    # fpath_pcl = f'/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/pcl/meta_mask/meta/bicycle/{sequence_name}/pcl.ply'
    # from od3d.cv.io import read_pts3d_with_colors_and_normals
    # pts3d, pts3d_colors, pts3d_normals = read_pts3d_with_colors_and_normals(fpath_pcl)
    # from od3d.cv.geometry.transform import transf3d_broadcast
    # pts3d = transf3d_broadcast(pts3d=pts3d, transf4x4=tform_obj)
    # , pts3d = [pts3d], pts3d_colors = [pts3d_colors]

    show.show_scene(meshes=meshes, meshes_colors=meshes.get_verts_ncds_cat_with_mesh_ids())
    # from pathlib import Path
    # fpath = Path('/home/sommerl/Downloads/not_watertight_mesh.ply')
    # verts, faces = load_ply(fpath)
    # print(verts.shape, faces.shape)
@app.command()
def show_pts(fpath: str = typer.Option(None, '-f', '--fpath'), device: str = typer.Option('cpu', '-d', '--device')):
    from pathlib import Path
    from od3d.cv.io import read_pts3d, read_pts3d_colors, read_pts3d_with_colors_and_normals

    # fpath = Path('/misc/lmbraid19/sommerl/datasets/MonoLMB_Preprocess/droid_slam/elephant/24_01_29__18_10/pcl_clean.ply')

    pts3d, pts3d_colors, pts3d_normals = read_pts3d_with_colors_and_normals(fpath)

    import open3d
    from od3d.cv.geometry.downsample import random_sampling
    import torch

    o3d_pcl = open3d.geometry.PointCloud()
    o3d_pcl.points = open3d.utility.Vector3dVector(pts3d.detach().cpu().numpy())
    o3d_pcl.normals = open3d.utility.Vector3dVector(pts3d_normals.detach().cpu().numpy())  # invalidate existing normals
    o3d_pcl.colors = open3d.utility.Vector3dVector(pts3d_colors.detach().cpu().numpy())
    print(pts3d.shape)
    pts3d = random_sampling(pts3d, pts3d_max_count=10000)  # 11 GB
    print(pts3d.shape)

    quantile = max(0.01, 3. / len(pts3d))
    particle_size = torch.cdist(pts3d[None,], pts3d[None,]).quantile(dim=-1, q=quantile).mean()
    alpha = particle_size / 2. * 2
    print(quantile, particle_size, alpha)

    o3d_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3d_pcl, alpha)
    show.show_scene(pts3d=[pts3d], pts3d_colors=[pts3d_colors], pts3d_normals=[pts3d_normals])


@app.command()
def show_scene():
    fpath = '/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned/latest_50s_to_5s_mesh/r0/mesh/car/mesh.ply'
    fpath = '/misc/lmbraid19/sommerl/datasets/ShapeNetCore.v2/02691156/885b7ba506c9d68ab6ecbbb2e4d05900/models/model_normalized.obj'
    fpath = '/misc/lmbraid19/sommerl/datasets/ShapeNetCore.v2/03325088/ad51249f5960d841c36fb70296e45483/models/model_normalized.obj'
    fpath = '/misc/lmbraid19/sommerl/datasets/ShapeNetCore.v2/02958343/adc6f0f3274cf92cd4f6529a209c5dc0/models/model_normalized.obj'
    fpath = '/misc/lmbraid19/sommerl/datasets/ShapeNetCore.v2/02958343/bac6953b6866ec02300856337cd5b2e/models/model_normalized.obj'
    fpath = 'not_watertight_mesh.ply'

    # 02958343 : car
    meshes = Meshes.load_from_files([fpath])

    # meshes.feats = meshes.rgb
    show.show_scene(meshes=meshes, meshes_colors=meshes.rgb)

    # import open3d as o3d
    #
    # def visualize(mesh):
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window()
    #     vis.add_geometry(mesh)
    #     vis.run()
    #     vis.destroy_window()
    #
    # def main():
    #     mesh = o3d.io.read_triangle_model(fpath) # read_triangle_mesh read_triangle_model
    #
    #     o3d.visualization.draw_geometries([mesh])
    #     # visualize(mesh)
    #
    # main()



# @app.command()
# def show(fpath: str = typer.Option(None, '-f', '--fpath'), device: str = typer.Option('cpu', '-d', '--device')):
#     fpath = "/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/pcls/206_21810_45890/cuboid_max_1000.ply"
#     verts, faces = load_ply(fpath)
#
#     show_pcl(verts.to(device=device))
#
@app.command()
def show_open3d():
    import open3d as o3d
    fpath = '/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned/latest_50s_to_5s_mesh/r0/mesh/car/mesh.ply'
    fpath = '/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned/all_20s_to_5s_mesh/r0/mesh/car/mesh.ply'

    from od3d.cv.io import read_pts3d
    pts3d = read_pts3d(fpath)
    from kaolin.ops.conversions import pointclouds_to_voxelgrids, voxelgrids_to_trianglemeshes

    # BxNx3
    mesh = voxelgrids_to_trianglemeshes(pointclouds_to_voxelgrids(pointclouds=pts3d[None,].cuda(), resolution=256))
    #print(pcd)
    print(mesh.is_watertight())
    print(mesh)

    o3d.visualization.draw_geometries([mesh])