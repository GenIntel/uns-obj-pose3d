import logging
import random
logger = logging.getLogger(__name__)
import typer
import od3d.io

app = typer.Typer()

from od3d.datasets.co3d import CO3D
from od3d.cv.visual.show import show_scene
import torch
from od3d.cv.visual.show import imgs_to_img

from od3d.cv.geometry.transform import transf3d_broadcast, tform4x4_from_transl3d, tform4x4, get_spherical_uniform_tform4x4
from od3d.datasets.co3d.enum import PCL_SOURCES, CUBOID_SOURCES, CAM_TFORM_OBJ_SOURCES
from od3d.cv.geometry.mesh import Meshes, Mesh

from od3d.cv.visual.show import show_imgs, get_img_from_plot, show_img
from od3d.cv.visual.crop import crop_white_border_from_img
import matplotlib.pyplot as plt
from od3d.cv.visual.resize import resize
from od3d.datasets.pascal3d.dataset import Pascal3D
from od3d.datasets.objectnet3d.dataset import ObjectNet3D

from od3d.methods.nemo import NeMo
from od3d.methods.zsp import ZSP
from pathlib import Path

from omegaconf.omegaconf import OmegaConf  # bottle, suitcase, cup, airplane, microwave, tv, train, hairdryer, remote
from od3d.cv.transforms.transform import OD3D_Transform
import torch.utils.data
from od3d.cv.geometry.transform import inv_tform4x4
from tqdm import tqdm
import open3d

WINDOW_WIDTH = 1980
WINDOW_HEIGHT = 1080

@app.command()
def feature_vectors(count: int = typer.Option(1, '-c', '--count')):
    from od3d.cv.visual.draw import random_colors_as_img
    from od3d.cv.visual.show import show_img
    for i in range(count):
        img = random_colors_as_img(6)
        show_img(img, fpath=f'feature_vector_{i}.png')
@app.command()
def multiple(benchmark: str = typer.Option('co3d_nemo_align3d', '-b', '--benchmark'),
             ablation: str = typer.Option(None, '-a', '--ablation'),
             platform: str = typer.Option(None, '-p', '--platform'),
             age_in_hours_gt: int = typer.Option(0, '-g', '--age-in-hours-gt'),
             age_in_hours_lt: int = typer.Option(1000, '-l', '--age-in-hours-lt'),
             metrics: str = typer.Option(None, '-m', '--metrics'),
             x_label: str = typer.Option(None, '-x', '--x-label'),
             y_label: str = typer.Option(None, '-y', '--y-label'),
             configs: str = typer.Option(None, '-c', '--configs'),):

    logging.basicConfig(level=logging.INFO)

    from od3d.cli.benchmark import get_dataframe_multiple

    if metrics is not None:
        metrics = metrics.split(',')
    else:
        metrics = []

    if configs is not None:
        configs = configs.split(',')
    else:
        configs = []

    df = get_dataframe_multiple(benchmark=benchmark, ablation=ablation, platform=platform, age_in_hours_gt=age_in_hours_gt, age_in_hours_lt=age_in_hours_lt, metrics=metrics, configs=configs)

    for metric in metrics:
        x = df[configs[0]].to_numpy()
        y = df[configs[1]].to_numpy()
        import numpy as np

        x_log = torch.from_numpy(x)  # torch.log(torch.from_numpy(x)) / torch.log(torch.Tensor([10])).numpy()
        y_log = torch.from_numpy(y)  # torch.log(torch.from_numpy(y)) / torch.log(torch.Tensor([10])).numpy()
        xi = np.sort(np.unique(x))
        yi = np.sort(np.unique(y))
        grid_xiyi = np.stack(np.meshgrid(xi, yi))
        logger.info(grid_xiyi)
        mask_grid_xiyi = (x[:, None] == grid_xiyi[0].flatten()[None, :]) * (
                    y[:, None] == grid_xiyi[1].flatten()[None, :])
        xi_log = np.sort(np.unique(x_log))  # np.arange(x.min(), x.max(), 0.01)
        yi_log = np.sort(np.unique(y_log))

        z = df[metric].to_numpy()  # 'pose/acc_pi6' 'pose/acc_pi18'
        z_max = (z[:, None] * mask_grid_xiyi).max(axis=0).reshape(grid_xiyi.shape[1:])
        x_max = x[(z[:, None] * mask_grid_xiyi).argmax(axis=0)].reshape(grid_xiyi.shape[1:])
        y_max = y[(z[:, None] * mask_grid_xiyi).argmax(axis=0)].reshape(grid_xiyi.shape[1:])
        xi_unique = xi.copy()
        yi_unique = yi.copy()
        xi, yi = np.meshgrid(xi, yi)
        xi_log, yi_log = np.meshgrid(xi_log, yi_log)

        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib
        # matplotlib.use("TkAgg")

        fig, ax = plt.subplots(1, 1, figsize=(len(xi_unique), len(yi_unique)))  # (subplot_kw={"projection": "3d"})

        aspect_ratio = 0.5

        font_size = 16
        # Create a heatmap using imshow
        im = ax.imshow(z_max, cmap='viridis', interpolation='nearest', aspect='auto')  # 'auto'
        ax.tick_params(axis='y', labelsize=font_size)
        ax.tick_params(axis='x', labelsize=font_size)

        if x_label is None:
            x_label = configs[0]
        if y_label is None:
            y_label = configs[1]
        ax.set_xlabel(x_label, fontsize=font_size)
        ax.set_ylabel(y_label, fontsize=font_size)

        ax.set(xticks=np.arange(z_max.shape[1]),
                  xticklabels=xi_unique)  # np.round(np.linspace(xi_log.min(), xi_log.max(), z_max.shape[1]), decimals=1))
        ax.set(yticks=np.arange(z_max.shape[0]),
                  yticklabels=yi_unique)  # np.round(np.linspace(yi_log.min(), yi_log.max(), z_max.shape[0]), decimals=0))

        # Add colorbar to the right of the plot
        cbar = fig.colorbar(im, ax=ax)  # , shrink='auto')
        cbar.ax.tick_params(labelsize=font_size)

        # Add colorbar to the right of the plot
        cbar.ax.tick_params(labelsize=font_size)

        max_coordinates = np.unravel_index(z_max.argmax(), z_max.shape)
        max_x = xi_log[max_coordinates]
        max_y = yi_log[max_coordinates]
        max_z = z_max[max_coordinates]

        max_coordinates = (max_coordinates[1], max_coordinates[0])

        # Plot the point using scatter
        ax.scatter(*max_coordinates, color='red', marker='o', label='max')

        # Annotate the point with a description
        desc = f'{max_z * 100:.1f}'
        ax.annotate(desc, max_coordinates, textcoords="offset points", xytext=(0, 10), ha='center',
                       fontsize=font_size,
                       color='red')

        plt.tight_layout()
        plt.savefig(f'ablation_{metric.replace(",", "_").replace("/", "_")}_{ablation.replace(",", "_").replace("/", "_")}.svg')
    logger.info(df)

@app.command()
def mesh():
    logging.basicConfig(level=logging.INFO)
    device = 'cuda'
    dtype = torch.float # co3dv1_10s_zsp_labeled co3d_10s_zsp_unlabeled
    co3d = CO3D.create_by_name('co3d_10s_zsp_unlabeled', config={'categories': ['bicycle']}) #co3d_5s_no_zsp_labeled 'co3d_50s_no_zsp_aligned' 'co3dv1_10s_zsp_aligned' 'co3d_10s_zsp_aligned' 'co3dv1_10s_zsp_unlabeled'
    categories = co3d.categories
    sequences = co3d.get_sequences()
    sequences_unique_names = [seq.name_unique for seq in sequences]
    instances_count = len(sequences)
    map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in sequences_unique_names])
    instance_ids = torch.LongTensor(list(range(instances_count)))

    # while True:
    #     rand_category_rand_instance_id = random.sample(instance_ids.tolist(), k=1)[0]
    #     logger.info(f'chosen ids are {rand_category_rand_instance_id}')
    #     seq1 = sequences[rand_category_rand_instance_id]
    #     show_scene(pts3d=[seq1.get_pcl(pcl_source=PCL_SOURCES.CO3D)],
    #                pts3d_colors=[seq1.get_pcl_colors(pcl_source=PCL_SOURCES.CO3D)])

    rand_category_rand_instance_id = 9  # 344 349 337 334 322 324
    viewpoint_id = 2
    logger.info(f'chosen ids are {rand_category_rand_instance_id}')
    mesh_source = CUBOID_SOURCES.DEFAULT
    seq1 = sequences[rand_category_rand_instance_id]
    mesh1 = seq1.get_mesh(mesh_source=mesh_source, add_rgb_from_pca=False, device=device)
    cams_tform4x4_world, cams_intr4x4, cams_imgs = seq1.get_cams(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D, cams_count=5, show_imgs=True)
    tform = torch.Tensor([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 0.], [0., 0., 0., 1.]]).to(device=device)
    pts3d_raw = transf3d_broadcast(seq1.get_pcl(pcl_source=PCL_SOURCES.CO3D).to(device=device), transf4x4=tform)
    img_pcl_raw = show_scene(pts3d=[pts3d_raw],
               pts3d_colors=[seq1.get_pcl_colors(pcl_source=PCL_SOURCES.CO3D)],
               #cams_tform4x4_world=cams_tform4x4_world, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs,
               return_visualization=True, viewpoints_count=viewpoint_id+2, meshes_as_wireframe=True)[viewpoint_id]
    # show_img(img_pcl_raw)


    cams_tform4x4_world, cams_intr4x4, cams_imgs = seq1.get_cams(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D, cams_count=5, show_imgs=True)
    pts3d_clean = transf3d_broadcast(seq1.get_pcl(pcl_source=PCL_SOURCES.CO3D_CLEAN.value).to(device=device), transf4x4=tform)

    img_pcl_clean = show_scene(pts3d=[pts3d_clean],
               pts3d_colors=[seq1.get_pcl_colors(pcl_source=PCL_SOURCES.CO3D_CLEAN.value)],
               #cams_tform4x4_world=cams_tform4x4_world, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs,
               return_visualization=True, viewpoints_count=viewpoint_id+2, meshes_as_wireframe=True)[viewpoint_id]

    mesh_vertices_count = 500
    pts3d = pts3d_clean # seq1.get_pcl(pcl_source=PCL_SOURCES.CO3D_CLEAN.value)
    o3d_pcl = open3d.geometry.PointCloud()
    o3d_pcl.points = open3d.utility.Vector3dVector(pts3d.detach().cpu().numpy())
    from od3d.cv.geometry.downsample import random_sampling
    pts3d = random_sampling(pts3d, pts3d_max_count=20000)

    particle_size = torch.cdist(pts3d[None,], pts3d[None,]).quantile(dim=-1, q=5. / len(pts3d)).mean()
    alpha = 10 * particle_size
    o3d_obj_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(o3d_pcl, alpha)
    logger.info(o3d_obj_mesh)
    o3d_obj_mesh = o3d_obj_mesh.remove_unreferenced_vertices()
    logger.info(o3d_obj_mesh)
    obj_mesh = Mesh.from_o3d(o3d_obj_mesh, device=device)
    obj_mesh = Meshes.load_from_meshes(meshes=[obj_mesh])
    #obj_mesh.rgb = obj_mesh.get_verts_ncds_cat_with_mesh_ids()

    cams_tform4x4_world, cams_intr4x4, cams_imgs = seq1.get_cams(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D, cams_count=5, show_imgs=True)
    img_mesh_fine = show_scene(meshes=obj_mesh, pts3d=[pts3d_clean],
               pts3d_colors=[seq1.get_pcl_colors(pcl_source=PCL_SOURCES.CO3D_CLEAN.value)],
               return_visualization=True, viewpoints_count=viewpoint_id+2, meshes_as_wireframe=True)[viewpoint_id]
                # ,cams_tform4x4_world=cams_tform4x4_world, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs)

    o3d_obj_mesh = o3d_obj_mesh.simplify_quadric_decimation(mesh_vertices_count)
    logger.info(o3d_obj_mesh)
    obj_mesh = Mesh.from_o3d(o3d_obj_mesh, device=device)
    obj_mesh = Meshes.load_from_meshes(meshes=[obj_mesh])
    #obj_mesh.rgb = obj_mesh.get_verts_ncds_cat_with_mesh_ids()
    cams_tform4x4_world, cams_intr4x4, cams_imgs = seq1.get_cams(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D, cams_count=5, show_imgs=True)
    img_mesh_coarse = show_scene(meshes=obj_mesh, pts3d=[pts3d_clean],
               pts3d_colors=[seq1.get_pcl_colors(pcl_source=PCL_SOURCES.CO3D_CLEAN.value)],
               return_visualization=True, viewpoints_count=viewpoint_id+2, meshes_as_wireframe=True)[viewpoint_id]
               #  , cams_tform4x4_world=cams_tform4x4_world, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs)

    img_pcl_raw = crop_white_border_from_img(img_pcl_raw)
    img_pcl_clean = crop_white_border_from_img(img_pcl_clean)
    img_mesh_fine = crop_white_border_from_img(img_mesh_fine)
    img_mesh_coarse = crop_white_border_from_img(img_mesh_coarse)
    # img_pcl_raw, img_pcl_clean, img_mesh_fine, img_mesh_coarse

    H, W = img_pcl_raw.shape[1:]
    img_pcl_clean = resize(img_pcl_clean, scale_factor=H / img_pcl_clean.shape[-2])
    img_mesh_fine = resize(img_mesh_fine, scale_factor=H / img_mesh_fine.shape[-2])
    img_mesh_coarse = resize(img_mesh_coarse, scale_factor=H / img_mesh_coarse.shape[-2])

    img = torch.cat([img_pcl_raw, img_pcl_clean, img_mesh_fine, img_mesh_coarse], dim=-1)
    show_img(rgb=img, fpath='mesh_extraction.png')
    show_img(rgb=img)


@app.command()
def videos_filtering():
    logging.basicConfig(level=logging.INFO)
    device = 'cuda'
    dtype = torch.float # co3dv1_10s_zsp_labeled co3d_10s_zsp_unlabeled
    co3d = CO3D.create_by_name('co3d_no_zsp', config={'categories': ['car'],
                                                      'sequences_require_good_cam_movement': False,
                                                      'sequences_require_gt_pose': False,
                                                      'sequences_require_mesh': False,}) #co3d_5s_no_zsp_labeled 'co3d_50s_no_zsp_aligned' 'co3dv1_10s_zsp_aligned' 'co3d_10s_zsp_aligned' 'co3dv1_10s_zsp_unlabeled'
    categories = co3d.categories
    sequences = co3d.get_sequences()
    sequences_unique_names = [seq.name_unique for seq in sequences]
    instances_count = len(sequences)
    map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in sequences_unique_names])
    instance_ids = torch.LongTensor(list(range(instances_count)))

    while True:
        rand_category_rand_instance_id = random.sample(instance_ids.tolist(), k=1)[0]
        # rand_category_rand_instance_id = 49
        logger.info(f'chosen ids are {rand_category_rand_instance_id}')
        seq1 = sequences[rand_category_rand_instance_id]
        cams_tform4x4_world, cams_intr4x4, cams_imgs = seq1.get_cams(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D,
                                                                     cams_count=5, show_imgs=True)

        if seq1.viewpoint_coverage < 0.10 or seq1.centered_accuracy < 0.8 or seq1.mask_coverage < 0.05:
            if seq1.viewpoint_coverage < 0.10:
                # category: bicycle, id: 49
                # category: car, id: 3
                logger.info('no viewpoint variance')

            if seq1.centered_accuracy < 0.8:
                # category: car, id: 39, 3
                logger.info('not centered')

            if seq1.mask_coverage < 0.05:
                logger.info('not enough mask coverage')

            show_scene(pts3d=[seq1.get_pcl(pcl_source=PCL_SOURCES.CO3D)],
                       pts3d_colors=[seq1.get_pcl_colors(pcl_source=PCL_SOURCES.CO3D.value)],
                       cams_tform4x4_world=cams_tform4x4_world, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs)


@app.command()
def viewpoints():
    logging.basicConfig(level=logging.INFO)

    bunny_data = open3d.data.BunnyMesh()
    bunny_mesh_open3d = open3d.io.read_triangle_mesh(bunny_data.path)
    bunny_mesh = Meshes.load_from_meshes([Mesh.from_o3d(bunny_mesh_open3d)])
    bunny_rot = torch.eye(4)
    bunny_rot = torch.Tensor(
        [[0., 0., 1., 0.,],
         [1., 0., 0., 0.,],
         [0., 1., 0., 0.,],
         [0., 0., 0., 1.,]])

    bunny_mesh.verts.data = transf3d_broadcast(pts3d=bunny_mesh.verts, transf4x4=bunny_rot)
    bunny_mesh.rgb = bunny_mesh.get_verts_ncds_cat_with_mesh_ids()
    # 'bottle', 'train'
    # 'bottle', 'suitcase', 'cup', 'microwave', 'tv', 'train', 'hairdryer', 'remote'
    categories = ['bottle', 'train', 'airplane']
    for category in categories:
        logger.info(f'category: {category}')
        config = {'categories': [category]}
        #dataset = ObjectNet3D.create_by_name('objectnet3d_test', config=config)
        dataset = Pascal3D.create_by_name('pascal3d_test', config=config)

        dataset.transform = OD3D_Transform.create_by_name('scale_mask_separate_centerzoom512')
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=False, collate_fn=dataset.collate_fn, num_workers=4)
        logging.info(f"Dataset contains {len(dataset)} frames.")
        all_vpts = []
        all_cams = []
        cam_intr4x4 = None
        for i, batch in tqdm(enumerate(dataloader)):
            if torch.cuda.is_available():
                batch.to(device='cuda:0')

            cam_intr4x4 = batch.cam_intr4x4[0]
            all_cams.append(batch.cam_tform4x4_obj)

            vpts = inv_tform4x4(batch.cam_tform4x4_obj)[:, :3, 3]
            all_vpts.append(vpts)

        all_vpts = torch.cat(all_vpts, dim=0)
        all_cams = torch.cat(all_cams, dim=0)
        all_vpts = torch.nn.functional.normalize(all_vpts, dim=-1)
        all_cams[:, :3, 3] = torch.nn.functional.normalize(all_cams[:, :3, 3], dim=-1)

        #img = show_scene(pts3d=[all_vpts], return_visualization=True, viewpoints_count=1, crop_white_border=True)
        imgs = show_scene(meshes=[bunny_mesh], cams_intr4x4=cam_intr4x4, cams_tform4x4_world=all_cams, return_visualization=True, viewpoints_count=3, crop_white_border=True, cams_imgs_depth_scale=0.05)
        img = imgs[1] #  img.permute(1, 2, 0, 3).flatten(2)
        #img = torch.cat(img, dim=-1)
        img = crop_white_border_from_img(img, white_pad=20)
        show_img(img, height=1080, width=1980, fpath=f'{dataset.name}_azimuth_{category}.png')


@app.command()
def align3d():
    logging.basicConfig(level=logging.INFO)
    device = 'cuda'
    dtype = torch.float
    co3d = CO3D.create_by_name('co3d_5s_no_zsp_labeled') #co3d_5s_no_zsp_labeled 'co3d_50s_no_zsp_aligned' 'co3dv1_10s_zsp_aligned' 'co3d_10s_zsp_aligned' 'co3dv1_10s_zsp_unlabeled'
    categories = co3d.categories
    sequences = co3d.get_sequences()
    sequences_unique_names = [seq.name_unique for seq in sequences]
    instances_count = len(sequences)
    map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in sequences_unique_names])
    instance_ids = torch.LongTensor(list(range(instances_count)))

    category = 'chair'
    category = 'bicycle'

    rand_category_id = categories.index(category) # 'car', 'chair',
    rand_category_instance_ids = instance_ids[map_seq_to_cat == rand_category_id]

    # # CALCULATING CATEGORICAL PCA
    # category_instance_ids = instance_ids[map_seq_to_cat == rand_category_id]
    # categorical_features = []
    # for instance_id_in_category, instance_id in enumerate(category_instance_ids):
    #     instance_feats = sequences[instance_id].feats
    #
    #     if isinstance(instance_feats, List):
    #         categorical_features += torch.cat([vert_feats for vert_feats in instance_feats], dim=0)
    #     else:
    #         categorical_features.append(instance_feats)
    # categorical_features = torch.stack(categorical_features, dim=0)
    # _, _, categorical_pca_V = torch.pca_lowrank(categorical_features)
    # sequences[category_instance_ids[0]].categorical_pca_V = categorical_pca_V

    while True:
        rand_category_rand_instance_ids = random.sample(rand_category_instance_ids.tolist(), k=2)

        rand_category_rand_instance_ids = [3, 4]#  344 349 337 334 322 324
        logger.info(f'chosen category is {category}')
        logger.info(f'chosen ids are {rand_category_rand_instance_ids}')

        mesh_source = CUBOID_SOURCES.DEFAULT
        import math
        #uniform_objs_tform_obj = get_spherical_uniform_tform4x4(azim_min = -math.pi / 2 , azim_max= + math.pi / 2, azim_steps=5,
        #                                                        elev_min=-math.pi / 2, elev_max=-math.pi / 2,
        #                                                        elev_steps=1, theta_min=0., theta_max=0., theta_steps=1).to(device=device)
        P = 3

        seq1 = sequences[rand_category_rand_instance_ids[0]]
        seq2 = sequences[rand_category_rand_instance_ids[1]]
        mesh1 = seq1.get_mesh(mesh_source=mesh_source, add_rgb_from_pca=True, device=device)
        mesh2 = seq2.get_mesh(mesh_source=mesh_source, add_rgb_from_pca=True, device=device)

        pts1 = mesh1.verts.to(device=device)
        pts2 = mesh2.verts.to(device=device)
        dist_2_1 = seq2.get_dist_verts_mesh_feats_to_other_sequence(seq1).to(device=device)
        dist_2_1 = dist_2_1 / 2.
        from od3d.cv.geometry.fit.tform4x4 import score_tform4x4_fit, fit_tform4x4

        N = len(dist_2_1)
        fits_count = P
        fit_pts_count = 4
        pts_sample_probs = torch.ones(size=(fits_count, N)).to(device=device)
        pts_ids = torch.multinomial(pts_sample_probs.view(-1, N), num_samples=fit_pts_count).view(
            (fits_count, fit_pts_count))
        pts1_ids = dist_2_1.argmin(dim=-1)[pts_ids]

        # sequences[rand_category_rand_instance_ids[0]]
        uniform_objs_tform_obj = fit_tform4x4(pts=pts2, pts_ref=pts1, pts_ids=pts_ids, dist_ref=dist_2_1)

        dist_appear_weight = 0.1
        _, proposal_dist_ref_geo_avg, proposal_dist_ref_appear_avg, proposal_dist_src_ref_2d_ids, proposal_dist_src_ref_weights = \
            score_tform4x4_fit(pts=pts2, pts_ref=pts1, tform4x4=uniform_objs_tform_obj, dist_app_ref=dist_2_1, return_dists=True, return_weights=True, cyclic_weight_temp=0.7, dist_app_weight=dist_appear_weight)


        proposal_dist_ref_geo_avg = proposal_dist_ref_geo_avg * (1. - dist_appear_weight)
        proposal_dist_ref_appear_avg = proposal_dist_ref_appear_avg * dist_appear_weight

        from od3d.cv.select import batched_index_fill
        #mesh1_weights = torch.zeros_like(mesh1.rgb[:, :1])
        #mesh1_weights = batched_index_fill(input=mesh1_weights.permute(1, 0).repeat(3, 1), value=proposal_dist_src_ref_weights,  index=proposal_dist_src_ref_2d_ids[:, :, 1]).permute(1, 0)
        #mesh1_weights = mesh1_weights.mean(dim=1, keepdim=True)
        #mesh1_weights = mesh1_weights / mesh1_weights.max(dim=0, keepdim=True).values
        #mesh1.rgb *= mesh1_weights

        mesh2_weights = torch.zeros_like(mesh2.rgb[:, :1])
        mesh2_weights = batched_index_fill(input=mesh2_weights.permute(1, 0).repeat(P, 1), value=proposal_dist_src_ref_weights,  index=proposal_dist_src_ref_2d_ids[:, :, 0]).permute(1, 0)
        mesh2_weights = mesh2_weights / mesh2_weights.max(dim=0, keepdim=True).values

        meshes = [mesh1]
        lines = []
        for t in range(P):
            uniform_obj_tform_obj = uniform_objs_tform_obj[t]
            uniform_obj_tform_obj[:3, 3] = 0.
            uniform_obj_tform_obj[0, 3] = 1. # shift in x
            uniform_obj_tform_obj[2, 3] = -1. * (t - P // 2) # shift in z
            mesh2_verts = transf3d_broadcast(pts3d=mesh2.verts.to(device=device, dtype=dtype), transf4x4=uniform_obj_tform_obj.to(device=device))

            #uniform_obj_tform_obj = torch.eye(4).to(device=device)
            uniform_obj_tform_obj[:3, 3] = 0.
            uniform_obj_tform_obj[0, 3] = 1.5  # shift in x
            uniform_obj_tform_obj[2, 3] = -1. * (t - P // 2)  # shift in z
            mesh2_weights_verts = transf3d_broadcast(pts3d=mesh2.verts.to(device=device, dtype=dtype), transf4x4=uniform_obj_tform_obj.to(device=device))
            mesh2_weights_rgb = torch.ones_like(mesh2.rgb)
            mesh2_weights_rgb *= mesh2_weights[:, t:t+1].clamp(0, 1)
            mesh = Mesh(verts=mesh2_verts,
                        faces=mesh2.faces, rgb=mesh2.rgb)
            mesh_weights = Mesh(verts=mesh2_weights_verts,
                        faces=mesh2.faces, rgb=mesh2_weights_rgb)
            meshes.append(mesh)
            meshes.append(mesh_weights)
            t_lines = torch.stack([mesh2_verts[pts_ids[t]], pts1[pts1_ids[t]]], dim=-1).permute(0, 2, 1)
            lines.append(t_lines)

        imgs= show_scene(pts3d=[], pts3d_colors=[], device=device, lines3d=lines,
                         meshes=meshes,
                         meshes_add_translation=False, pts3d_add_translation=False,
                         return_visualization=True, viewpoints_count=1, crop_white_border=True)

        imgs = crop_white_border_from_img(imgs, white_pad=30)
        H, W = imgs.shape[-2:]
        # show_imgs(imgs, height=640, width=1280)

        max_y = (proposal_dist_ref_geo_avg + proposal_dist_ref_appear_avg).max().item()
        fig, ax = plt.subplots(P, 1, figsize=(2, 6))
        for p in range(P):
            dist_geo = proposal_dist_ref_geo_avg[p].item()
            dist_appear = proposal_dist_ref_appear_avg[p].item()
            dist_total = dist_geo + dist_appear
            #if p == P-1:
            #    x = ['total', 'geometry', 'appearance']
            #else:
            x = [0, 1, 2] #, 'appearance']
            y = [dist_total, dist_geo, dist_appear]
            bar_labels = ['total', 'geometry', 'appearance']
            bar_colors = ['cornflowerblue', 'slategrey', 'lightgreen'] #''tab:green']
            ax[p].bar(x, y, label=bar_labels, color=bar_colors,  width=0.8)
            if p == 0:
                ax[p].legend(title='')
            # ax[p].set_ylabel(None) # 'distance'
            ax[p].set_title('')
            ax[p].set_xticks([], minor=False)
            ax[p].set_yticks([], minor=False)
            ax[p].set_ylim([0., max_y * 1.1])
            ax[p].set_xlim([-0.55, 2.55])
            # ax[p].legend(title='Distance')

        img = get_img_from_plot(ax=ax, fig=fig, axis_off=False)
        img = resize(img, scale_factor=H/img.shape[-2])
        total_imgs = torch.cat([imgs* 255, img], dim=-1)
        show_img(total_imgs, height=1080, width=1980, fpath='method_align3d.png')
        show_img(total_imgs, height=1080, width=1980)

@app.command()
def mv_pose_inference():
    logging.basicConfig(level=logging.INFO)
    device = 'cuda'
    dtype = torch.float

    run_name = '11-11_20-30-50_CO3D_NeMo_cat1_bicycle_ref4_filtered_mesh_slurm'
    mesh_fpath = Path('/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned/all_20s_to_5s_mesh/r4/mesh/bicycle/mesh.ply')
    aligned_name = 'all_20s_to_5s_mesh/r4'
    # od3d bench rsync -r 11-11_20-30-50_CO3D_NeMo_cat1_bicycle_ref4_filtered_mesh_slurm

    # /misc/lmbraid19/sommerl/exps/11-11_20-30-27_CO3D_NeMo_cat1_bicycle_ref0_filtered_mesh_slurm/nemo.ckpt

    #config_loaded.train.transform.transforms[0].config = None
    #config_loaded.categories = ['bicycle']
    #config_loaded.fpaths_meshes = {'bicycle': ''}
    config_transform = od3d.io.read_config_intern(rfpath=Path("methods").joinpath('transform', f"scale_mask_shorter_1_centerzoom512.yaml"))
    nemo = NeMo.create_by_name('nemo',
                               logging_dir=Path('nemo_out'),
                               config={'texture_dataset': None,
                                        'train': {'transform': {'transforms': [config_transform]}},
                                       'categories': ['bicycle'],
                                       'fpaths_meshes': {'bicycle': str(mesh_fpath)},
                                       'checkpoint': f'/misc/lmbraid19/sommerl/exps/{run_name}/nemo.ckpt',
                                       'multiview': {'batch_size': 3}
                                       })
    #config_dataset = {'categories': ['bicycle'], 'dict_nested_frames': {'val': ['n03792782_687']}} # n03792782_6218, n03792782_687

    sequences_count = 10
    samples_count = 3
    samples_max_position = 2
    mv_final_count = 2
    category = 'bicycle'

    co3d = CO3D.create_by_name('co3d_no_zsp_20s', config={'categories': [category], 'aligned_name': aligned_name, 'sequences_count_max_per_category': sequences_count}) # co3d_no_zsp_20s_aligned #co3d_5s_no_zsp_labeled 'co3d_50s_no_zsp_aligned' 'co3dv1_10s_zsp_aligned' 'co3d_10s_zsp_aligned' 'co3dv1_10s_zsp_unlabeled'
    categories = co3d.categories
    sequences = co3d.get_sequences()
    sequences_unique_names = [seq.name_unique for seq in sequences]
    instances_count = len(sequences)
    map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in sequences_unique_names])
    instance_ids = torch.LongTensor(list(range(instances_count)))

    dict_category_sequences = {category: list(sequence_dict.keys()) for category, sequence_dict in co3d.dict_nested_frames.items()}
    co3d.transform = nemo.transform_train
    # co3d.transform = nemo.transform_train

    dataset_sub = co3d.get_subset_by_sequences(dict_category_sequences=dict_category_sequences,
                                                  frames_count_max_per_sequence=nemo.config.multiview.batch_size)
    dataset_sub.transform = nemo.transform_train

    dataloader = torch.utils.data.DataLoader(dataset=dataset_sub, batch_size=nemo.config.multiview.batch_size,
                                             shuffle=False,
                                             collate_fn=dataset_sub.collate_fn,
                                             num_workers=nemo.config.test.dataloader.num_workers,
                                             pin_memory=nemo.config.test.dataloader.pin_memory)

    nemo.meshes.to(device=device)
    nemo.net.to(device=device)
    nemo.net.eval()
    from od3d.cv.geometry.mesh import MESH_RENDER_MODALITIES
    # next(nemo.net.parameters()).is_cuda
    for i, batch in tqdm(enumerate(dataloader)):
        if torch.cuda.is_available():
            batch.to(device=device)

        # show_imgs(rgbs=batch.rgb[:, :5])
        batch_res = nemo.inference_batch_multiview(batch)
        samples_cam_tform4x4_obj = batch_res['samples_cam_tform4x4_obj']
        pred_cam_tform4x4_obj = batch_res['cam_tform4x4_obj']
        C = samples_cam_tform4x4_obj.shape[1]
        sim = batch_res['samples_sim'].mean(dim=0)  #  batch_res['samples_sim']
        samples_ids = random.sample(torch.arange(C).tolist(), samples_count)
        samples_ids[samples_max_position] = sim.argmax().item()
        samples_ids = torch.LongTensor(samples_ids).to(device=device)
        samples_sim = sim[samples_ids]

        sims = torch.cat([torch.Tensor([samples_sim.min(), samples_sim.min()]).to(device=device), samples_sim, batch_res['sim'].mean(dim=0)], dim=0).detach()
        sims = (sims - sims.min()) / (sims.max() - sims.min())
        sims[2:] += 0.1

        S = len(sims)
        fig, ax = plt.subplots(1, 1, figsize=(6, 1))
        x = torch.arange(S).tolist()
        y = (sims).tolist()
        #bar_labels = ['total', 'geometry', 'appearance']
        #label = bar_labels
        #bar_colors = ['cornflowerblue', 'slategrey', 'lightgreen'] #''tab:green']
        # color=bar_colors
        ax.bar(x, y,  color='slategrey', width=0.4)
        # ax.legend(title='log probability')
        # ax[p].set_ylabel(None) # 'distance'
        ax.set_title('')
        ax.set_xticks([], minor=False)
        ax.set_yticks([], minor=False)
        ax.set_ylim([0., 1.2])
        ax.set_xlim([-0.55, S - 1 + 0.55])
        img_log_prob = get_img_from_plot(ax=ax, fig=fig, axis_off=True)
        #img_log_prob = resize(img_log_prob, )
        # show_img(img_log_prob)

        cam_intr4x4 = batch.cam_intr4x4
        img_feats = nemo.net(batch.rgb)
        size = torch.Tensor([img_feats.shape[2] * nemo.down_sample_rate, img_feats.shape[3] * nemo.down_sample_rate]).to(device=device)
        B, F, H, W = img_feats.shape
        mesh_feats = nemo.meshes.render_feats(cams_tform4x4_obj=samples_cam_tform4x4_obj,
                                              cams_intr4x4=cam_intr4x4[:, None],
                                              imgs_sizes=size, meshes_ids=batch.label,
                                              modality=MESH_RENDER_MODALITIES.FEATS,
                                              down_sample_rate=nemo.down_sample_rate,
                                              broadcast_batch_and_cams=True)
        C = mesh_feats.shape[1]
        _, _, pca_V = torch.pca_lowrank(torch.cat([img_feats.permute(0, 2, 3, 1).reshape(-1, F), mesh_feats.permute(0, 1, 3, 4, 2).reshape(-1, F)], dim=0))
        img_feats_pca = torch.matmul(img_feats.permute(0, 2, 3, 1).reshape(-1, F), pca_V[:, 1:4]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
        mesh_feats_pca = torch.matmul(mesh_feats.permute(0, 1, 3, 4, 2).reshape(-1, F), pca_V[:, 1:4]).reshape(B, C, H, W, -1).permute(0, 1, 4, 2, 3)
        mesh_feats_pca_mask_bg = (mesh_feats_pca == 0.).all(dim=2, keepdim=True)

        mesh_feats_optimal = nemo.meshes.render_feats(cams_tform4x4_obj=pred_cam_tform4x4_obj,
                                                      cams_intr4x4=cam_intr4x4,
                                                      imgs_sizes=size, meshes_ids=batch.label,
                                                      modality=MESH_RENDER_MODALITIES.FEATS,
                                                      down_sample_rate=nemo.down_sample_rate,
                                                      broadcast_batch_and_cams=False)
        mesh_feats_optimal_pca = torch.matmul(mesh_feats_optimal.permute(0, 2, 3, 1).reshape(-1, F), pca_V[:, 1:4]).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        mesh_feats_optimal_pca_mask_bg = (mesh_feats_optimal_pca == 0.).all(dim=1, keepdim=True)

        feats_pca = torch.cat([img_feats_pca[:, None], mesh_feats_pca[:, samples_ids], mesh_feats_optimal_pca[:, None]], dim=1)
        feats_pca = (feats_pca - feats_pca.min()) / (feats_pca.max()- feats_pca.min())
        feats_pca[:, 1:-1][mesh_feats_pca_mask_bg[:, samples_ids].expand(*feats_pca[:, 1:-1].shape)] = 1.
        feats_pca[:, -1][mesh_feats_optimal_pca_mask_bg.expand(*feats_pca[:, -1].shape)] = 1.

        rgb = (batch.rgb - batch.rgb.min()) / (batch.rgb.max() - batch.rgb.min())
        rgb_no_rgb_mask = (rgb < 0.1).all(dim=1, keepdim=True)
        rgb[rgb_no_rgb_mask.expand(*rgb.shape)] = 1.
        _, _, H_rgb, W_rgb = batch.rgb.shape
        scale_factor = H_rgb / H
        feats_pca = resize(feats_pca.reshape(-1, 3, H, W), scale_factor=H_rgb / H, mode='nearest_v2').reshape(B, -1, 3, int(H * scale_factor), int(W* scale_factor))
        img = torch.cat([rgb[:, None], feats_pca], dim=1)
        from od3d.cv.visual.show import imgs_to_img
        img = imgs_to_img(img[:mv_final_count], pad = 10)
        img_log_prob = (resize(img_log_prob, scale_factor=img.shape[-1] / img_log_prob.shape[-1])).to(device=device)
        img_mv_pose_inference = torch.cat([img * 255, img_log_prob], dim=-2)
        show_img(rgb=img_mv_pose_inference, height=1080, width=1980, fpath='mv_pose_inference.png')
        show_img(rgb=img_mv_pose_inference, height=1080, width=1980)


    rand_category_id = categories.index(category) # 'car', 'chair',
    rand_category_instance_ids = instance_ids[map_seq_to_cat == rand_category_id]

    # # CALCULATING CATEGORICAL PCA
    # category_instance_ids = instance_ids[map_seq_to_cat == rand_category_id]
    # categorical_features = []
    # for instance_id_in_category, instance_id in enumerate(category_instance_ids):
    #     instance_feats = sequences[instance_id].feats
    #
    #     if isinstance(instance_feats, List):
    #         categorical_features += torch.cat([vert_feats for vert_feats in instance_feats], dim=0)
    #     else:
    #         categorical_features.append(instance_feats)
    # categorical_features = torch.stack(categorical_features, dim=0)
    # _, _, categorical_pca_V = torch.pca_lowrank(categorical_features)
    # sequences[category_instance_ids[0]].categorical_pca_V = categorical_pca_V


    rand_category_rand_instance_ids = random.sample(rand_category_instance_ids.tolist(), k=1)
    seq1 = sequences[rand_category_rand_instance_ids[0]]


@app.command()
def mv_pose_train():
    logging.basicConfig(level=logging.INFO)
    device = 'cuda'
    dtype = torch.float

    config_transform = od3d.io.read_config_intern(
        rfpath=Path("methods").joinpath('transform', f"scale_mask_separate_centerzoom512.yaml")) # scale_mask_shorter_1_centerzoom512, scale_mask_separate_centerzoom512

    run_name = '11-11_20-30-50_CO3D_NeMo_cat1_bicycle_ref4_filtered_mesh_slurm'
    mesh_fpath = Path('/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned/all_20s_to_5s_mesh/r4/mesh/bicycle/mesh.ply')
    aligned_name = 'all_20s_to_5s_mesh/r4'
    # od3d bench rsync -r 11-11_20-30-50_CO3D_NeMo_cat1_bicycle_ref4_filtered_mesh_slurm

    # 11-14_23-51-54_CO3D_NeMo_cat1_car_ref2_mesh_slurm
    run_name = '11-14_23-51-54_CO3D_NeMo_cat1_car_ref2_mesh_slurm'
    mesh_fpath = Path('/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned/all_20s_to_5s_mesh/r2/mesh/car/mesh.ply')
    aligned_name = 'all_20s_to_5s_mesh/r2'
    # od3d bench rsync -r 11-14_23-51-54_CO3D_NeMo_cat1_car_ref2_mesh_slurm




    # /misc/lmbraid19/sommerl/exps/11-11_20-30-27_CO3D_NeMo_cat1_bicycle_ref0_filtered_mesh_slurm/nemo.ckpt

    #config_loaded.train.transform.transforms[0].config = None
    #config_loaded.categories = ['bicycle']
    #config_loaded.fpaths_meshes = {'bicycle': ''}

    frames_count = 6
    sequences_count = 2
    depth_scale = 0.3
    category = 'car' # bicycle
    pad = 50
    viewpoint_id = 2

    # _aligned # _aligned
    co3d = CO3D.create_by_name('co3d_no_zsp_20s_aligned', config={
        'categories': [category], 'aligned_name': aligned_name,
        'sequences_count_max_per_category': sequences_count,
        'frames_count_max_per_sequence': frames_count,
        'dict_nested_frames_ban':{
            'bicycle': {
                '354_37645_70054': None,
            },
            'car': {
                '185_19982_37678': None,
                '194_20939_43630': None,
                #'194_20900_41097': None,
                '206_21810_45890': None,
            }
        }
    })

    co3d.transform = OD3D_Transform.create_by_name('scale_mask_separate_centerzoom512')  # scale_mask_shorter_1_centerzoom512, scale_mask_separate_centerzoom512 OD3D_Transform.create_by_name('scale_mask_separate_centerzoom512')
    co3d.transform.center_use_mask = True
    cams_tform4x4_world = []
    cams_intr4x4 = []
    cams_imgs = []
    meshes_aligned = []
    imgs_not_aligned_imgs = []
    logger.info('rendering not aligned images')


    nemo = NeMo.create_by_name('nemo',
                               logging_dir=Path('nemo_out'),
                               config={'texture_dataset': None,
                                        'train': {'transform': {'transforms': [config_transform]}},
                                       'categories': [category],
                                       'fpaths_meshes': {category: str(mesh_fpath)},
                                       'checkpoint': f'/misc/lmbraid19/sommerl/exps/{run_name}/nemo.ckpt'})

    aligned_mesh = nemo.meshes

    _, _, categorical_pca_V = torch.pca_lowrank(nemo.meshes.feats)
    verts_feats_pca = torch.matmul(nemo.meshes.feats.to(device=device), categorical_pca_V[:, 0:3])
    aligned_mesh.rgb =  (verts_feats_pca.nan_to_num() + 0.5).clamp(0, 1)

    # aligned_mesh.rgb = aligned_mesh.get_verts_ncds_cat_with_mesh_ids()

    img_aligned_correspondences = show_scene(meshes=aligned_mesh, viewpoints_count=viewpoint_id+1, return_visualization=True, cams_imgs_depth_scale=depth_scale)[viewpoint_id]
    img_aligned_correspondences = crop_white_border_from_img(img_aligned_correspondences, white_pad=pad).detach().cpu()
    show_img(img_aligned_correspondences, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)



    dataloader = torch.utils.data.DataLoader(dataset=co3d, batch_size=frames_count, shuffle=False,
                                             collate_fn=co3d.collate_fn, num_workers=4)

    # next(nemo.net.parameters()).is_cuda
    cams_imgs = []
    for i, batch in tqdm(enumerate(dataloader)):
        if torch.cuda.is_available():
            batch.to(device=device)
        for b in range(len(batch)):
            show_img(batch.rgb[b], fpath=f'img_video_{i}_{b}.png')
        cams_imgs.append(batch.rgb)
    cams_imgs = torch.stack(cams_imgs, dim=0)
    from od3d.cv.visual.show import imgs_to_img
    imgs_video = cams_imgs
    img_videos = imgs_to_img(imgs_video, pad=pad //2, pad_value=0).detach().cpu()
    show_img(img_videos, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    for seq in co3d.get_sequences():
        logger.info(f'seq {seq}')
        seq_cams_tform4x4_world, seq_cams_intr4x4, seq_cams_imgs = seq.get_cams(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.PCL, cams_count=frames_count, show_imgs=True)
        seq_mesh = seq.get_mesh(mesh_source=CUBOID_SOURCES.DEFAULT, add_rgb_from_pca=True, device=device)
        seq_tform = torch.Tensor([
            [1., 0., 0., 0.],
            [0., 0., -1., 0.],
            [0., -1., 0., 0.],
            [0., 0., 0., 1.]
        ]).to(device)
        seq_mesh.verts = transf3d_broadcast(seq_mesh.verts, transf4x4=seq_tform)
        img_not_aligned_img = show_scene(meshes=[seq_mesh], viewpoints_count=viewpoint_id+1, return_visualization=True, cams_imgs_depth_scale=depth_scale)[viewpoint_id]
        imgs_not_aligned_imgs.append(crop_white_border_from_img(img_not_aligned_img))

        show_img(img_not_aligned_img, fpath=f'{seq.name}.png')
        # show_img(imgs_not_aligned)
        #
        # seq_cams_tform4x4_world, seq_cams_intr4x4, seq_cams_imgs = seq.get_cams(
        #     cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.ALIGNED, cams_count=5, show_imgs=True)
        # cams_intr4x4.append(torch.stack(seq_cams_intr4x4, dim=0))
        # cams_tform4x4_world.append(torch.stack(seq_cams_tform4x4_world, dim=0))
        # cams_imgs += seq_cams_imgs

    imgs_not_aligned_imgs = [resize(img, scale_factor=imgs_not_aligned_imgs[0].shape[-1] / img.shape[-1]) for img in imgs_not_aligned_imgs]
    img_not_aligned_imgs = torch.cat(imgs_not_aligned_imgs, dim=-2)
    img_not_aligned_imgs = crop_white_border_from_img(img_not_aligned_imgs, white_pad=pad)
    show_img(img_not_aligned_imgs, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    #cams_intr4x4 = torch.cat(cams_intr4x4, dim=0)
    #cams_tform4x4_world = torch.cat(cams_tform4x4_world, dim=0)

    img_videos = resize(img_videos, scale_factor= img_not_aligned_imgs.shape[-2] / img_videos.shape[-2])
    img_aligned_correspondences = resize(img_aligned_correspondences, scale_factor= img_not_aligned_imgs.shape[-2] / img_aligned_correspondences.shape[-2])
    # img_pipeline = torch.cat([255 * img_not_aligned_imgs , img_videos, 255 * img_aligned_correspondences], dim=-1)
    img_pipeline = torch.cat([img_not_aligned_imgs , img_aligned_correspondences], dim=-1)

    show_img(img_pipeline, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, fpath='pipeline_template.png')
    show_img(img_pipeline, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    #
    # # logger.info('rendering aligned images')
    # # img_aligned_imgs = show_scene(meshes=aligned_mesh, cams_tform4x4_world=cams_tform4x4_world, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs, viewpoints_count=3, return_visualization=True, cams_imgs_depth_scale=depth_scale)[1]
    # # img_aligned_imgs = crop_white_border_from_img(img_aligned_imgs, white_pad=50)
    # # show_img(img_aligned_imgs)
    #
    # from od3d.cv.geometry.mesh import MESH_RENDER_MODALITIES
    #
    # logger.info('rendering correspondence images...')
    # cams_imgs_rendered = []
    # for c, cam_img in tqdm(enumerate(cams_imgs)):
    #     img_size = torch.Tensor(list(cams_imgs[c].shape[-2:]))
    #     cams_imgs_rendered.append(255 * aligned_mesh.render_feats(#cams_tform4x4_obj=cams_tform4x4_world[c:c + 1],
    #                                                               #cams_intr4x4=cams_intr4x4[c:c + 1], imgs_sizes=img_size,
    #                                                               #broadcast_batch_and_cams=False,
    #                                                               modality=MESH_RENDER_MODALITIES.RGB)[0])
    # img_aligned_correspondences = show_scene(meshes=aligned_mesh, cams_tform4x4_world=cams_tform4x4_world, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs_rendered, viewpoints_count=3, return_visualization=True, cams_imgs_depth_scale=depth_scale)[1]
    # img_aligned_correspondences = crop_white_border_from_img(img_aligned_correspondences, white_pad=pad)
    #
    # # imgs_not_aligned_imgs = [resize(img, scale_factor=imgs_not_aligned_imgs[0].shape[-1] / img.shape[-1]) for img in imgs_not_aligned_imgs]
    #
    # #img_aligned_imgs = resize(img_aligned_imgs, scale_factor= img_not_aligned_imgs.shape[-2] / img_aligned_imgs.shape[-2])
    # img_aligned_correspondences = resize(img_aligned_correspondences, scale_factor= img_not_aligned_imgs.shape[-2] / img_aligned_correspondences.shape[-2])
    # img_videos = resize(img_videos, scale_factor= img_not_aligned_imgs.shape[-2] / img_videos.shape[-2])
    #
    # # img_aligned_imgs
    # img_pipeline = torch.cat([img_not_aligned_imgs, img_videos, img_aligned_correspondences], dim=-1)
    #
    # show_img(img_pipeline, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, fpath='pipeline.png')
    # show_img(img_pipeline, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    #
    # #meshes_not_aligned = Meshes.load_from_meshes(meshes_not_aligned, device=device)
    # #show_scene(meshes=meshes_not_aligned)
    #
    # from od3d.cv.visual.show import imgs_to_img
    # imgs_video = cams_imgs
    # #img_videos = imgs_to_img(cams_imgs, pad=10, pad_value=255)
    # #show_img(img_videos)
    #
    #
    # nemo = NeMo.create_by_name('nemo',
    #                            logging_dir=Path('nemo_out'),
    #                            config={'texture_dataset': None,
    #                                     'train': {'transform': {'transforms': [config_transform]}},
    #                                    'categories': ['bicycle'],
    #                                    'fpaths_meshes': {'bicycle': str(mesh_fpath)},
    #                                    'checkpoint': f'/misc/lmbraid19/sommerl/exps/{run_name}/nemo.ckpt'})
    # config_dataset = {'categories': ['bicycle'], 'dict_nested_frames': {'val': ['n03792782_687']}} # n03792782_6218, n03792782_687
    # dataset = ObjectNet3D.create_by_name('objectnet3d', config=config_dataset)
    # dataset.transform = nemo.transform_train  # OD3D_Transform.create_by_name('scale_mask_separate_centerzoom512')
    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False,
    #                                          collate_fn=dataset.collate_fn, num_workers=4)
    # nemo.net.to(device=device)
    # nemo.net.eval()
    # # next(nemo.net.parameters()).is_cuda
    # for i, batch in tqdm(enumerate(dataloader)):
    #     if torch.cuda.is_available():
    #         batch.to(device=device)
    #
    #     from od3d.cv.visual.blend import blend_rgb
    #     batch_res = nemo.inference_batch_single_view(batch)
    #
    #     #batch_res.keys()
    #     pred_cam_tform4x4_obj = batch_res['cam_tform4x4_obj']
    #     pred_verts_ncds = nemo.get_ncds_with_cam(cam_intr4x4=batch.cam_intr4x4, cam_tform4x4_obj=pred_cam_tform4x4_obj,
    #                                          categories_ids=batch.label, size=batch.size,
    #                                          down_sample_rate=1., pre_rendered=False)
    #     for b in range(len(batch)):
    #         img_rgb = batch.rgb[b].clone()
    #         img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())
    #         img_in_the_wild = blend_rgb(resize(img_rgb, scale_factor=1.), pred_verts_ncds[b], alpha1=0.7, alpha2=0.7)
    #         show_img(img_in_the_wild, fpath='mv_pose_in_the_wild.png')
    #
    # samples_count = 3
    # sequences_count = 3
    # co3d = CO3D.create_by_name('co3d_no_zsp_20s_aligned', config={'aligned_name': aligned_name, 'sequences_count_max_per_category': sequences_count}) # co3d_no_zsp_20s_aligned #co3d_5s_no_zsp_labeled 'co3d_50s_no_zsp_aligned' 'co3dv1_10s_zsp_aligned' 'co3d_10s_zsp_aligned' 'co3dv1_10s_zsp_unlabeled'
    # categories = co3d.categories
    # sequences = co3d.get_sequences()
    # sequences_unique_names = [seq.name_unique for seq in sequences]
    # instances_count = len(sequences)
    # map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in sequences_unique_names])
    # instance_ids = torch.LongTensor(list(range(instances_count)))
    #
    # category = 'bicycle'
    # rand_category_id = categories.index(category) # 'car', 'chair',
    # rand_category_instance_ids = instance_ids[map_seq_to_cat == rand_category_id]
    #
    # # # CALCULATING CATEGORICAL PCA
    # # category_instance_ids = instance_ids[map_seq_to_cat == rand_category_id]
    # # categorical_features = []
    # # for instance_id_in_category, instance_id in enumerate(category_instance_ids):
    # #     instance_feats = sequences[instance_id].feats
    # #
    # #     if isinstance(instance_feats, List):
    # #         categorical_features += torch.cat([vert_feats for vert_feats in instance_feats], dim=0)
    # #     else:
    # #         categorical_features.append(instance_feats)
    # # categorical_features = torch.stack(categorical_features, dim=0)
    # # _, _, categorical_pca_V = torch.pca_lowrank(categorical_features)
    # # sequences[category_instance_ids[0]].categorical_pca_V = categorical_pca_V
    #
    # while True:
    #
    #     rand_category_rand_instance_ids = random.sample(rand_category_instance_ids.tolist(), k=samples_count)
    #     seq1 = sequences[rand_category_rand_instance_ids[0]]
    #     cams_imgs = []
    #     cams_intr4x4 = []
    #     cams_tform4x4_obj = []
    #     for s in range(samples_count):
    #         seq = sequences[rand_category_rand_instance_ids[s]]
    #         frames_ids = torch.arange(seq.frames_count)[::50]
    #         frames = seq.get_frames(frames_ids=frames_ids)
    #         _cams_imgs = torch.stack([frame.rgb for frame in frames], dim=0).to(device=device)
    #         _cams_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=device)
    #         _cams_tform4x4_obj = torch.stack([frame.cam_tform4x4_obj for frame in frames], dim=0).to(device=device)
    #         cams_imgs += [cam_img.to(device=device) for cam_img in _cams_imgs]
    #         cams_intr4x4.append(_cams_intr4x4)
    #         cams_tform4x4_obj.append(_cams_tform4x4_obj)
    #
    #     cams_intr4x4 = torch.cat(cams_intr4x4, dim=0).to(device=device)
    #     cams_tform4x4_obj = torch.cat(cams_tform4x4_obj, dim=0).to(device=device)
    #
    #     #pts3d = seq1.get_pcl(pcl_source=seq1.pcl_source)
    #     #pts3d_colors = seq1.get_pcl_colors(pcl_source=seq1.pcl_source)
    #     mesh = Meshes.load_from_meshes([seq1.get_mesh(mesh_source=CUBOID_SOURCES.ALIGNED)], device=device)
    #     #mesh = Meshes.load_from_files(fpaths_meshes=[mesh_fpath], device=device)
    #     mesh.rgb = mesh.get_verts_ncds_cat_with_mesh_ids()
    #     #pts3d = transf3d_broadcast(pts3d=pts3d.to(device=device, dtype=dtype),
    #     #                           transf4x4=seq1.droid_slam_aligned_tform_droid_slam.to(device=device))
    #     # meshes=mesh,
    #
    #     cams_imgs_depth_scale=0.3
    #     viewpoints_count = 5
    #     logger.info('rendering real images...')
    #     imgs_real = show_scene(meshes=mesh, cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs, viewpoints_count=viewpoints_count, return_visualization=True, crop_white_border=True, cams_imgs_depth_scale=cams_imgs_depth_scale, cams_show_wireframe=False)
    #     # frames_ids = torch.arange(seq1.frames_count)[::50]
    #     # frames = seq1.get_frames(frames_ids=frames_ids)
    #     # cams_imgs = torch.stack([frame.rgb for frame in frames], dim=0).to(device=device)
    #     # cams_intr4x4 = torch.stack([frame.cam_intr4x4 for frame in frames], dim=0).to(device=device)
    #     # cams_tform4x4_obj = torch.stack([frame.cam_tform4x4_obj for frame in frames], dim=0).to(device=device)
    #     from od3d.cv.geometry.mesh import MESH_RENDER_MODALITIES
    #
    #     logger.info('rendering rendered images...')
    #     cams_imgs_rendered = []
    #     for c, cam_img in enumerate(cams_imgs):
    #         img_size = torch.Tensor(list(cams_imgs[c].shape[-2:]))
    #         cams_imgs_rendered.append(255 * mesh.render_feats(cams_tform4x4_obj=cams_tform4x4_obj[c:c+1], cams_intr4x4=cams_intr4x4[c:c+1], imgs_sizes=img_size, broadcast_batch_and_cams=False, modality=MESH_RENDER_MODALITIES.RGB)[0])
    #
    #     imgs_rendered = show_scene(meshes=mesh, cams_tform4x4_world=cams_tform4x4_obj, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs_rendered, viewpoints_count=viewpoints_count, return_visualization=True, crop_white_border=True, cams_imgs_depth_scale=cams_imgs_depth_scale, cams_show_wireframe=False)
    #
    #     for i in range(viewpoints_count):
    #         img_real = crop_white_border_from_img(imgs_real[i], white_pad=30)
    #         img_rendered = crop_white_border_from_img(imgs_rendered[i], white_pad=30)
    #         img_rendered = resize(img_rendered, scale_factor=img_real.shape[-2]/img_rendered.shape[-2])
    #         img_in_the_wild = resize(img_in_the_wild.detach().cpu(), scale_factor=img_real.shape[-2]/img_in_the_wild.shape[-2])
    #         logger.info('writing img...')
    #         show_imgs(torch.cat([img_real * 255, img_rendered* 255, img_in_the_wild], dim=-1), height=640, width=1280, fpath=f'mv_pose_train_{i}.png')
    #


@app.command()
def teaser():
    logging.basicConfig(level=logging.INFO)
    device = 'cuda'
    dtype = torch.float

    config_transform = od3d.io.read_config_intern(
        rfpath=Path("methods").joinpath('transform', f"scale_mask_separate_centerzoom512.yaml")) # scale_mask_shorter_1_centerzoom512, scale_mask_separate_centerzoom512

    run_name = '11-11_20-30-50_CO3D_NeMo_cat1_bicycle_ref4_filtered_mesh_slurm'
    run_name = '11-11_20-30-50_CO3D_NeMo_cat1_bicycle_ref4_filtered_mesh_slurm'
    mesh_fpath = Path('/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned/all_20s_to_5s_mesh/r4/mesh/bicycle/mesh.ply')
    mesh_fpath = Path('/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned/all_50s_to_5s_mesh/r4/mesh/bicycle/mesh.ply')
    aligned_name = 'all_20s_to_5s_mesh/r4'
    aligned_name = 'all_50s_to_5s_mesh/r4'

    # od3d bench rsync -r 11-11_20-30-50_CO3D_NeMo_cat1_bicycle_ref4_filtered_mesh_slurm

    # /misc/lmbraid19/sommerl/exps/11-11_20-30-27_CO3D_NeMo_cat1_bicycle_ref0_filtered_mesh_slurm/nemo.ckpt

    #config_loaded.train.transform.transforms[0].config = None
    #config_loaded.categories = ['bicycle']
    #config_loaded.fpaths_meshes = {'bicycle': ''}

    frames_count = 6
    frames_shown_in_the_wild_count = 6
    frames_shown_count = 4
    sequences_count = 3
    pad = 30
    depth_scale = 0.3
    viewpoint = 2
    category = 'bicycle'

    # _aligned
    co3d = CO3D.create_by_name('co3d_no_zsp_20s_aligned', config={
        'categories': [category], 'aligned_name': aligned_name,
        'sequences_count_max_per_category': sequences_count,
        'frames_count_max_per_sequence': frames_count,
        'dict_nested_frames_ban':{
            'bicycle': {
                '354_37645_70054': None,
            }
        }
    })

    co3d.transform = OD3D_Transform.create_by_name('scale_mask_separate_centerzoom512')  # scale_mask_shorter_1_centerzoom512, scale_mask_separate_centerzoom512 OD3D_Transform.create_by_name('scale_mask_separate_centerzoom512')
    co3d.transform.center_use_mask = True

    cams_tform4x4_world = []
    cams_intr4x4 = []
    cams_imgs = []
    meshes_aligned = []
    imgs_not_aligned_imgs = []
    logger.info('rendering not aligned images')
    for seq in co3d.get_sequences():
        logger.info(f'seq {seq}')
        seq_cams_tform4x4_world, seq_cams_intr4x4, seq_cams_imgs = seq.get_cams(
            cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.ALIGNED, cams_count=frames_count, show_imgs=True)
        cams_intr4x4.append(torch.stack(seq_cams_intr4x4, dim=0))
        cams_tform4x4_world.append(torch.stack(seq_cams_tform4x4_world, dim=0))
        cams_imgs += seq_cams_imgs

    cams_intr4x4 = torch.cat(cams_intr4x4, dim=0)
    cams_tform4x4_world = torch.cat(cams_tform4x4_world, dim=0)

    block_imgs_ids = torch.LongTensor([0, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 16])
    # for  4 frames 0, 4, 7
    # for 6 frames 0, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 16
    filter_imgs_mask = torch.ones(size=(len(cams_intr4x4),)).to(dtype=bool) # , device=device)
    filter_imgs_mask[block_imgs_ids] = False
    cams_intr4x4 = cams_intr4x4[filter_imgs_mask]
    cams_tform4x4_world = cams_tform4x4_world[filter_imgs_mask]
    block_imgs_ids = block_imgs_ids.sort(descending=True).values
    for block_img_id in block_imgs_ids:
        cams_imgs.pop(block_img_id)

    aligned_mesh = co3d.get_sequences()[0].get_mesh(mesh_source=CUBOID_SOURCES.ALIGNED, add_rgb_from_pca=False, device=device)
    aligned_mesh = Meshes.load_from_meshes([aligned_mesh], device=device)
    aligned_mesh.rgb = aligned_mesh.get_verts_ncds_cat_with_mesh_ids()
    logger.info('rendering aligned images')
    img_aligned_imgs = show_scene(meshes=aligned_mesh, cams_tform4x4_world=cams_tform4x4_world, cams_intr4x4=cams_intr4x4, cams_imgs=cams_imgs, viewpoints_count=viewpoint+1, return_visualization=True, cams_imgs_depth_scale=depth_scale)[viewpoint]
    img_aligned_imgs = crop_white_border_from_img(img_aligned_imgs, white_pad=pad).to(device=device)
    show_img(img_aligned_imgs)

    dataloader = torch.utils.data.DataLoader(dataset=co3d, batch_size=frames_count, shuffle=False,
                                             collate_fn=co3d.collate_fn, num_workers=4)

    # next(nemo.net.parameters()).is_cuda
    cams_imgs = []
    for i, batch in tqdm(enumerate(dataloader)):
        if torch.cuda.is_available():
            batch.to(device=device)
        cams_imgs.append(batch.rgb)

    cams_imgs = torch.stack(cams_imgs, dim=0)
    imgs_video = cams_imgs[:, :frames_shown_count]
    img_videos = imgs_to_img(imgs_video, pad=pad * 3, pad_value=0)
    #img_videos = imgs_to_img(cams_imgs, pad=10, pad_value=255)
    #show_img(img_videos)

    # 'n02834778_10025',
    # 'n02834778_10058',
    # 'n02834778_10129',
    # 'n02834778_10218',
    # 'n02834778_10227',
    # 'n02834778_10327',
    # 'n02834778_10363',
    # 'n02834778_10619',
    # 'n02834778_1107',
    # 'n02834778_11107',
    #
    # n02834778_10218
    # n02834778_10753
    # n02834778_10979
    # n02834778_1130

    nemo = NeMo.create_by_name('nemo',
                               logging_dir=Path('nemo_out'),
                               config={'texture_dataset': None,
                                        'train': {'transform': {'transforms': [config_transform]}},
                                       'categories': ['bicycle'],
                                       'fpaths_meshes': {'bicycle': str(mesh_fpath)},
                                       'checkpoint': f'/misc/lmbraid19/sommerl/exps/{run_name}/nemo.ckpt'})
    config_dataset = {'categories': ['bicycle'],
                      'dict_nested_frames': {
                          'val': ['n03792782_687', 'n04126066_11131',], # 'n03792782_687', 'n04126066_11131', 'n03792782_6218'
                          'test': ['n02834778_10129', 'n02834778_10753', 'n02834778_10979', 'n02834778_1130' ]}} # n02834778_10218
    dataset = ObjectNet3D.create_by_name('objectnet3d', config=config_dataset)
    dataset.transform = nemo.transform_train  # OD3D_Transform.create_by_name('scale_mask_separate_centerzoom512')
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                                             collate_fn=dataset.collate_fn, num_workers=4)

    imgs_in_the_wild = []
    nemo.net.to(device=device)
    nemo.net.eval()
    # next(nemo.net.parameters()).is_cuda
    for i, batch in tqdm(enumerate(dataloader)):
        if i == frames_shown_in_the_wild_count:
            break
        if torch.cuda.is_available():
            batch.to(device=device)

        from od3d.cv.visual.blend import blend_rgb
        batch_res = nemo.inference_batch_single_view(batch)

        #batch_res.keys()
        pred_cam_tform4x4_obj = batch_res['cam_tform4x4_obj']
        pred_verts_ncds = nemo.get_ncds_with_cam(cam_intr4x4=batch.cam_intr4x4, cam_tform4x4_obj=pred_cam_tform4x4_obj,
                                             categories_ids=batch.label, size=batch.size,
                                             down_sample_rate=1., pre_rendered=False)
        for b in range(len(batch)):
            img_rgb = batch.rgb[b].clone()
            img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())
            imgs_in_the_wild.append(torch.cat([img_rgb * 255, blend_rgb(resize(img_rgb, scale_factor=1.), pred_verts_ncds[b], alpha1=0.3, alpha2=0.7)], dim=-1))

    imgs_in_the_wild = torch.stack(imgs_in_the_wild, dim=0)

    img_in_the_wild = imgs_to_img(imgs_in_the_wild.reshape(sequences_count, frames_shown_in_the_wild_count // sequences_count, *imgs_in_the_wild.shape[1:]), pad=pad, pad_value=0)
    # img_in_the_wild = imgs_to_img(imgs_in_the_wild[:, None], pad=pad, pad_value=0)

    img_aligned_imgs = resize(img_aligned_imgs, scale_factor=img_videos.shape[-2] / img_aligned_imgs.shape[-2] )
    img_in_the_wild = resize(img_in_the_wild, scale_factor=img_videos.shape[-2] / img_in_the_wild.shape[-2] )
    img = torch.cat([img_videos, img_aligned_imgs * 255, img_in_the_wild], dim=-1)
    img[:, (img == 0).all(dim=0)] = 255

    show_img(img, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, fpath='teaser.png')
    show_img(img, width=WINDOW_WIDTH, height=WINDOW_HEIGHT,)


"""    
    logging.basicConfig(level=logging.INFO)
    viewpoints_count = 2
    # category_meshes = self.meshes.get_meshes_with_ids(meshes_ids=category_instance_ids)
    device = 'cuda:0'
    dtype = torch.float

    co3d = CO3D.create_by_name('co3d_50s_no_zsp_aligned') # 'co3dv1_10s_zsp_aligned' 'co3d_10s_zsp_aligned' 'co3dv1_10s_zsp_unlabeled'
    categories = co3d.categories
    sequences = co3d.get_sequences()


    sequences_unique_names = [seq.name_unique for seq in sequences]
    instances_count = len(sequences)
    map_seq_to_cat = torch.LongTensor([categories.index(name.split('/')[0]) for name in sequences_unique_names])
    categories_count = len(categories)
    instances_count_per_category = [(map_seq_to_cat == c).sum().item() for c in range(categories_count)]
    instance_ids = torch.LongTensor(list(range(instances_count)))

    mesh_source = CUBOID_SOURCES.DEFAULT
    meshes = Meshes.load_from_meshes([seq.get_mesh(mesh_source=mesh_source, add_rgb_from_pca=True, device=device) for seq in sequences], device=device)
    sequences_mesh_ids_for_verts = meshes.get_mesh_ids_for_verts()


    # CALCULATING CATEGORICAL PCA
    logger.info('calculating categorical pca...')
    categorical_features = {}
    categorical_pca_V = {}
    for cat_id, category in enumerate(categories):
        logger.info(category)
        category_instance_ids = instance_ids[map_seq_to_cat == cat_id]
        categorical_features[category] = []
        for instance_id_in_category, instance_id in enumerate(category_instance_ids):
            instance_feats = sequences[instance_id].feats

            if isinstance(instance_feats, List):
                categorical_features[category] += torch.cat([vert_feats for vert_feats in instance_feats], dim=0)
            else:
                categorical_features[category].append(instance_feats)
        categorical_features[category] = torch.stack(categorical_features[category], dim=0)
        _, _, categorical_pca_V[category] = torch.pca_lowrank(categorical_features[category])
        sequences[category_instance_ids[0]].categorical_pca_V = categorical_pca_V[category]

    ## SHOW SINGLE SEQUENCES
    # for sequence in sequences:
    #     logger.info(sequence.name_unique)
    #     mesh = sequence.mesh
    #     instance_feats = sequence.feats
    #     pca_V = sequence.categorical_pca_V.to(device=device)
    #     if isinstance(instance_feats, List):
    #         verts_feats_pca = torch.stack([torch.matmul(vert_feats, pca_V[:, :3]).mean(dim=0) for vert_feats in instance_feats], dim=0)
    #     else:
    #         verts_feats_pca = torch.matmul(instance_feats, pca_V[:, :3])
    #     mesh.rgb = (verts_feats_pca.nan_to_num() + 1.) / 2.
    #     show_scene(meshes=[mesh])
    #     sequence.show(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.DROID_SLAM, pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN, cams_count=200, show_imgs=False)
    #     #sequence.show(cam_tform_obj_source=CAM_TFORM_OBJ_SOURCES.CO3D, pcl_source=PCL_SOURCES.CO3D, cams_count=200, show_imgs=False)
    #

    offset_x = 3.
    offset_y = 0.
    offset_z = -5.
    offset_y_prev = 0.
    pts3d = []
    pts3d_colors = []

    for cat_id, category in enumerate(categories):
        logger.info(category)
        category_instance_ids = instance_ids[map_seq_to_cat == cat_id]

        for instance_id_in_category, instance_id in enumerate(category_instance_ids):

            if instance_id_in_category == 0:
                droid_slam_aligned_tform_droid_slam = sequences[instance_id].aligned_obj_tform_obj.to(
                    device=device, dtype=dtype)
                pts3d_first = transf3d_broadcast(pts3d=sequences[instance_id].get_pcl(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN).to(device=device, dtype=dtype),transf4x4=droid_slam_aligned_tform_droid_slam)

                offset_y = offset_y_prev - pts3d_first[:, 1].min().item() * 1.2
                offset_y_prev = offset_y + pts3d_first[:, 1].max().item() * 1.2
                logger.info(f'category {category} offset {offset_y}')
            droid_slam_aligned_tform_droid_slam = sequences[instance_id].aligned_obj_tform_obj.to(device=device, dtype=dtype)

            offset_aligned_tform_aligned = tform4x4_from_transl3d(torch.Tensor([offset_x * instance_id_in_category, offset_y, 0.]).to(device=device, dtype=dtype))

            droid_slam_aligned_tform_droid_slam = tform4x4(offset_aligned_tform_aligned, droid_slam_aligned_tform_droid_slam)

            #droid_slam_labeled_cuboid_tform_droid_slam_instance = tform4x4(droid_slam_labeled_cuboid_tform_droid_slam,
            #                                                               all_pred_ref_tform_src[category][0, instance_id_in_category])

            pts3d.append(transf3d_broadcast(
                pts3d=sequences[instance_id].get_pcl(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN).to(device=device, dtype=dtype),
                transf4x4=droid_slam_aligned_tform_droid_slam))

            pts3d_colors.append(
                sequences[instance_id].get_pcl_colors(pcl_source=PCL_SOURCES.DROID_SLAM_CLEAN).to(device=device, dtype=dtype))

            if mesh_source == CUBOID_SOURCES.DEFAULT:
                mesh_verts_mask = sequences_mesh_ids_for_verts == instance_id
                offset_aligned_tform_aligned = tform4x4_from_transl3d(
                    torch.Tensor([0., 0., offset_z]).to(device=device, dtype=dtype))
                droid_slam_aligned_tform_droid_slam = tform4x4(offset_aligned_tform_aligned,
                                                               droid_slam_aligned_tform_droid_slam)

                #mesh_verts = meshes.get_verts_with_mesh_id(mesh_id=instance_id)
                meshes.verts[mesh_verts_mask] = transf3d_broadcast(pts3d=meshes.verts[mesh_verts_mask] .to(device=device, dtype=dtype),
                                                                   transf4x4=droid_slam_aligned_tform_droid_slam)

        # category_meshes = meshes.get_meshes_with_ids(meshes_ids=category_instance_ids)

    viewpoints_count = 2
    # show_scene(pts3d=pts3d, pts3d_colors=pts3d_colors, device=device,
    #            meshes=None, # category_meshes,
    #            meshes_add_translation=False, pts3d_add_translation=False,
    #            return_visualization=False, viewpoints_count=viewpoints_count)

    show_scene(pts3d=pts3d, pts3d_colors=pts3d_colors, device=device,
               meshes=meshes,
               meshes_add_translation=False, pts3d_add_translation=False,
               return_visualization=False, viewpoints_count=viewpoints_count)
"""

@app.command()
def temps_weight_ablation():
    from od3d.datasets.co3d.enum import MAP_CATEGORIES_OD3D_TO_CO3D
    from od3d.cli.benchmark import get_dataframe
    # dino_vits8_acc_dist_appear_weight_05_cyclic_temp_10
    #dinov2_vitb14_acc_dist_appear_weight_05_cyclic_temp_07
    name_regex = '12-[12][90].*_CO3D_NeMo_Align3D_geo.*_slurm'
    name_regex = '01-1[89].*_CO3D_NeMo_Align3D_dino_vits8_acc_dist_appear_weight.*_slurm'
    piDiv = 6
    align3d_df = get_dataframe(configs=['ablation_name', 'method.dist_appear_weight', 'method.app_cyclic_weight_temp', 'method.geo_cyclic_weight_temp'],
                                    metrics=['pose/acc_pi6', 'pose/acc_pi18'],
                                    name_regex=name_regex)


    dist_appear_weight = align3d_df['method.dist_appear_weight'].to_numpy()

    x = align3d_df['method.geo_cyclic_weight_temp'].to_numpy()
    y = align3d_df['method.app_cyclic_weight_temp'].to_numpy()
    import numpy as np
    x_log = torch.from_numpy(x) # torch.log(torch.from_numpy(x)) / torch.log(torch.Tensor([10])).numpy()
    y_log = torch.from_numpy(y) # torch.log(torch.from_numpy(y)) / torch.log(torch.Tensor([10])).numpy()
    xi = np.sort(np.unique(x))
    yi = np.sort(np.unique(y))
    grid_xiyi = np.stack(np.meshgrid(xi, yi))
    logger.info(grid_xiyi)
    mask_grid_xiyi = (x[:, None] == grid_xiyi[0].flatten()[None, : ]) * (y[:, None] == grid_xiyi[1].flatten()[None, :])
    xi_log = np.sort(np.unique(x_log)) #  np.arange(x.min(), x.max(), 0.01)
    yi_log = np.sort(np.unique(y_log))

    z = align3d_df[f'pose/acc_pi{piDiv}'].to_numpy() # 'pose/acc_pi6' 'pose/acc_pi18'
    z_max = (z[:, None] * mask_grid_xiyi).max(axis=0).reshape(grid_xiyi.shape[1:])
    dist_appear_weight_max = dist_appear_weight[(z[:, None] * mask_grid_xiyi).argmax(axis=0)].reshape(grid_xiyi.shape[1:])
    x_max = x[(z[:, None] * mask_grid_xiyi).argmax(axis=0)].reshape(grid_xiyi.shape[1:])
    y_max = y[(z[:, None] * mask_grid_xiyi).argmax(axis=0)].reshape(grid_xiyi.shape[1:])
    xi_unique = xi.copy()
    yi_unique = yi.copy()
    xi, yi = np.meshgrid(xi, yi)
    xi_log, yi_log = np.meshgrid(xi_log, yi_log)

    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    #matplotlib.use("TkAgg")


    fig, ax = plt.subplots(2, 1, figsize=(8, 6))  # (subplot_kw={"projection": "3d"})

    aspect_ratio = 0.5

    font_size = 16
    # Create a heatmap using imshow
    im = ax[0].imshow(z_max, cmap='viridis', interpolation='nearest', aspect='auto') # 'auto'
    im2 = ax[1].imshow(dist_appear_weight_max, cmap='viridis', interpolation='nearest', aspect='auto') # 'auto'

    ax[0].tick_params(axis='y', labelsize=font_size)
    ax[0].tick_params(axis='x', labelsize=font_size)
    ax[1].tick_params(axis='y', labelsize=font_size)
    ax[1].tick_params(axis='x', labelsize=font_size)

    ax[0].set_xlabel(r'geo. temp. ($\log_{10}(\tau{})$)', fontsize=font_size)
    ax[0].set_ylabel(r'app. temp. ($\log_{10}(\tau{})$)', fontsize=font_size)
    ax[1].set_xlabel(r'geo. temp. ($\log_{10}(\tau{})$)', fontsize=font_size)
    ax[1].set_ylabel(r'app. temp. ($\log_{10}(\tau{})$)', fontsize=font_size)
    ax[0].set(xticks=np.arange(z_max.shape[1]), xticklabels=xi_unique) #np.round(np.linspace(xi_log.min(), xi_log.max(), z_max.shape[1]), decimals=1))
    ax[0].set(yticks=np.arange(z_max.shape[0]), yticklabels=yi_unique) #np.round(np.linspace(yi_log.min(), yi_log.max(), z_max.shape[0]), decimals=0))
    ax[1].set(xticks=np.arange(z_max.shape[1]), xticklabels=xi_unique) #np.round(np.linspace(xi_log.min(), xi_log.max(), z_max.shape[1]), decimals=1))
    ax[1].set(yticks=np.arange(z_max.shape[0]), yticklabels=yi_unique) #np.round(np.linspace(yi_log.min(), yi_log.max(), z_max.shape[0]), decimals=0))

    # Add colorbar to the right of the plot
    cbar = fig.colorbar(im, ax=ax[0])  # , shrink='auto')
    cbar.ax.tick_params(labelsize=font_size)

    # Add colorbar to the right of the plot
    cbar = fig.colorbar(im2, ax=ax[1])  # , shrink='auto')
    cbar.ax.tick_params(labelsize=font_size)

    #max_point = np.unravel_index(np.argmax(z), z.shape)
    #max_x, max_y, max_z = x[max_point], y[max_point], z[max_point]
    max_coordinates = np.unravel_index(z_max.argmax(), z_max.shape)
    max_x = xi_log[max_coordinates]
    max_y = yi_log[max_coordinates]
    max_z = z_max[max_coordinates]
    max_dist_appear_weight_max = dist_appear_weight_max[max_coordinates]
    max_coordinates = (max_coordinates[1], max_coordinates[0])

    # Plot the point using scatter
    ax[0].scatter(*max_coordinates, color='red', marker='o', label='max')

    # Annotate the point with a description
    desc = 'PI/6=' + f'{max_z*100:.1f}%'
    ax[0].annotate(desc, max_coordinates, textcoords="offset points", xytext=(0, 10), ha='center', fontsize=font_size,
                color='red')

    # Plot the point using scatter
    ax[1].scatter(*max_coordinates, color='red', marker='o', label='max')
    # Annotate the point with a description
    #desc = r'$\log_{10}(\tau{})=$' + f'{max_x:.1f}' + r' , $\log_{10}(\tau{})=$' + f'{max_y:.1f}' + r', $\alpha{}$=' + f'{max_dist_appear_weight_max*100:.1f}%'
    desc = r'$\alpha{}$=' + f'{max_dist_appear_weight_max*100:.1f}%'
    ax[1].annotate(desc, max_coordinates, textcoords="offset points", xytext=(0, 10), ha='center', fontsize=font_size,
                color='red')

    plt.tight_layout()
    plt.savefig('ablation_dist_cycle.png')

    # plt.show()
    # img = get_img_from_plot(ax=ax[0], fig=fig, axis_off=False)
    # show_img(img, height=1080, width=1980, fpath='ablation_dist.png')
    # show_img(img, height=1080, width=1980)
    # plt.show()

@app.command()
def temp_weight_ablation():
    from od3d.datasets.co3d.enum import MAP_CATEGORIES_OD3D_TO_CO3D
    from od3d.cli.benchmark import get_dataframe

    categories = od3d.io.read_config_intern(Path('datasets/categories/zsp.yaml'))
    # categories = TABLE_CATEGORIES_CO3D_28[:-1]
    #categories = ['bottle', 'couch', 'motorcycle', 'laptop']
    # align3d_1on1_name_partial = '11-13_1[789].*CO3Dv1_NeMo_Align3D_dist_appear_weight.*'
    # align3d_1on1_name_partial = '11-14_1[78].*CO3D_NeMo_Align3D_dist_appear_weight.*'
    align3d_1on1_name_partial = '11-14_2[012].*CO3D_NeMo_Align3D_dist_appear_weight.*'

    name_regex = '12-[12][90].*_CO3D_NeMo_Align3D_geo.*_slurm'
    name_regex = '01-1[89].*_CO3D_NeMo_Align3D_dino_vits8_acc_dist_appear_weight.*_slurm'
    name_regex = '01-1[89].*_CO3D_NeMo_Align3D_dinov2_vitb14_acc_dist_appear_weight.*_slurm'
    name_regex = '01-22.*_CO3D_NeMo_Align3D_dinov2_vits14_no_norm.*_slurm'
    name_regex = '01-22.*_CO3D_NeMo_Align3D_dino_vits8_no_norm.*_slurm'
    name_regex = '01-2[23].*_CO3D_NeMo_Align3D_dinov2_vitb14_no_norm.*_slurm'
    name_regex = '01-23.*_CO3D_NeMo_Align3D_dist_appear_weight.*_slurm'

    piDiv = 6 # 6 18

    # align3d_1on1_name_partial = '.*_CO3D_NeMo_Align3D_cyclic_temp_.*_dist_appear_weight_.*_slurm'

    align3d_1on1_metrics = ['pose/acc_pi6', 'pose/acc_pi18']
    align3d_1on1_columns_map = {}
    align3d_1on1_columns_map[align3d_1on1_metrics[-2]] = "Acc. Pi/6. [%]"
    align3d_1on1_columns_map[align3d_1on1_metrics[-1]] = "Acc. Pi/18. [%]"
    # age_in_hours = None
    configs = ['ablation_name', 'method.dist_appear_weight', 'method.geo_cyclic_weight_temp']
    #configs = ['method.dist_appear_weight']
    #align3d_1on1_columns_map['method.dist_appear_weight'] = 'Appear. Weight'
    align3d_1on1_columns_map['ablation_name'] = 'Name'
    for category in categories:
        align3d_1on1_metrics.append(f'pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi6')
        align3d_1on1_metrics.append(f'pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi18')
        align3d_1on1_columns_map[align3d_1on1_metrics[-2]] = category
        align3d_1on1_columns_map[align3d_1on1_metrics[-1]] = category

    align3d_1on1_df = get_dataframe(configs=configs, metrics=align3d_1on1_metrics, name_regex=name_regex)

    x = align3d_1on1_df['method.dist_appear_weight'].to_numpy()
    y = align3d_1on1_df['method.geo_cyclic_weight_temp'].to_numpy()
    z = align3d_1on1_df[f'pose/acc_pi{piDiv}'].to_numpy() # 'pose/acc_pi6' 'pose/acc_pi18'
    # z = (align3d_1on1_df[f'pose/acc_pi6'].to_numpy() + align3d_1on1_df[f'pose/acc_pi18'].to_numpy()) / 2.# 'pose/acc_pi6' 'pose/acc_pi18'

    import matplotlib.pyplot as plt
    import numpy as np

    from matplotlib import cm

    from scipy.interpolate import griddata
    import matplotlib
    matplotlib.use("TkAgg")

    # y = (torch.log(torch.from_numpy(y)) / torch.log(torch.Tensor([10]))).numpy()
    y = y
    xi = np.sort(np.unique(x)) #  np.arange(x.min(), x.max(), 0.01)
    yi = np.sort(np.unique(y))
    xi_unique = xi.copy()
    yi_unique = yi.copy()
    #  np.arange(x.min(), x.max(), 0.01)
    xi, yi = np.meshgrid(xi, yi)
    #yi_log = torch.log(torch.from_numpy(yi)) / torch.log(torch.Tensor([10])).numpy()

    zi = griddata((x, y), z, (xi, yi),  method='linear') #


    fig, ax = plt.subplots(1, 1, figsize=(8, 3))  # (subplot_kw={"projection": "3d"})

    aspect_ratio = 0.5

    font_size = 16
    # Create a heatmap using imshow
    im = ax.imshow(zi, cmap='viridis', interpolation='nearest', aspect='auto') # 'auto'

    ax.tick_params(axis='y', labelsize=font_size)
    ax.tick_params(axis='x', labelsize=font_size)

    ax.set_xlabel(r'appear. weight ($\alpha{}$)', fontsize=font_size)
    ax.set_ylabel(r'cyclical dist. temp. ($\tau{}$)', fontsize=font_size)
    ax.set(xticks=np.arange(zi.shape[1]), xticklabels=xi_unique) #np.round(np.linspace(xi.min(), xi.max(), zi.shape[1]), decimals=1))
    ax.set(yticks=np.arange(zi.shape[0]), yticklabels=yi_unique) #np.round(np.linspace(yi.min(), yi.max(), zi.shape[0]), decimals=0))

    # Add colorbar to the right of the plot
    cbar = fig.colorbar(im, ax=ax)  # , shrink='auto')
    cbar.ax.tick_params(labelsize=font_size)

    #max_point = np.unravel_index(np.argmax(z), z.shape)
    #max_x, max_y, max_z = x[max_point], y[max_point], z[max_point]
    max_coordinates = np.unravel_index(zi.argmax(), zi.shape)
    max_x = xi[max_coordinates]
    max_y = yi[max_coordinates]
    max_z = zi[max_coordinates]
    max_coordinates = (max_coordinates[1], max_coordinates[0])

    # Plot the point using scatter
    ax.scatter(*max_coordinates, color='red', marker='o', label='max')

    # Annotate the point with a description
    desc = r'$\alpha{}=$' + f'{max_x:.1f}' + r' , $\tau{}=$' + f'{max_y:.1f}' + f', PI/{piDiv}=' + f'{max_z*100:.1f}%'
    ax.annotate(desc, max_coordinates, textcoords="offset points", xytext=(100, 10), ha='center', fontsize=font_size,
                color='black')

    plt.tight_layout()
    plt.savefig(f'ablation_dist_{piDiv}.eps')
    plt.show()

    #
    # img = get_img_from_plot(ax=ax, fig=fig, axis_off=False)
    # show_img(img, height=1080, width=1980, fpath='ablation_dist.png')
    # show_img(img, height=1080, width=1980)
    # plt.show()
    #
    # from matplotlib.ticker import LinearLocator
    # # Plot the surface.
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #
    #
    # ax.set_xlabel(r'appear. weight ($\alpha{}$)')
    # ax.set_ylabel(r'cyclical dist. temp. ($\tau{}$)')
    #
    # # Find the coordinates of the maximum point
    # max_point = np.unravel_index(np.argmax(z), z.shape)
    # max_x, max_y, max_z = x[max_point], y[max_point], z[max_point]
    #
    # # Annotate the maximum point
    # ax.scatter(max_x, max_y, max_z+0.01, s=200, color='blue')
    # ax.text(max_x, max_y+0.5, max_z+0.2, r'$\alpha{}=$' + f'{max_x:.1f}' + r' , $\tau{}=$' + f'{max_y:.1f}' + ', PI/6=' + f'{max_z*100:.1f}%', color='black')
    # surf = ax.plot_surface(xi, yi, zi, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    #
    # plt.show()
    # img = get_img_from_plot(ax=ax, fig=fig, axis_off=False)
    # show_img(img, height=1080, width=1980)
    #
    # '11-13_19-07-18_CO3Dv1_NeMo_Align3D_dist_appear_weight_04_dist_cyclic_temp_1_slurm'

@app.command()
def pose_alignment():
    'latest_20_zsp'
    from od3d.datasets.co3d.enum import PCL_SOURCES
    from od3d.cv.geometry.transform import transf3d_broadcast, transf3d
    from od3d.cv.geometry.downsample import random_sampling
    logging.basicConfig(level=logging.INFO)
    dtype = torch.float
    if torch.cuda.is_available():
       device = 'cuda'
    else:
        device = 'cpu'

    viewpoint_id = 0
    pad = 0
    categories = ['backpack', 'car', 'chair', 'keyboard', 'laptop', 'motorcycle']
    categories = ['chair', 'bicycle', 'teddybear', 'car'] #  'car',

    categories = ['backpack', 'keyboard', 'laptop', 'motorcycle']
    categories = ['truck', 'train', 'bus', 'handbag']
    categories = ['remote', 'airplane', 'toilet', 'hairdryer']
    categories = ['mouse', 'toaster', 'hydrant', 'book']

    categories = ['mouse', 'toaster', 'hydrant', 'book', 'remote', 'airplane', 'toilet', 'hairdryer', 'truck', 'train', 'bus', 'handbag', 'backpack', 'keyboard', 'laptop', 'motorcycle']
    # categories = ['toaster', 'hydrant', 'airplane', 'hairdryer'] # toaster
    categories = ['handbag', 'laptop', 'motorcycle', 'hydrant', 'airplane', 'hairdryer'] #  'handbag', 'backpack', 'keyboard',] # #v'backpack'

    # categories = ['mouse', 'toaster', 'hydrant', 'book']

        # aligned_name = 'latest_20_zsp/r0'
        # aligned_name = 'latest_20_zsp/r0'
        # aligned_name = 'zsp/r0'
        # backpack car chair keyboard laptop motorcycle


    imgs_categories = []
    for category in categories:

        imgs_category = []
        for dataset_name in ['ours_zsp/r4', 'zsp/r4']:
            # category = 'car'
            co3d = CO3D.create_by_name('co3dv1_10s_zsp_labeled_cuboid_ref', config={
                'categories': [category], 'aligned_name': dataset_name, # , 'pcl_source': CAM_TFORM_OBJ_SOURCES.PCL,
            })

            sequences = co3d.get_sequences()

            pcls_category = []
            pcls_category_colors = []

            for seq in sequences[:8]:
                pcl = seq.get_pcl(PCL_SOURCES.CO3D).to(device)
                pcl_colors = seq.get_pcl_colors(PCL_SOURCES.CO3D).to(device)
                # show_scene(pts3d=[pcl], pts3d_colors=[pcl_colors]) # meshes=[seq.get_mesh(mesh_source=CUBOID_SOURCES.DEFAULT)]

                pcl, pcl_mask = random_sampling(pcl, 50000, return_mask=True)
                pcl_colors = pcl_colors[pcl_mask]
                aligned_tform_obj = seq.aligned_obj_tform_obj.to(device)
                pcl = transf3d_broadcast(pcl, aligned_tform_obj)
                pcl = pcl - pcl.mean(dim=0, keepdim=True)
                #img_pcl = show_scene(pts3d=[pcl], pts3d_colors=[pcl_colors], pts3d_add_translation=True, return_visualization=True, viewpoints_count=viewpoint_id+2)[viewpoint_id]
                #img_pcl = crop_white_border_from_img(img_pcl)
                #show_img(img_pcl)

                pcls_category.append(pcl)
                pcls_category_colors.append(pcl_colors)

            pcls_category = torch.stack(pcls_category, dim=0)
            pcls_category_colors = torch.stack(pcls_category_colors, dim=0)

            # aligned_obj_tform_obj
            #aligned_pcl_tform_pcl = torch.stack([sequence.aligned_obj_tform_obj  for sequence in sequences], dim=0).to(device)
            #pcls_aligned = transf3d_broadcast(pcls, aligned_pcl_tform_pcl[:, None])

            img_category = show_scene(pts3d=pcls_category, pts3d_colors=pcls_category_colors, pts3d_add_translation=True, return_visualization=True, viewpoints_count=viewpoint_id+2,
                                      W=WINDOW_WIDTH * 4, H=WINDOW_HEIGHT * 4 )[viewpoint_id]
            img_category = crop_white_border_from_img(img_category)

            imgs_category.append(img_category)

            # show_img(img_category)
            #imgs_categories.append(img_category)

        W = imgs_category[0].shape[-1]
        H = imgs_category[0].shape[-2]

        imgs_category = [resize(img_category, scale_factor=W / img_category.shape[-1]) for img_category in imgs_category]
        imgs_category = [resize(img_category, H_out=img_category.shape[-2], W_out=W) for img_category in
                           imgs_category]

        # imgs_category = [resize(img_category, H_out=H, W_out=W) for img_category in imgs_category]

        img_category = torch.cat(imgs_category, dim=-2)
        imgs_categories.append(img_category)
        # show_img(img_category)

    W = imgs_categories[0].shape[-1]
    imgs_categories = [resize(img_categories, scale_factor=W / img_categories.shape[-1]) for img_categories in imgs_categories]
    imgs_categories = [resize(img_categories, H_out=img_categories.shape[-2], W_out=W) for img_categories in imgs_categories]


    H_max = max([img_categories.shape[-2] for img_categories in imgs_categories])
    img_categories_placeholder = torch.ones(size=(3, H_max, W)).to(device=device)
    imgs_categories_final = []
    for img_categories in imgs_categories:
        img_categories_final = img_categories_placeholder.clone()
        H = img_categories.shape[-2]
        img_categories_margin = (H_max - H) // 2
        img_categories_final[:, img_categories_margin:img_categories_margin+H, :] = img_categories
        imgs_categories_final.append(img_categories_final)

    imgs_categories_final = torch.stack(imgs_categories_final, dim=0)
    rows = 3
    imgs_categories_final = imgs_categories_final.reshape(rows, len(imgs_categories_final) // rows, * (imgs_categories_final.shape[1:]))

    imgs_categories_final = [crop_white_border_from_img(img_row.permute(1, 2, 0, 3).reshape(3, H_max, -1)) for img_row in imgs_categories_final]

    img_categories_final = torch.cat(imgs_categories_final, dim=-2)

    #imgs_categories_final = torch.cat(imgs_categories_final, dim=-2)
    #img_categories_final = imgs_to_img(imgs_categories_final, pad=0, pad_value=1.)
    show_img(img_categories_final, fpath='alignment.png', height=WINDOW_HEIGHT, width=WINDOW_WIDTH)
    show_img(img_categories_final, height=WINDOW_HEIGHT, width=WINDOW_WIDTH)

    # img_category = imgs_to_img(imgs_category[:, None], pad=pad, pad_value=1.)

    # nemo3d_align_config = OmegaConf.create({
    #     'ransac': {'samples': 1000, 'score_perc': 1.},
    #     'cyclic_weight_temp': 100.,
    #     'dist_appear_weight': 0.2
    # })
    # for category in categories:
    #     dataset_src =
    #     dataset_ref =
    #     src_sequences = dataset_src.get_sequences()
    #     ref_sequences = dataset_ref.get_sequences()
    #
    #     src_instance_ids = torch.LongTensor(list(range(src_instances_count)))
    #     ref_instance_ids = torch.LongTensor(list(range(ref_instances_count)))
    #
    #     src_mesh_ids = src_instance_ids[src_map_seq_to_cat == cat_id]
    #     ref_mesh_ids = ref_instance_ids[ref_map_seq_to_cat == cat_id]
    #
    #     for r, ref_mesh_id in enumerate(ref_mesh_ids):
    #         for s, src_mesh_id in enumerate(src_mesh_ids):
    #             # src_sequences[src_mesh_id].show(show_imgs=True)
    #             src_vertices_mask = src_sequences_mesh_ids_for_verts == src_mesh_id
    #             # src_vertices = torch.arange(src_vertices_count).to(device=self.device)[src_vertices_mask]
    #             pts_src = src_meshes.verts[src_vertices_mask].clone()
    #
    #
    #             logger.info(f'category: {category}, pts-src: {pts_src.shape}, pts-ref: {pts_ref.shape}')
    #
    #             dist_src_ref = src_sequences[src_mesh_id].get_dist_verts_mesh_feats_to_other_sequence(
    #                 ref_sequences[ref_mesh_id]).to(device=device, dtype=dtype)
    #
    #             # division by two to normalize to 0. - 1.
    #             dist_src_ref = dist_src_ref / 2.
    #
    #             # four points required, otherwise rotation yields an ambiguity. like planes without normals
    #             ref_tform4x4_src = ransac(pts=pts_src,
    #                                       fit_func=partial(fit_tform4x4, pts_ref=pts_ref,
    #                                                        dist_ref=dist_src_ref),
    #                                       score_func=partial(score_tform4x4_fit, pts_ref=pts_ref,
    #                                                          dist_ref=dist_src_ref,
    #                                                          dist_appear_weight=nemo3d_align_config.dist_appear_weight,
    #                                                          cyclic_weight_temp=nemo3d_align_config.cyclic_weight_temp,
    #                                                          score_perc=nemo3d_align_config.ransac.score_perc),
    #                                       fits_count=1000, fit_pts_count=4)
    #             # else:
    #             #    ref_tform4x4_src = torch.eye(4).to(device=self.device, dtype=dtype)
    #             _, pose_dist_geo, pose_dist_appear = score_tform4x4_fit(pts=pts_src, tform4x4=ref_tform4x4_src[None,], pts_ref=pts_ref,
    #                                                                     dist_ref=dist_src_ref, return_dists=True,
    #                                                                     dist_appear_weight=nemo3d_align_config.dist_appear_weight,
    #                                                                     cyclic_weight_temp=nemo3d_align_config.cyclic_weight_temp,
    #                                                                     score_perc=nemo3d_align_config.ransac.score_perc)
    #             all_pred_pose_dist_geo[category][r, s] = pose_dist_geo
    #             all_pred_pose_dist_appear[category][r, s] = pose_dist_appear
    #
    #             pred_ref_tform_src = ref_tform4x4_src.clone()
    #             # pred_ref_tform_src[:3, :3] /= torch.linalg.norm(pred_ref_tform_src[:3, :3], dim=-1, keepdim=True)
    #             all_pred_ref_tform_src[category][r, s] = pred_ref_tform_src

@app.command()
def pose_in_the_wild():
    logging.basicConfig(level=logging.INFO)
    device = 'cuda'
    dtype = torch.float

    run_name = '11-11_20-30-50_CO3D_NeMo_cat1_bicycle_ref4_filtered_mesh_slurm'
    mesh_fpath = Path('/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned/all_20s_to_5s_mesh/r4/mesh/bicycle/mesh.ply')
    aligned_name = 'all_20s_to_5s_mesh/r4'
    # od3d bench rsync -r 11-11_20-30-50_CO3D_NeMo_cat1_bicycle_ref4_filtered_mesh_slurm

    pad = 5
    pad_sample = 5
    max_frames_count_per_category = 4 # 28
    batch_size = 4 # 4 0.22 versus 10.92 s,
    add_zsp = True # run zsp, takes long time
    threshold = torch.pi / 18

    # 'couch', 'microwave',
    categories = ['bicycle', 'car', 'motorcycle', 'couch', 'microwave', 'bench', 'chair']
    categories = ['car', 'motorcycle', 'bench', 'chair',]
    categories = ['toybus', 'bicycle', 'couch', 'microwave', ]
    # categories = ['toybus']

    run_names = {
        'bicycle': '11-11_20-30-50_CO3D_NeMo_cat1_bicycle_ref4_filtered_mesh_slurm',
        'car': '11-14_23-52-28_CO3D_NeMo_cat1_car_ref3_filtered_mesh_slurm',
        'motorcycle': '11-14_05-02-27_CO3D_NeMo_cat1_motorcycle_ref3_filtered_mesh_slurm',
        'couch': '11-14_23-49-27_CO3D_NeMo_cat1_couch_ref1_filtered_mesh_slurm',
        'microwave': '11-15_03-09-43_CO3D_NeMo_cat1_microwave_ref3_filtered_mesh_slurm',
        'bench': '11-15_02-01-39_CO3D_NeMo_cat1_bench_ref0_filtered_mesh_slurm',
        'toaster': '11-15_03-11-37_CO3D_NeMo_cat1_toaster_ref3_filtered_mesh_slurm',
        'chair': '11-15_03-12-23_CO3D_NeMo_cat1_chair_ref1_filtered_mesh_slurm',
        #'toybus': '11-14_23-47-12_CO3D_NeMo_cat1_bus_ref4_filtered_mesh_slurm',
        # 'toybus': '11-14_23-46-04_CO3D_NeMo_cat1_bus_ref2_filtered_mesh_slurm',
        'toybus': '11-14_23-45-52_CO3D_NeMo_cat1_bus_ref1_mesh_slurm',
    }

    ref_ids = {
        'bicycle': 4,
        'car': 3,
        'motorcycle': 3,
        'couch': 1,
        'microwave': 3,
        'bench': 0,
        'toaster': 3,
        'chair': 1,
        #'toybus': 2,
        #'toybus': 4,
        'toybus': 1,
    }

    objectnet3d_frames_per_category = {
        'bicycle': {'test': None},
        'car': {'test': None},
        'motorcycle': {'test': None},
        'couch': {'test': None},
        'microwave': {'test': None},
        'bench': {'test': None},
        'toaster': {'test': None},
        'chair': {'test': None},
        'toybus': {'test': None},
    }

    objectnet3d_frames_per_category = {
        'bicycle': {'test': [
            'n02834778_10025',
            'n02834778_10058',
            'n02834778_10129',
            'n02834778_10218',
            'n02834778_10227',
            'n02834778_10327',
            'n02834778_10363',
            'n02834778_10619',
            'n02834778_1107',
            'n02834778_11107',
            # 'n00000004_520',
            # 'n02814533_10069',
            # 'n02814533_10290',
            # 'n02814533_10594',
            # 'n02814533_10637',
            # 'n02814533_10995',
            # 'n02814533_10999',
            # 'n02814533_11124',
            # 'n02814533_11667',
            # 'n02814533_11762',
        ]},
        'car': {'test':[
                'n02814533_10329',
                'n02814533_10818',
                'n02814533_10995',
                'n02814533_11056',
                # 'n02814533_10069',
                # 'n02814533_10329',
                # 'n02814533_10818',
                # 'n02814533_10911',
                # 'n02814533_10995',
                # 'n02814533_10999',
                # 'n02814533_11051',
                # 'n02814533_11056',
                # 'n02814533_11124',
                # 'n02814533_11246',
        ]},
        'motorcycle': {'test': [
            'n03790512_10075',
            'n03790512_10667',
            'n03790512_10965',
            'n03790512_10970',
                # 'n03790512_10075',
                # 'n03790512_10146',
                # 'n03790512_10269',
                # 'n03790512_10547',
                # 'n03790512_10652',
                # 'n03790512_10662',
                # 'n03790512_10667',
                # 'n03790512_10738',
                # 'n03790512_10965',
                # 'n03790512_10970',
        ]},
        'couch': {'test': None},
        'microwave': {'test': [
                'n03761084_10021',
                'n03761084_10024',
                'n03761084_10099',
                'n03761084_10101',
                'n03761084_10108',
                'n03761084_10145',
                'n03761084_10149',
                'n03761084_10391',
                'n03761084_10396',
                'n03761084_10399',
        ]},
        'bench': {'test': [
            'n03891251_1023',
            'n03891251_1024',
            'n03891251_1025',
            'n03891251_1028',
                # 'n03891251_1013',
                # 'n03891251_1018',
                # 'n03891251_1023',
                # 'n03891251_1024',
                # 'n03891251_1025',
                # 'n03891251_1028',
                # 'n03891251_1035',
                # 'n03891251_104',
                # 'n03891251_1040',
                # 'n03891251_1041',
                # # 'n03891251_108',
        ]},
        'toaster': {'test': None},
        'toybus': {'test': None},
        'chair': {'test': [
            'n03001627_130',
            'n03001627_13055',
            'n03001627_1403',
            'n03001627_14224',
                    # 'n03001627_1015',
                    # 'n03001627_1018',
                    # 'n03001627_15080',
                    # 'n03001627_10558',
                    # 'n03001627_1545',
                    # 'n03001627_1560',
                    # 'n03001627_130',
                    # 'n03001627_13055',
                    # 'n03001627_1403',
                    # 'n03001627_14224',
                  ]},
    }


    """
            'motorcycle': {'test':
                           sorted(['n03790512_7678', 'n03791053_14900', 'n03791053_27777', 'n03791053_16363',
                            'n04466871_6266', 'n03790512_489', 'n04466871_10130', 'n03790512_35174',
                            'n03790512_3472', 'n03791053_21599', 'n03790512_8493', 'n03791053_6960',
                            'n03790512_2887', 'n04466871_6000', 'n03790512_22770', 'n03790512_6771',
                            'n03791053_3562', 'n03790512_12168', 'n04466871_13199', 'n03790512_574', ])
                       },
                       """

    imgs_categories = []
    for category in categories:

        ref_id = ref_ids[category]
        run_name = run_names[category]
        aligned_name = f'all_50s_to_5s_mesh_filtered/r{ref_id}'

        #  rsync -avrzP slurm:/work/dlclarge1/sommerl-od3d/exps/11-14_05-02-27_CO3D_NeMo_cat1_motorcycle_ref3_filtered_mesh_slurm /misc/lmbraid19/sommerl/exps
        #  rsync -avrzP slurm:/work/dlclarge1/sommerl-od3d/datasets/CO3D_Preprocess/aligned/all_50s_to_5s_mesh_filtered /misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned
        #  /misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned

        mesh_fpath = Path(f'/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/aligned/{aligned_name}/mesh/{category}/mesh.ply')
        # scale_mask_larger_1_centerzoom512 scale_mask_shorter_1_centerzoom512 scale_mask_separate_centerzoom512
        config_transform = od3d.io.read_config_intern(rfpath=Path("methods").joinpath('transform', f"scale_mask_shorter_1_centerzoom512.yaml"))
        # scale_mask_shorter_1_centerzoom512
        from od3d.datasets.co3d.enum import MAP_CO3D_OBJECTNET3D

        nemo = NeMo.create_by_name('nemo',
                                   logging_dir=Path('nemo_out'),
                                   config={'texture_dataset': None,
                                            'train': {'transform': {'transforms': [config_transform]}},
                                           'categories': [category],
                                           'fpaths_meshes': {category: str(mesh_fpath)},
                                           'checkpoint': f'/misc/lmbraid19/sommerl/exps/{run_name}/nemo.ckpt'})

        config_dataset = {'categories': [MAP_CO3D_OBJECTNET3D[category]]} # , 'dict_nested_frames': {'val': objectnet3d_frames}} # n03792782_6218, n03792782_687
        config_dataset['modalities'] = ['size', 'category', 'cam_intr4x4', 'cam_tform4x4_obj', 'category', 'rgb', 'mask', 'depth', 'depth_mask']

        # config_dataset['dict_nested_frames'] = objectnet3d_frames_per_category[category]

        config_dataset['subset_fraction'] = 1.

        dataset_nemo = ObjectNet3D.create_by_name('objectnet3d_test', config=config_dataset)
        dataset_nemo.transform = nemo.transform_train  # OD3D_Transform.create_by_name('scale_mask_separate_centerzoom512')
        dataloader_nemo = torch.utils.data.DataLoader(dataset=dataset_nemo, batch_size=batch_size, shuffle=False,
                                                 collate_fn=dataset_nemo.collate_fn, num_workers=0)

        if add_zsp:
            zsp = ZSP.create_by_name('zsp', logging_dir=Path('zsp_out'),
                                     config={'use_gt_src': False,
                                             'use_train_only_to_collect_target_data': True
                                             })

            dataset_zsp = ObjectNet3D.create_by_name('objectnet3d_test', config=config_dataset)
            dataset_zsp.transform = zsp.transform_test  # OD3D_Transform.create_by_name('scale_mask_separate_centerzoom512')

            dataloader_zsp = torch.utils.data.DataLoader(dataset=dataset_zsp, batch_size=batch_size, shuffle=False,
                                                          collate_fn=dataset_zsp.collate_fn, num_workers=0)

            co3d_5refs = CO3D.create_by_name('co3d_no_zsp_5s_labeled_ref', config={'categories': [category],
                                                                                   'modalities': ['size', 'category',
                                                                                                  'cam_intr4x4',
                                                                                                  'cam_tform4x4_obj',
                                                                                                  'category', 'rgb',
                                                                                                  'mask',
                                                                                                  'sequence_name',
                                                                                                  'sequence', 'depth',
                                                                                                  'depth_mask']})
            co3d_5refs_dict_category_sequence = {category: [co3d_5refs.dict_category_sequences_names[category][ref_id]]}
            co3d_5refs = co3d_5refs.get_subset_by_sequences(co3d_5refs_dict_category_sequence)
            train_datasets = {
                'src': co3d_5refs,
                'labeled': co3d_5refs,
            }
            zsp.train(datasets_train=train_datasets, datasets_val=None)
        else:
            dataset_zsp = dataset_nemo
            dataloader_zsp = torch.utils.data.DataLoader(dataset=dataset_zsp, batch_size=batch_size, shuffle=False,
                                                         collate_fn=dataset_zsp.collate_fn, num_workers=0)

        imgs_in_the_wild = []
        nemo.net.to(device=device)
        nemo.net.eval()
        # next(nemo.net.parameters()).is_cuda
        # for i, batch in tqdm(enumerate(dataloader_nemo)):
        logger.info('start eval...')
        dataloader_zsp_iter = iter(dataloader_zsp)
        dataloader_nemo_iter = iter(dataloader_nemo)
        name_list = []
        for i in range(len(dataset_zsp)): #  (batch_zsp, batch_nemo) in enumerate(tqdm(zip(dataloader_zsp, dataloader_nemo))):
            batch_zsp = next(dataloader_zsp_iter)
            batch_nemo = next(dataloader_nemo_iter)
            logger.info(batch_zsp.name)
            logger.info(batch_nemo.name)

            name_list.append(batch_nemo.name)

            if torch.cuda.is_available():
                batch_nemo.to(device=device)
                # batch_zsp.to(device=device)


            from od3d.cv.visual.blend import blend_rgb
            from time import time
            time_nemo_start = time()
            batch_res_nemo = nemo.inference_batch_single_view(batch_nemo)
            duration_nemo = time() - time_nemo_start
            logger.info(f'NeMo took {duration_nemo}s, per sample it is {duration_nemo/batch_size}s')
            nemo_pred_cam_tform4x4_obj = batch_res_nemo['cam_tform4x4_obj']
            logger.info(batch_res_nemo['rot_diff_rad'])

            if add_zsp:
                time_zsp_start = time()
                batch_res_zsp = zsp.inference_batch(batch_zsp)
                duration_zsp = time() - time_zsp_start

                logger.info(f'ZSP took {duration_zsp}s, persample it is {duration_zsp/batch_size}s')
                zsp_pred_cam_tform4x4_obj = batch_res_zsp['cam_tform4x4_obj']
                zsp_pred_cam_tform4x4_obj = zsp_pred_cam_tform4x4_obj.to(device=device)
                zsp_pred_cam_tform4x4_obj[..., :3, :3] /= torch.linalg.norm(zsp_pred_cam_tform4x4_obj[..., :3, :3],
                                                                            dim=-1,
                                                                            keepdim=True)
                zsp_pred_cam_tform4x4_obj[..., :3, 3] = batch_nemo.cam_tform4x4_obj[..., :3, 3]

            else:
                zsp_pred_cam_tform4x4_obj = batch_nemo.cam_tform4x4_obj


            nemo_pred_verts_ncds = nemo.get_ncds_with_cam(cam_intr4x4=batch_nemo.cam_intr4x4, cam_tform4x4_obj=nemo_pred_cam_tform4x4_obj,
                                                 categories_ids=batch_nemo.label, size=batch_nemo.size,
                                                 down_sample_rate=1., pre_rendered=False)

            zsp_pred_verts_ncds = nemo.get_ncds_with_cam(cam_intr4x4=batch_nemo.cam_intr4x4, cam_tform4x4_obj=zsp_pred_cam_tform4x4_obj,
                                                 categories_ids=batch_nemo.label, size=batch_nemo.size,
                                                 down_sample_rate=1., pre_rendered=False)

            for b in range(len(batch_nemo)):
                # if batch_res_nemo['']
                if batch_res_nemo['rot_diff_rad'][b] > (threshold):
                    continue
                logger.info(batch_nemo.name_unique[b])
                img_rgb = batch_nemo.rgb[b].clone()

                img_rgb_min = img_rgb.flatten(1).min(dim=-1).values[:, None, None]
                img_rgb_max = img_rgb.flatten(1).max(dim=-1).values[:, None, None]
                img_rgb = (img_rgb - img_rgb_min) / (img_rgb_max - img_rgb_min)

                # img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min())

                img_rgb_nemo_overlay = blend_rgb(resize(img_rgb, scale_factor=1.), nemo_pred_verts_ncds[b], alpha1=0.2, alpha2=0.8)
                img_rgb_zsp_overlay = blend_rgb(resize(img_rgb, scale_factor=1.), zsp_pred_verts_ncds[b], alpha1=0.2, alpha2=0.8)

                img_in_the_wild_single = torch.stack([img_rgb * 255, img_rgb_nemo_overlay, img_rgb_zsp_overlay], dim=0)
                img_in_the_wild_single = imgs_to_img(img_in_the_wild_single[:, None], pad=pad_sample, pad_value=255)

                # img_in_the_wild_single_no_rgb_mask = (img_in_the_wild_single < 0.1).all(dim=0, keepdim=True)
                # img_in_the_wild_single[img_in_the_wild_single_no_rgb_mask.expand(*img_in_the_wild_single.shape)] = 255.

                # show_img(img_in_the_wild_single)
                imgs_in_the_wild.append(img_in_the_wild_single)

            if len(imgs_in_the_wild) >= max_frames_count_per_category:
                break

        od3d.io.write_list_as_yaml(Path(f'pose_in_the_wild_names_{category}_{max_frames_count_per_category}.yaml'), name_list)
        imgs_in_the_wild = torch.stack(imgs_in_the_wild, dim=0)[:max_frames_count_per_category]
        img_in_the_wild = imgs_to_img(imgs_in_the_wild[None, :], pad=pad, pad_value=255)

        imgs_categories.append(img_in_the_wild)
        show_img(img_in_the_wild, width=2 * WINDOW_WIDTH, height=2 * WINDOW_HEIGHT, fpath=f'pose_in_the_wild_{category}_{max_frames_count_per_category}.png')
        # show_img(img_in_the_wild, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    imgs_categories = torch.stack(imgs_categories, dim=0)
    img_categories = imgs_to_img(imgs_categories.reshape(len(imgs_categories)//2, 2, *imgs_categories.shape[1:]))
    show_img(img_categories, width=2 * WINDOW_WIDTH, height=2 * WINDOW_HEIGHT,
             fpath=f'pose_in_the_wild_categories_{max_frames_count_per_category}.png')

    # img_aligned_imgs = resize(img_aligned_imgs, scale_factor=img_videos.shape[-2] / img_aligned_imgs.shape[-2] )
        # img_in_the_wild = resize(img_in_the_wild, scale_factor=img_videos.shape[-2] / img_in_the_wild.shape[-2] )
        # img = torch.cat([img_videos, img_aligned_imgs * 255, img_in_the_wild], dim=-1)
        # img[:, (img == 0).all(dim=0)] = 255