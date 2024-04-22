import logging
import shutil

logger = logging.getLogger(__name__)

import typer
import od3d.io
from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES
from omegaconf import OmegaConf
from pathlib import Path
from od3d.cv.transforms.crop import Crop
from od3d.cv.transforms.centerzoom3d import CenterZoom3D
from od3d.cv.transforms.randomcenterzoom3d import RandomCenterZoom3D
from od3d.cv.transforms.sequential import SequentialTransform

app = typer.Typer()

@app.command()
def classes():
    print(list(OD3D_Dataset.subclasses.keys()))

@app.command()
def visualize_category_frames(
        dataset: str = typer.Option('co3d_no_zsp_1s_labeled_ref', '-d', '--dataset'),
        imgs_count: int = typer.Option(5, '-i', '--imgs-count'),
        viewpoints_count: int = typer.Option(16, '-v', '--viewpoints-count'), # 16
        height: int = typer.Option(1080, '-h', '--height'),
        width: int = typer.Option(1080, '-h', '--height'),
        platform: str = typer.Option('local', '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name].create_from_config(config=config.dataset)
    dataset.visualize_category_frames(imgs_count=imgs_count, viewpoints_count=viewpoints_count, H=height, W=width)

@app.command()
def visualize_category_pcls(
        dataset: str = typer.Option('co3d_no_zsp_1s_labeled_ref', '-d', '--dataset'),
        viewpoints_count: int = typer.Option(16, '-v', '--viewpoints-count'), # 16
        height: int = typer.Option(1080, '-h', '--height'),
        width: int = typer.Option(1080, '-h', '--height'),
        platform: str = typer.Option('local', '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name].create_from_config(config=config.dataset)
    dataset.visualize_category_pcls(viewpoints_count=viewpoints_count, H=height, W=width)

@app.command()
def visualize_category_meshes(
        dataset: str = typer.Option('co3d_no_zsp_1s_labeled_ref', '-d', '--dataset'),
        viewpoints_count: int = typer.Option(16, '-v', '--viewpoints-count'),
        height: int = typer.Option(1080, '-h', '--height'),
        width: int = typer.Option(1080, '-h', '--height'),
        platform: str = typer.Option('local', '-p', '--platform'),
        modalities: str = typer.Option('ncds,nn_geo,nn_app,nn_cycle,cycle_weight,nn_app_cycle_weight', '-m', '--modalities')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name].create_from_config(config=config.dataset)
    dataset.visualize_category_meshes(viewpoints_count=viewpoints_count, H=height, W=width, modalities=modalities.split(','))
#
@app.command()
def save_sequences_as_video(
        dataset: str = typer.Option('co3d_no_zsp_1s_labeled_ref', '-d', '--dataset'),
        platform: str = typer.Option('local', '-p', '--platform'),):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    logger.info(config)

    dataset = OD3D_Dataset.subclasses[config.dataset.class_name].create_from_config(config=config.dataset)
    dataset.save_sequences_as_video()

@app.command()
def sequences(dataset: str = typer.Option('co3d', '-d', '--dataset'),
              dataset_ban: str = typer.Option(None, '-b', '--dataset-ban'),
              platform: str = typer.Option('local', '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name].create_from_config(config=config.dataset)
    sequences_names = dataset.sequences_names
    if dataset_ban is not None:
        config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset_ban])
        dataset_ban = OD3D_Dataset.subclasses[config.dataset.class_name].create_from_config(config=config.dataset)
        sequences_names = list(filter(lambda sname: sname not in dataset_ban.sequences_names, sequences_names))
    sequences_names_as_str = '\n  - '.join(sequences_names) # sequence.pcl_quality_score
    logger.info(f"Dataset sequences names: \n {sequences_names_as_str}")
    for i in range(len(sequences_names)):
        seq = dataset.get_sequence_by_name(sequences_names[i])
        logger.info(f"Sequence {seq.name} {seq.pcl_quality_score}")
    logger.info(f'There are {len(sequences_names)} sequences in the dataset.')


@app.command()
def setup(dataset: str = typer.Option('pascal3d', '-d', '--dataset'),
          platform: str = typer.Option('local', '-p', '--platform'),
          override: bool = typer.Option(False, '-o', '--override'),
          remove_previous: bool = typer.Option(False, '-r', '--remove-previous')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])

    if remove_previous is not None:
        config.dataset.setup.remove_previous = remove_previous
    if override is not None:
        config.dataset.setup.override = override

    OD3D_Dataset.subclasses[config.dataset.class_name].setup(config.dataset)


@app.command()
def preprocess(dataset: str = typer.Option('pascal3d', '-d', '--dataset'),
               platform: str = typer.Option('local', '-p', '--platform'),
               override: bool = typer.Option(None, '-o', '--override'),
               remove_previous: bool = typer.Option(None, '-r', '--remove-previous')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])

    if remove_previous is not None:
        for key in config.dataset.preprocess.keys():
            config.dataset.preprocess[key].remove_previous = remove_previous
    if override is not None:
        for key in config.dataset.preprocess.keys():
            config.dataset.preprocess[key].override = override

    dataset = OD3D_Dataset.subclasses[config.dataset.class_name].create_from_config(config=config.dataset)

# @app.command()
# def record_video(dataset: str = typer.Option('pascal3d', '-d', '--dataset'),
#                  platform: str = typer.Option('local', '-p', '--platform')):
#     logging.basicConfig(level=logging.INFO)
#     config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
#     OD3D_Dataset.subclasses[config.dataset.class_name].record_video(config.dataset)


@app.command()
def extract_meta(dataset: str = typer.Option('pascal3d', '-d', '--dataset'),
                 platform: str = typer.Option('local', '-p', '--platform'),
                 override: bool = typer.Option(None, '-o', '--override'),
                 remove_previous: bool = typer.Option(None, '-r', '--remove-previous')):
    logging.basicConfig(level=logging.INFO)

    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    if remove_previous is not None:
        config.dataset.extract_meta.remove_previous = remove_previous
    if override is not None:
        config.dataset.extract_meta.override = override

    OD3D_Dataset.subclasses[config.dataset.class_name].extract_meta(config.dataset)

@app.command()
def rsync(dataset: str = typer.Option('co3d_only_first', '-d', '--dataset'),
          platform_source: str = typer.Option('local', '-s', '--source'),
          platform_target: str = typer.Option('slurm', '-t', '--target'),):
    logging.basicConfig(level=logging.INFO)
    rsync_raw(dataset=dataset, platform_source=platform_source, platform_target=platform_target)
    rsync_preprocess(dataset=dataset, platform_source=platform_source, platform_target=platform_target)

@app.command()
def rsync_raw(dataset: str = typer.Option('co3d_only_first', '-d', '--dataset'),
          platform_source: str = typer.Option('local', '-s', '--source'),
          platform_target: str = typer.Option('slurm', '-t', '--target'),):
    logging.basicConfig(level=logging.INFO)
    config_source = od3d.io.load_hierarchical_config(platform=platform_source, overrides=["+datasets@dataset=" + dataset])
    config_target = od3d.io.load_hierarchical_config(platform=platform_target, overrides=["+datasets@dataset=" + dataset])

    paths_source = Path(config_source.dataset.path_raw)
    paths_target = Path(config_target.dataset.path_raw)

    source_link = f'{config_source.platform.link}:' if config_source.platform.link != 'local' else ''
    target_link = f'{config_target.platform.link}:' if config_target.platform.link != 'local' else ''

    od3d.io.run_cmd(cmd=f'rsync -avrzP --delete {source_link}{paths_source} {target_link}{paths_target.parent}', live=True, logger=logger)

@app.command()
def rsync_preprocess(dataset: str = typer.Option('co3d_only_first', '-d', '--dataset'),
          platform_source: str = typer.Option('local', '-s', '--source'),
          platform_target: str = typer.Option('slurm', '-t', '--target'),
          rpath: str = typer.Option('', '-r', '--relative-path')):
    logging.basicConfig(level=logging.INFO)
    config_source = od3d.io.load_hierarchical_config(platform=platform_source, overrides=["+datasets@dataset=" + dataset])
    config_target = od3d.io.load_hierarchical_config(platform=platform_target, overrides=["+datasets@dataset=" + dataset])

    paths_source = Path(config_source.dataset.path_preprocess).joinpath(rpath)
    paths_target = Path(config_target.dataset.path_preprocess).joinpath(rpath)

    source_link = f'{config_source.platform.link}:' if config_source.platform.link != 'local' else ''
    target_link = f'{config_target.platform.link}:' if config_target.platform.link != 'local' else ''

    od3d.io.run_cmd(cmd=f'rsync -avrzP --delete {source_link}{paths_source} {target_link}{paths_target.parent}', live=True, logger=logger)

@app.command()
def visualize_categories(dataset: str = typer.Option('coco', '-d', '--dataset'),
              platform: str = typer.Option('local', '-p', '--platform'),
              rfpath: Path = typer.Option(None, '-f', '--rfpath'),
              frames_count_per_category: int = typer.Option(10, '-c', '--count')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    loggging_dir = Path(config.logger.local_dir)



    import torchvision
    H = 128
    W = 128


    from od3d.datasets.co3d.enum import CO3D_CATEGORIES, MAP_CO3D_OBJECTNET3D, MAP_CO3D_PASCAL3D, MAP_CO3D_COCO
    co3d_categories = CO3D_CATEGORIES.list()
    map_co3d_to_dataset = None
    if 'objectnet3d' in config.dataset.name:
        map_co3d_to_dataset = MAP_CO3D_OBJECTNET3D
    elif 'pascal3d' in config.dataset.name:
        map_co3d_to_dataset = MAP_CO3D_PASCAL3D
    elif 'coco' in config.dataset.name:
        map_co3d_to_dataset = MAP_CO3D_COCO
    else:
        map_co3d_to_dataset = None

    if map_co3d_to_dataset is not None:
        dataset_categories = [map_co3d_to_dataset[cat] for cat in co3d_categories  if map_co3d_to_dataset[cat] is not None]
        from omegaconf import open_dict
        with open_dict(config):
            config.dataset.categories = dataset_categories

    dataset = OD3D_Dataset.subclasses[config.dataset.class_name].create_from_config(config=config.dataset)
    dataset.transform = SequentialTransform([
        Crop(H=H, W=W), #
        # RandomCenterZoom3D(H=640, W=800, dist=5., center3d_min=[0., 0., 0.], center3d_max=[0., 0., 0.], apply_txtr=False, config=config.dataset),
        dataset.transform,
    ]
    )
    if rfpath is None:
        rfpath = f'{dataset.name}.png'

    if map_co3d_to_dataset is None:
        dataset_categories = dataset.categories
        all_categories = dataset_categories
    else:
        all_categories = co3d_categories


    dict_imgs_stacked = dataset.get_frames_categories(max_frames_count_per_category=frames_count_per_category)

    if map_co3d_to_dataset is not None:
        dict_imgs_stacked_remapped = {}
        for co3d_cat in co3d_categories:
            if map_co3d_to_dataset[co3d_cat] is not None:
                dict_imgs_stacked_remapped[co3d_cat] = dict_imgs_stacked[map_co3d_to_dataset[co3d_cat]]
        dict_imgs_stacked = dict_imgs_stacked_remapped

    from od3d.cv.visual.show import show_imgs
    from od3d.cv.visual.draw import draw_text_as_img
    dtype = torch.float
    device = 'cpu'
    imgs = []
    for i, category in enumerate(all_categories):
        if i + 1 < len(all_categories):
            text = category
        else:
            text = category + f'\n {len(all_categories)}'
        img_category_text = draw_text_as_img(H=H, W=W, text=text, fontScale=0.6, lineThickness=1).to(dtype=dtype, device=device)
        if category in dict_imgs_stacked.keys():
            imgs_category = dict_imgs_stacked[category]
        else:
            imgs_category = torch.zeros(size=(0, 3, H, W), dtype=dtype, device=device)
        imgs_place_holders = torch.zeros(size=(frames_count_per_category - len(imgs_category), 3, H, W), dtype=dtype, device=device)
        imgs.append(torch.cat([img_category_text[None,], imgs_category, imgs_place_holders], dim=0))

    imgs = torch.stack(imgs, dim=0)
    if rfpath is not None:
        fpath = loggging_dir.joinpath('datasets', 'categories', rfpath)
        logger.info(f'writing image at {fpath}')
        show_imgs(rgbs=imgs, fpath=fpath)
    else:
        show_imgs(rgbs=imgs)

@app.command()
def visualize_sequences(dataset: str = typer.Option('pascal3d', '-d', '--dataset'),
              platform: str = typer.Option('local', '-p', '--platform')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset, "+datasets@dtd=dtd"])
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name].create_from_config(config=config.dataset)
    for sequence in dataset.get_sequences():
        sequence.visualize()

@app.command()
def visualize(dataset: str = typer.Option('pascal3d', '-d', '--dataset'),
              platform: str = typer.Option('local', '-p', '--platform')):
    import torch.utils.data
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset, "+datasets@dtd=dtd"])
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name].create_from_config(config=config.dataset)
    # import torchvision
    # # modalities = [OD3D_FRAME_MODALITIES(mod) for mod in config.dataset.modalities]
    # dataset.transform = SequentialTransform([
    #     #RandomCenterZoom3D(H=640, W=800, dist=25., center3d_min=[0., 0., 0.], center3d_max=[0., 0., 0.], apply_txtr=True, config=config.dtd),
    #     CenterZoom3D(H=640, W=800, scale=1., center_rel_shift_xy=[0., 0.],
    #                        apply_txtr=False, config=config.dtd),
    #
    # ]
    # )
    from od3d.cv.transforms.transform import OD3D_Transform

    #sequences = dataset.get_sequences()
    #for seq in sequences:
    #    logger.info(seq.name_unique)
    #    seq.show(show_imgs=True)
    dataset.transform = OD3D_Transform.create_by_name('centerzoom512')
    # dataset.transform = OD3D_Transform.create_by_name('scale_mask_separate_centerzoom512')
    #dataset.transform = OD3D_Transform.create_by_name('scale_mask_shorter_1_centerzoom512')

    #dataset.transform = OD3D_Transform.create_by_name('scalemask1_centerzoom224')
    #dataset.transform = OD3D_Transform.create_by_name('scalemask1_centerzoom896')


    """
    from od3d.cv.visual.show import show_pcl
    from od3d.cv.geometry.downsample import voxel_downsampling
    from od3d.cv.geometry.transform import transf3d_broadcast
    pcls = []
    for sequence_name in dataset.sequences_names:
        sequence = dataset.get_sequence_by_name(sequence_name)
        pcls.append(transf3d_broadcast(voxel_downsampling(sequence.pcl_clean, K=1000).to('cuda:0'), transf4x4=sequence.cuboid_front_tform4x4_obj.to('cuda:0')))
        # batch[0].sequence_name
        # dataset.visualize(i)
    pcl_max_pts_id = torch.Tensor([pcl.shape[0] for pcl in pcls]).max(dim=0)[1]
    pcls.append(pcls[0])
    pcls[0] = pcls[pcl_max_pts_id]
    show_pcl(pcls)
    """

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    logging.info(f"Dataset contains {len(dataset)} frames.")
    imgs = []
    for batch in iter(dataloader):
        logger.info(f'{batch.name_unique[0]}') #sequence_name[0]}')
        if torch.cuda.is_available():
            batch.to(device='cuda:0')
        img = batch.visualize()
        imgs.append(img)

    from od3d.cv.visual.video import save_gif, save_video
    save_video(imgs=imgs, fpath=Path(config.platform.path_exps).joinpath('videos', dataset.name + '.mkv'))
import http.server
import socketserver
import torchvision
from od3d.io import run_cmd

@app.command()
def label_add_local(dataset: str = typer.Option('co3d_only_first', '-d', '--dataset'),
                 platform: str = typer.Option('local', '-p', '--platform'),
                 name: str = typer.Option('co3d_only_first_2_car', '-n', '--name'),
                 restart: bool = typer.Option(False, '-r', '--restart')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    config.dataset.preprocess = False
    config.dataset.preprocess_cuboids = False
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name].create_from_config(config=config.dataset)
    #logging.info(f"Dataset contains {len(dataset)} frames.")

    path_labelstudio = dataset.path_preprocess.joinpath('labelstudio', )
    path_labelstudio_in = path_labelstudio.joinpath('in')
    path_labelstudio_out = path_labelstudio.joinpath('out')

    map_name_to_id = get_label_projects_map_name_to_id()
    data_arg = f'"path": "{path_labelstudio_in}", "title": "test", "project": "{map_name_to_id[name]}"' # + str(map_name_to_id[name])
    #cmd = "curl -X POST http://localhost:8080/api/storages/localfiles/" + " -H Content-Type:application/json -H 'Authorization: Token 3b92be7993e450f032f653b15c03f157162513cc' --data '{" + data_arg + "}'"
    #run_cmd(cmd, live=True, logger=logger)

    cmd = "curl -X POST http://localhost:8080/api/storages/localfiles/" + str(map_name_to_id[name]) + "/sync/ -H Content-Type:application/json -H 'Authorization: Token 3b92be7993e450f032f653b15c03f157162513cc' --data '{" + data_arg + "}'"
    run_cmd(cmd, live=True, logger=logger)


def get_label_projects_map_name_to_id():
    import json

    tmp_fpath = 'tmp.json'
    cmd = f"curl -X GET http://localhost:8080/api/projects/ -H 'Authorization: Token 3b92be7993e450f032f653b15c03f157162513cc' > {tmp_fpath}"
    run_cmd(cmd, live=True, logger=logger)

    with open(tmp_fpath, mode='r') as f:
        projects = json.load(f)
    # logger.info(projects)

    count = projects['count']
    results = projects['results']
    map_id_to_name = {}
    map_name_to_id = {}
    for i in range(count):
        map_id_to_name[results[i]['id']] = results[i]['title']
        map_name_to_id[results[i]['title']] = results[i]['id']
    return map_name_to_id

def get_local_storages(project_id):
    import json

    cmd = f"curl -X GET http://localhost:8080/api/storages/localfiles?project={project_id} -H 'Authorization: Token 3b92be7993e450f032f653b15c03f157162513cc'"
    res = run_cmd(cmd, live=False, logger=logger)

    local_storages = json.loads(res)
    #logger.info(local_storages)

    return local_storages


@app.command()
def label_export(dataset: str = typer.Option('co3d_only_first', '-d', '--dataset'),
                 platform: str = typer.Option('local', '-p', '--platform'),
                 restart: bool = typer.Option(False, '-r', '--restart'),
                 category: str = typer.Option('car', '-c', '--category')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    config.dataset.preprocess = False
    config.dataset.preprocess_cuboids = False
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name].create_from_config(config=config.dataset)
    project_name = f'{dataset.config.name}_{category}'
    #logging.info(f"Dataset contains {len(dataset)} frames.")

    #path_labelstudio_in = dataset.path_preprocess.joinpath('labelstudio', 'in')
    #path_labelstudio_out = dataset.path_preprocess.joinpath('labelstudio', 'out')


    path_labelstudio_labels = dataset.path_preprocess.joinpath('labels', 'kpts2d_orient')

    if restart and path_labelstudio_labels.exists():
        shutil.rmtree(path_labelstudio_labels)

    if not path_labelstudio_labels.exists():
        path_labelstudio_labels.mkdir(parents=True, exist_ok=True)

    #import json
    #tmp_fpath = 'tmp.json'

    #map_name_to_id = get_label_projects_map_name_to_id()

    frames = label_export_project(project_name=project_name)
    #cmd = f"curl -X GET http://localhost:8080/api/projects/{map_name_to_id[name]}/export?exportType=JSON -H 'Authorization: Token 3b92be7993e450f032f653b15c03f157162513cc' --output {tmp_fpath}"
    ## cmd = f'label-studio export {map_name_to_id[name]} json --data-dir out --export-path {tmp_fpath}'
    #run_cmd(cmd, live=True, logger=logger)
    #
    #with open(tmp_fpath, mode='r') as f:
    #    images = json.load(f)

    kpoints_pairs = {
        'left': 'right',
        'right': 'left',
        'back': 'front',
        'front': 'back',
        'top': 'bottom',
        'bottom': 'top',
    }
    for frame in frames.values():
        # logger.info(label['data']['image'])
        unique_name = frame['data']['image'].split('/')[-1].split('.')[0]
        logger.info(unique_name)
        fpath_label = path_labelstudio_labels.joinpath(unique_name + '.pt')
        l_orients_kpoints_pairs = {'left-right': [], 'back-front' : [], 'top-bottom': []}
        for annots in frame['annotations']:
            count_kpts = 0
            for annot in annots["result"]:
                count_kpts += 1
                if count_kpts % 2 == 1:
                    kpoint1_name = annot["value"]["keypointlabels"][0]
                    kpoint_pair = []
                    kpoint_pair.append(torch.Tensor([annot["value"]["x"]  / 100. * annot["original_width"],  annot["value"]["y"] / 100. * annot["original_height"]]))
                elif count_kpts % 2 == 0:
                    kpoint2_name = annot["value"]["keypointlabels"][0]
                    if kpoint2_name != kpoints_pairs[kpoint1_name]:
                        logger.warning(f"Expected kpoint {kpoints_pairs[kpoint1_name]}, got {kpoint2_name}. Skipping kpoint...")
                        continue
                    kpoint_pair.append(torch.Tensor([annot["value"]["x"] / 100. * annot["original_width"], annot["value"]["y"] / 100. * annot["original_height"]]))
                    if kpoint1_name in ['left', 'back', 'top']:
                        kpoint_pair_name = f'{kpoint1_name}-{kpoint2_name}'
                    else:
                        kpoint_pair_name = f'{kpoint2_name}-{kpoint1_name}'
                        kpoint_pair = kpoint_pair[::-1]
                    kpoint_pair = torch.stack(kpoint_pair, dim=0)
                    l_orients_kpoints_pairs[kpoint_pair_name].append(kpoint_pair)
        for orient in l_orients_kpoints_pairs.keys():
            if len(l_orients_kpoints_pairs[orient]) > 0:
                l_orients_kpoints_pairs[orient] = torch.stack(l_orients_kpoints_pairs[orient], dim=0)
            else:
                l_orients_kpoints_pairs[orient] = torch.Tensor(l_orients_kpoints_pairs[orient])
        logger.info(l_orients_kpoints_pairs)
        torch.save(l_orients_kpoints_pairs, fpath_label)
        #with open(fpath_label, mode='w') as f:
        #    json.dump(label_clean, fp=f)

@app.command()
def label_delete_project(project_name: str = typer.Option(None, '-p', '--project')):
    logging.basicConfig(level=logging.INFO)

    map_name_to_id = get_label_projects_map_name_to_id()
    cmd = f"curl -X DELETE http://localhost:8080/api/projects/{map_name_to_id[project_name]}/ -H 'Authorization: Token 3b92be7993e450f032f653b15c03f157162513cc'"
    # cmd = f'label-studio export {map_name_to_id[name]} json --data-dir out --export-path {tmp_fpath}'
    run_cmd(cmd, live=True, logger=logger)

# @app.command()
def label_export_project(project_name: str): # = typer.Option(None, '-p', '--project')):
    logging.basicConfig(level=logging.INFO)

    import json
    map_name_to_id = get_label_projects_map_name_to_id()

    cmd = f"curl -X GET http://localhost:8080/api/projects/{map_name_to_id[project_name]}/export?exportType=JSON -H 'Authorization: Token 3b92be7993e450f032f653b15c03f157162513cc'" #  --output {tmp_fpath}
    # cmd = f'label-studio export {map_name_to_id[name]} json --data-dir out --export-path {tmp_fpath}'
    res = run_cmd(cmd, live=False, logger=None)
    export_data = json.loads(res)
    #logger.info(export_data)
    frames_names = [Path(frame['data']['image']).name.split('-')[-1] for frame in export_data]
    logger.info(frames_names)


    logger.info(export_data[0].keys())
    return dict(zip(frames_names, export_data))
    # logger.info(export_data[0]['data'])
@app.command()
def label_start(dataset: str = typer.Option('co3d_only_first', '-d', '--dataset'),
                platform: str = typer.Option('local', '-p', '--platform'),
                category: str = typer.Option('car', '-c', '--category'),
                restart: bool = typer.Option(False, '-r', '--restart')):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config(platform=platform, overrides=["+datasets@dataset=" + dataset])
    config.dataset.all_categories = [category]
    dataset = OD3D_Dataset.subclasses[config.dataset.class_name].create_from_config(config=config.dataset)
    logging.info(f"Dataset contains {len(dataset)} frames.")
    project_name = f'{dataset.name}_{category}'

    path_labelstudio = dataset.path_preprocess.joinpath('labelstudio')
    path_labelstudio_in = path_labelstudio.joinpath('in', project_name)
    path_labelstudio_out = path_labelstudio.joinpath('out')
    if restart:
        if path_labelstudio_in.exists():
            shutil.rmtree(path_labelstudio_in)
        if path_labelstudio_out.exists():
            shutil.rmtree(path_labelstudio_out)

    path_labelstudio_labelconfig = dataset.path_preprocess.joinpath('labelstudio', f'label_{category}_config.xml')
    # keypoints = dataset.config.keypoints[category]

    labelconfig_str="""
<View>
<Header value="Select label and click the image to start"/>
  <Image strokeWidth="3" name="image" value="$image" zoom="true"/>
  <KeyPointLabels name="keypoints" toName="image">
    <Label value="left" background="red"/>
    <Label value="right" background="yellow"/>
    <Label value="back" background="green"/>
    <Label value="front" background="orange"/>
    <Label value="top" background="blue"/>
    <Label value="bottom" background="purple"/>
  </KeyPointLabels>
</View>

    """


#     labelconfig_str = """
#         <View>
#         <KeyPointLabels name="kp-1" toName="img-1">
#     """
#     for keypoint in keypoints:
#         labelconfig_str += f'<Label value="{keypoint}"/>' #  background="red"
#     labelconfig_str += """
#   </KeyPointLabels>
#   <Image name="img-1" value="$img" />
# </View>
#     """


    with open(file=path_labelstudio_labelconfig, mode='w') as f:
        f.write(labelconfig_str)

    if not path_labelstudio_in.exists():
        path_labelstudio_in.mkdir(parents=True, exist_ok=True)

    if not path_labelstudio_out.exists():
        path_labelstudio_out.mkdir(parents=True, exist_ok=True)


    for i in range(len(dataset)):
        frame = dataset.__getitem__(i)
        fname = f'{frame.name_unique}{frame.fpath_rgb.suffix}'
        if not path_labelstudio_in.joinpath(fname).exists():
            cmd = f'cp "{frame.fpath_rgb}" "{path_labelstudio_in.joinpath(fname)}"'
            run_cmd(cmd, live=True, logger=logger)

    cmd = f'label-studio init {project_name} --username abc@def.com --password abcdefghj --data-dir {path_labelstudio_out} --label-config {path_labelstudio_labelconfig}'
    run_cmd(cmd, live=True, logger=logger)

    cmd = f'( sleep 10 && od3d dataset label-start-finish -p {project_name} -i {path_labelstudio_in} ) &' # -o {path_labelstudio_out} -c {path_labelstudio_labelconfig} ) &'
    run_cmd(cmd, live=False, logger=logger, background=True)
    # export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=  LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
    cmd = f'export LOCAL_FILES_SERVING_ENABLED=true && export LOCAL_FILES_DOCUMENT_ROOT={path_labelstudio} && label-studio start --data-dir {path_labelstudio_out}'
    run_cmd(cmd, live=True, logger=logger)

@app.command()
def label_start_finish(project_name: str = typer.Option(None, '-p', '--project'),
                       path_labelstudio_in: str = typer.Option(None, '-i', '--path-labelstudio-in'),
                       basis_project_name: str = typer.Option('co3d_only_first_car', '-b', '--basis-project')):
    logging.basicConfig(level=logging.INFO)
    import json
    local_storage_default_name = 'local-default'
    map_name_to_id = get_label_projects_map_name_to_id()
    project_id = map_name_to_id[project_name]
    local_storages = get_local_storages(project_id)
    if path_labelstudio_in is None:
        path_labelstudio_in = Path('/misc/lmbraid19/sommerl/datasets/CO3D_Preprocess/labelstudio/in').joinpath(project_name)
        if not path_labelstudio_in.exists():
            path_labelstudio_in.mkdir(parents=True, exist_ok=True)
    else:
        path_labelstudio_in = Path(path_labelstudio_in)
    local_storages_keys = [local_storage['title'] for local_storage in local_storages]
    logger.info(f"local storages {local_storages_keys}")

    if local_storage_default_name not in local_storages_keys:
        data_arg = f'"path": "{path_labelstudio_in}", "title": "{local_storage_default_name}", "project": "{project_id}"'
        cmd = "curl -X POST http://localhost:8080/api/storages/localfiles/" + " -H Content-Type:application/json -H 'Authorization: Token 3b92be7993e450f032f653b15c03f157162513cc' --data '{" + data_arg + "}'"
        run_cmd(cmd, live=True, logger=logger)

        import time
        time.sleep(3)

        #tmp_fpath = 'tmp.json'

        if basis_project_name is not None:
            basis_frames = label_export_project(project_name=basis_project_name)
        else:
            basis_frames = None
        import_json_content = []
        for path in path_labelstudio_in.iterdir():
            logger.info(path)
            logger.info(path.suffix)
            if path.suffix == '.png' or path.suffix == '.jpg':
                if basis_frames is not None and path.name in basis_frames.keys():
                    frame = basis_frames[path.name]
                else:
                    frame = {}
                frame['data'] = {'image': f'/data/local-files/?d=in/{project_name}/{path.name}'}
                # '/data/local-files/?d=co3d_only_first_car/206_21810_45890_1.jpg'
                import_json_content.append(frame)  # {project_name}/

        cmd = f"curl -X POST 'http://localhost:8080/api/projects/{project_id}/import' -H 'Content-Type: application/json' -H 'Authorization: Token 3b92be7993e450f032f653b15c03f157162513cc' --data '{json.dumps(import_json_content)}'"
        run_cmd(cmd, live=True, logger=logger)

        #logger.info(import_json_content)
        #with open(tmp_fpath, 'w') as fp:
        #    json.dump(import_json_content, fp=fp)

        #cmd = "curl -X POST http://localhost:8080/api/storages/localfiles/" + str(map_name_to_id[project_name]) + "/sync/ -H Content-Type:application/json -H 'Authorization: Token 3b92be7993e450f032f653b15c03f157162513cc' --data '{" + data_arg + "}'"
        #run_cmd(cmd, live=True, logger=logger)
