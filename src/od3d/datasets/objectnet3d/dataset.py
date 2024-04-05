import logging
logger = logging.getLogger(__name__)
import od3d.io
import shutil
# from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.pascal3d.frame import Pascal3DFrameMeta
from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame
from od3d.datasets.objectnet3d.enum import OBJECTNET3D_CATEGORIES, MAP_CATEGORIES_OD3D_TO_OBJECTNET3D, MAP_CATEGORIES_OBJECTNET3D_TO_OD3D, OBJECTNET3D_SCALE_NORMALIZE_TO_REAL
from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
from od3d.datasets.objectnet3d.frame import ObjectNet3D_FrameMeta, ObjectNet3D_Frame # , OD3D_Frame
from tqdm import tqdm
from od3d.cv.geometry.mesh import Meshes
from od3d.datasets.object import OD3D_MESH_TYPES


class ObjectNet3D(OD3D_Dataset):
    map_od3d_categories = MAP_CATEGORIES_OD3D_TO_OBJECTNET3D
    all_categories = list(OBJECTNET3D_CATEGORIES)
    frame_type = ObjectNet3D_Frame
    filter_frames_categorical = False

    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List[str]=None, transform=None, index_shift=0, subset_fraction=1.,
                 dict_nested_frames: Dict=None, dict_nested_frames_ban: Dict=None, filter_frames_categorical=False):
        self.filter_frames_categorical = filter_frames_categorical
        super().__init__(name=name, modalities=modalities, path_raw=path_raw, path_preprocess=path_preprocess,
                 categories=categories, transform=transform, index_shift=index_shift, subset_fraction=subset_fraction,
                 dict_nested_frames=dict_nested_frames, dict_nested_frames_ban=dict_nested_frames_ban)

    def filter_list_frames_unique(self, list_frames_unique):
        list_frames_unique = super().filter_list_frames_unique(list_frames_unique)

        if self.filter_frames_categorical:
            logger.info('filtering frames categorical...')

            allowed_frames_unique = []
            allowed_subsets = set([f_unique.split('/')[0] for f_unique in list_frames_unique])
            for subset in allowed_subsets:
                for category in self.categories:
                    logger.info(f'{subset}, {category}')
                    allowed_frames_unique += self.get_subset_category_names_unique(subset=subset, category=category)
            list_frames_unique_filtered = list(set.intersection(set(allowed_frames_unique), set(list_frames_unique)))
            list_frames_unique_filtered = sorted(list_frames_unique_filtered)
            # for frame_name_unique in tqdm(list_frames_unique):
            #     meta = ObjectNet3D_FrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=frame_name_unique)
            #     if meta.category in self.categories: # filter with only first object's category
            #         list_frames_unique_filtered.append(frame_name_unique)
            #     #if len(set(self.categories).intersection(set(meta.categories))) > 0:
            #     #    list_frames_unique_filtered.append(frame_name_unique)
            list_frames_unique = list_frames_unique_filtered
        return list_frames_unique

    @staticmethod
    def setup(config: DictConfig):
        # logger.info(OmegaConf.to_yaml(config))
        path_raw = Path(config.path_raw)
        if path_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous ObjectNet3D")
            shutil.rmtree(path_raw)

        path_raw.mkdir(parents=True, exist_ok=True)

        dict_name_to_url = {
            "Images": "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_images.zip",
            "Annotations": "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_annotations.zip",
            "CAD": "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_cads.zip",
            "Image_sets": "ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_image_sets.zip",
        }
        for name, url in dict_name_to_url.items():
            fpath=path_raw.joinpath(f'{name}.zip')
            path_dir = path_raw.joinpath(name)

            if path_dir.exists() and not config.setup.override:
                logger.info(f"Found {name} of ObjectNet3D at {path_dir}")
            else:
                od3d.io.download(url=url, fpath=fpath)
                od3d.io.unzip(path_raw.joinpath(f'{name}.zip'), dst=path_raw.joinpath('tmp'))
                od3d.io.move_dir(src=path_raw.joinpath('tmp', 'ObjectNet3D'), dst=path_raw)

    @staticmethod
    def extract_meta(config: DictConfig):
        path = Path(config.path_raw)
        path_meta = ObjectNet3D.get_path_meta(config=config)
        path_raw = Path(config.path_raw)
        rfpath_meshes = Path('CAD').joinpath('off')
        rfpath_annotations = Path('Annotations')
        rfpath_images = Path('Images')

        subsets = ['train', 'test', 'val']
        for subset in subsets:
            path_frames_subset = ObjectNet3D_FrameMeta.get_path_frames_meta_with_subset(path_meta=path_meta, subset=subset)
            if not path_frames_subset.exists() or config.extract_meta.override:
                fpath_image_set_subset = path_raw.joinpath('Image_sets', subset + '.txt')
                frames_str = od3d.io.read_str_from_file(fpath=fpath_image_set_subset)
                logger.info(f'preprocess subset {subset}')
                frames_names = frames_str.split()
                for i in tqdm(range(len(frames_names))):
                    frame_name = frames_names[i]
                    # logger.info(f'preprocess {frame_name}')
                    frame_meta = ObjectNet3D_FrameMeta.load_from_raw(path_raw=path_raw,
                                                                     rfpath_annotations=rfpath_annotations,
                                                                     rfpath_images=rfpath_images,
                                                                     rfpath_meshes=rfpath_meshes, subset=subset,
                                                                     name=frame_name)

                    if frame_meta is not None:
                        frame_meta.save(path_meta=path_meta)
            else:
                logger.info(f'found subset frames at {path_frames_subset}')

    def get_frame_by_name_unique(self, name_unique):
        from od3d.datasets.object import OD3D_CAM_TFORM_OBJ_TYPES, OD3D_FRAME_MASK_TYPES, OD3D_MESH_TYPES, \
            OD3D_MESH_FEATS_TYPES, OD3D_MESH_FEATS_DIST_REDUCE_TYPES, OD3D_FRAME_DEPTH_TYPES, \
            OD3D_TFROM_OBJ_TYPES
        return self.frame_type(path_raw=self.path_raw, path_preprocess=self.path_preprocess,
                               name_unique=name_unique, all_categories=self.categories,
                               mask_type=OD3D_FRAME_MASK_TYPES.MESH,
                               cam_tform4x4_obj_type=OD3D_CAM_TFORM_OBJ_TYPES.META,
                               mesh_type=OD3D_MESH_TYPES.META,
                               mesh_feats_type=OD3D_MESH_FEATS_TYPES.M_DINOV2_VITB14_FROZEN_BASE_NO_NORM_T_CENTERZOOM512_R_ACC,
                               mesh_feats_dist_reduce_type=OD3D_MESH_FEATS_DIST_REDUCE_TYPES.MIN_AVG,
                               modalities=self.modalities,
                               tform_obj_type=OD3D_TFROM_OBJ_TYPES.RAW,
                               depth_type=OD3D_FRAME_DEPTH_TYPES.MESH)

    ##### PREPROCESS
    def preprocess(self, config_preprocess: DictConfig):
        logger.info("preprocess")
        for key in config_preprocess.keys():
            if key == 'cuboid' and config_preprocess.cuboid.get('enabled', False):
                override = config_preprocess.cuboid.get('override', False)
                remove_previous = config_preprocess.cuboid.get('remove_previous', False)
                self.preprocess_cuboid(override=override, remove_previous=remove_previous)
            if key == 'mask' and config_preprocess.mask.get('enabled', False):
                override = config_preprocess.mask.get('override', False)
                remove_previous = config_preprocess.mask.get('remove_previous', False)
                self.preprocess_mask(override=override, remove_previous=remove_previous)
            if key == 'depth' and config_preprocess.depth.get('enabled', False):
                override = config_preprocess.depth.get('override', False)
                remove_previous = config_preprocess.mask.get('remove_previous', False)
                self.preprocess_depth(override=override, remove_previous=remove_previous)
            if key == 'subset_category_names_unique' and config_preprocess.subset_category_names_unique.get('enabled', False):
                override = config_preprocess.subset_category_names_unique.get('override', False)
                remove_previous = config_preprocess.subset_category_names_unique.get('remove_previous', False)
                self.preprocess_subset_category_names_unique(override=override, remove_previous=remove_previous)


    def preprocess_cuboid(self, override=False, remove_previous=False):
        logger.info('preprocess cuboid...')

        scale_objectnet3d_to_od3d = {}
        for category in self.categories:
            if category not in self.all_categories:
                continue
            mesh_types = [OD3D_MESH_TYPES.CUBOID250, OD3D_MESH_TYPES.CUBOID500, OD3D_MESH_TYPES.CUBOID1000]
            for mesh_type in mesh_types:
                fpath_mesh_out = self.path_preprocess.joinpath(ObjectNet3D_Frame.get_rfpath_pp_categorical_mesh(mesh_type=mesh_type, category=category))

                if fpath_mesh_out.exists() and not override:
                    logger.warning(f'mesh already exists {fpath_mesh_out}')
                    return
                else:
                    logger.info(f'preprocessing mesh for {category} with type {mesh_type}')

                import re
                match = re.match(r"([a-z]+)([0-9]+)", mesh_type, re.I)
                if match and len(match.groups()) == 2:
                    mesh_type, mesh_vertices_count = match.groups()
                    mesh_vertices_count = int(mesh_vertices_count)
                else:
                    msg = f'could not retrieve mesh type and vertices count from mesh name {mesh_type}'
                    raise Exception(msg)

                fpaths_meshes_category = [fpath for fpath in self.path_raw.joinpath(ObjectNet3D_Frame.get_rpath_raw_categorical_meshes(category=category)).iterdir()]
                fpaths_meshes_category = [fpath for fpath in fpaths_meshes_category if re.match(r"[0-9][0-9]\.off", fpath.name)]

                meshes = Meshes.load_from_files(fpaths_meshes_category)
                pts3d = meshes.verts

                from od3d.cv.geometry.fit.cuboid import fit_cuboid_to_pts3d
                from od3d.datasets.enum import OD3D_CATEGORIES_SIZES_IN_M
                cuboids, tform_obj = fit_cuboid_to_pts3d(pts3d=pts3d,
                                                 optimize_rot=False,
                                                 optimize_transl=False,
                                                 q=0.95,
                                                 vertices_max_count=mesh_vertices_count,
                                                 optimize_steps=0,
                                                 size=OD3D_CATEGORIES_SIZES_IN_M[MAP_CATEGORIES_OBJECTNET3D_TO_OD3D[category]])

                scale_objectnet3d_to_od3d[category] = tform_obj[:3, :3].norm(dim=-1).mean()

                # show:
                #meshes.verts *= scale_objectnet3d_to_od3d[category]
                #Meshes.load_from_meshes([meshes.get_mesh_with_id(i) for i in range(meshes.meshes_count)] + [cuboids.get_mesh_with_id(0)]).show(meshes_add_translation=False)

                obj_mesh = cuboids.get_mesh_with_id(0)
                obj_mesh.write_to_file(fpath=fpath_mesh_out)

        log_str = '\n'
        for key, val in scale_objectnet3d_to_od3d.items():
            log_str += str(key) + f': {val}, \n'
        logger.info(log_str)


    def preprocess_subset_category_names_unique(self, override=False, remove_previous=False):
        logger.info('preprocess subset_category_names_unique...')
        dict_subset_category_names_unique = {}

        # if self.filter_frames_categorical:
        #     msg = f'Preprocessing requires to not filter categorically. Set `filter_frames_categorical` to `False`'
        #     raise Exception(msg)

        dataset = self.get_subset_with_dict_nested_frames(dict_nested_frames={})
        for frame_id in tqdm(range(len(dataset))):
            frame_meta = self.get_item(frame_id).meta
            if frame_meta.subset not in dict_subset_category_names_unique.keys():
                dict_subset_category_names_unique[frame_meta.subset] = {}
            if frame_meta.category not in dict_subset_category_names_unique[frame_meta.subset]:
                dict_subset_category_names_unique[frame_meta.subset][frame_meta.category] = []
            dict_subset_category_names_unique[frame_meta.subset][frame_meta.category].append(frame_meta.name_unique)

        for subset in dict_subset_category_names_unique.keys():
            for category in dict_subset_category_names_unique[subset]:
                fpath = self.path_preprocess.joinpath('subset_category_names_unique', f'{subset}_{category}.yaml')
                if not fpath.exists() or override:
                    od3d.io.write_list_as_yaml(fpath=fpath, _list=dict_subset_category_names_unique[subset][category])
                else:
                    logger.warning(f'not overriding {fpath}, set override flag if desired.')
    def get_subset_category_names_unique(self, subset: str, category: str):
        fpath = self.path_preprocess.joinpath('subset_category_names_unique', f'{subset}_{category}.yaml')
        if not fpath.exists():
            logger.warning('preprocess subset_category_names_unique first ...')
            return []
        names_unique = od3d.io.read_list_from_yaml(fpath)
        return names_unique
