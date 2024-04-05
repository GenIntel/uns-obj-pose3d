import logging
logger = logging.getLogger(__name__)
import torch.nn
from typing import List, Tuple

from od3d.datasets.dataset import OD3D_Dataset, OD3D_FRAME_MODALITIES
from od3d.datasets.frame import OD3D_FRAME_KPTS2D_ANNOT_TYPES
from od3d.datasets.object import OD3D_FRAME_DEPTH_TYPES
from omegaconf import DictConfig
from pathlib import Path
import od3d.io
import shutil
from tqdm import tqdm
from od3d.cv.geometry.mesh import Meshes
from od3d.cv.geometry.primitives import Cuboids
from od3d.cv.io import save_ply
from od3d.datasets.pascal3d.frame import Pascal3DFrame, Pascal3DFrameMeta
from od3d.datasets.pascal3d.enum import PASCAL3D_CATEGORIES, PASCAL3D_SUBSETS, MAP_CATEGORIES_PASCAL3D_TO_OD3D, PASCAL3D_SCALE_NORMALIZE_TO_REAL, MAP_CATEGORIES_OD3D_TO_PASCAL3D
from typing import Dict
import inspect
from od3d.datasets.object import OD3D_MESH_TYPES


class Pascal3D(OD3D_Dataset):
    map_od3d_categories = MAP_CATEGORIES_OD3D_TO_PASCAL3D
    all_categories = list(PASCAL3D_CATEGORIES)
    frame_type = Pascal3DFrame

    @staticmethod
    def setup(config):
        path_pascal3d_raw = Path(config.path_raw)

        if path_pascal3d_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous Pascal3D+")
            shutil.rmtree(path_pascal3d_raw)

        if path_pascal3d_raw.exists():
            logger.info(f"Found Pascal3D+ dataset at {path_pascal3d_raw}")
        else:
            logger.info(f"Downloading Pascal3D+ dataset at {path_pascal3d_raw}")
            fpath = path_pascal3d_raw.joinpath("pascal3d.zip")
            od3d.io.download(url=config.url_pascal3d_raw, fpath=fpath)
            od3d.io.unzip(fpath=fpath, dst=fpath.parent)
            od3d.io.move_dir(src=fpath.parent.joinpath(Path(config.url_pascal3d_raw).with_suffix("").name),
                             dst=fpath.parent)

    ##### SETUP
    def filter_dict_nested_frames(self, dict_nested_frames):
        dict_nested_frames = super().filter_dict_nested_frames(dict_nested_frames)
        logger.info('filtering frames categorical...')
        dict_nested_frames_filtered = {}
        for subset, dict_category_frames in dict_nested_frames.items():
            dict_nested_frames_filtered[subset] = {}
            for category, list_frames in dict_category_frames.items():
                if category in self.categories:
                    dict_nested_frames_filtered[subset][category] = list_frames

        dict_nested_frames = dict_nested_frames_filtered
        return dict_nested_frames



    #### PREPROCESS META
    @staticmethod
    def extract_meta(config: DictConfig):
        subsets = config.get("subsets", None)
        if subsets is None:
            subsets = PASCAL3D_SUBSETS.list()
        categories = config.get("categories", None)
        if categories is None:
            categories = PASCAL3D_CATEGORIES.list()

        path_raw = Pascal3D.get_path_raw(config=config)
        path_meta = Pascal3D.get_path_meta(config=config)
        rpath_meshes = Pascal3D.get_rpath_meshes()

        if config.extract_meta.remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)

        frames_names = []
        frames_categories = []
        frames_subsets = []
        for subset in subsets:
            for category in categories:
                fpath_frame_names_partial = path_raw.joinpath("Image_sets", f"{category}_imagenet_{subset}.txt")
                with fpath_frame_names_partial.open() as f:
                    frame_names_partial = f.read().splitlines()
                frames_names += frame_names_partial
                frames_categories += [category] * len(frame_names_partial)
                frames_subsets += [subset] * len(frame_names_partial)

        #frames_subsets, frames_categories, frames_names = Pascal3DFrameMeta.get_frames_names_from_subsets_and_cateogories_from_raw(path_pascal3d_raw=path_raw, subsets=subsets, categories=categories)

        if config.get('frames', None) is not None:
            frames_names = list(filter(lambda f: f in config.frames, frames_names))

        for i in tqdm(range(len(frames_names))):
            fpath = path_meta.joinpath(Pascal3DFrameMeta.get_rfpath_from_name_unique(name_unique=
                                                                                     Pascal3DFrameMeta.get_name_unique_from_category_subset_name(subset=frames_subsets[i],
                                                                                                                                                             category=frames_categories[i], name=frames_names[i])))
            if not fpath.exists() or config.extract_meta.override:

                frame_meta = Pascal3DFrameMeta.load_from_raw(frame_name=frames_names[i], subset=frames_subsets[i],
                                                             category=frames_categories[i],
                                                             path_raw=path_raw, rpath_meshes=rpath_meshes)

                if frame_meta is not None:
                    frame_meta.save(path_meta=path_meta)



    ##### PREPROCESS
    def preprocess(self, config_preprocess: DictConfig):
        logger.info("preprocess")
        for key in config_preprocess.keys():
            if key == 'cuboid' and config_preprocess.cuboid.get('enabled', False):
                override = config_preprocess.cuboid.get('override', False)
                remove_previous = config_preprocess.cuboid.get('remove_previous', False)
                self.preprocess_cuboid(override=override, remove_previous=remove_previous)
            elif key == 'mask' and config_preprocess.mask.get('enabled', False):
                override = config_preprocess.mask.get('override', False)
                remove_previous = config_preprocess.mask.get('remove_previous', False)
                self.preprocess_mask(override=override, remove_previous=remove_previous)
            elif key == 'depth' and config_preprocess.depth.get('enabled', False):
                override = config_preprocess.depth.get('override', False)
                remove_previous = config_preprocess.depth.get('remove_previous', False)
                self.preprocess_depth(override=override, remove_previous=remove_previous)

    def preprocess_cuboid(self, override=False, remove_previous=False):
        logger.info('preprocess cuboid...')

        scale_pascal3d_to_od3d = {}
        for category in self.categories:
            if category not in self.all_categories:
                continue
            mesh_types = [OD3D_MESH_TYPES.CUBOID250, OD3D_MESH_TYPES.CUBOID500, OD3D_MESH_TYPES.CUBOID1000]
            for mesh_type in mesh_types:
                fpath_mesh_out = self.path_preprocess.joinpath(Pascal3DFrame.get_rfpath_pp_categorical_mesh(mesh_type=mesh_type, category=category))

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

                fpaths_meshes_category = [fpath for fpath in self.path_raw.joinpath(Pascal3DFrame.get_rpath_raw_categorical_meshes(category=category)).iterdir()]
                meshes = Meshes.load_from_files(fpaths_meshes_category)
                meshes.verts.data = meshes.verts
                pts3d = meshes.verts

                from od3d.cv.geometry.fit.cuboid import fit_cuboid_to_pts3d
                from od3d.datasets.enum import OD3D_CATEGORIES_SIZES_IN_M

                cuboids, tform_obj = fit_cuboid_to_pts3d(pts3d=pts3d,
                                                         optimize_rot=False,
                                                         optimize_transl=False,
                                                         vertices_max_count=mesh_vertices_count,
                                                         optimize_steps=0,
                                                         q=0.95,
                                                         size=OD3D_CATEGORIES_SIZES_IN_M[MAP_CATEGORIES_PASCAL3D_TO_OD3D[category]])

                scale_pascal3d_to_od3d[category] = tform_obj[:3, :3].norm(dim=-1).mean()

                # show:
                #meshes.verts *= scale_pascal3d_to_od3d[category]
                #Meshes.load_from_meshes([meshes.get_mesh_with_id(i) for i in range(meshes.meshes_count)] + [cuboids.get_mesh_with_id(0)]).show(meshes_add_translation=False)

                obj_mesh = cuboids.get_mesh_with_id(0)
                obj_mesh.write_to_file(fpath=fpath_mesh_out)

        log_str = '\n'
        for key, val in scale_pascal3d_to_od3d.items():
            log_str += str(key) + f': {val}, \n'
        logger.info(log_str)

    ##### DATASET PROPERTIES
    def get_frame_by_name_unique(self, name_unique):
        from od3d.datasets.object import OD3D_CAM_TFORM_OBJ_TYPES, OD3D_FRAME_MASK_TYPES, OD3D_MESH_TYPES, \
            OD3D_MESH_FEATS_TYPES, OD3D_MESH_FEATS_DIST_REDUCE_TYPES, \
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
                               depth_type=OD3D_FRAME_DEPTH_TYPES.MESH,
                               kpts2d_annot_type=OD3D_FRAME_KPTS2D_ANNOT_TYPES.META,)


    # def get_item(self, item):
    #     frame_meta = Pascal3DFrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.list_frames_unique[item])
    #     return Pascal3DFrame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
    #                          path_meshes=self.path_meshes, meta=frame_meta, modalities=self.modalities,
    #                          categories=self.categories)

    # @staticmethod
    # def get_rpath_meshes():
    #     return Path("CAD")
    # @staticmethod
    # def get_path_meshes(path_raw: Path):
    #     return path_raw.joinpath(Pascal3D.get_rpath_meshes())
    #
    # @property
    # def path_meshes(self):
    #     return Pascal3D.get_path_meshes(path_raw=self.path_raw)

