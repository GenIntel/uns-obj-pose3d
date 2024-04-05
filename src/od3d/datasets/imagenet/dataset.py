import logging

import od3d.io

logger = logging.getLogger(__name__)
from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.frame_meta import OD3D_FrameMeta, OD3D_FrameMetaRGBMixin, \
    OD3D_FrameMetaBBoxMixin, OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaSubsetMixin, OD3D_FrameMetaSizeMixin
from od3d.datasets.frame import OD3D_FRAME_MODALITIES, OD3D_Frame
#from od3d.datasets.objectnet3d.enum import OBJECTNET3D_CATEOGORIES
from od3d.datasets.imagenet.enum import IMAGENET_CATEOGORIES, IMAGENET_CATEGORIES_CRYPTED
from od3d.datasets.co3d.enum import CO3D_CATEGORIES
from pathlib import Path
from typing import List, Dict
from omegaconf import DictConfig
import shutil
import torch
import od3d.cv.io
from tqdm import tqdm

from dataclasses import dataclass

@dataclass
class ImageNetFrameMeta(OD3D_FrameMetaSubsetMixin, OD3D_FrameMetaRGBMixin, OD3D_FrameMetaSizeMixin,
                        OD3D_FrameMetaCategoryMixin, OD3D_FrameMetaBBoxMixin, OD3D_FrameMeta):

    @property
    def name_unique(self):
        return f'{self.subset}/{self.name}'

    @staticmethod
    def load_from_raw():
        pass

class ImageNet(OD3D_Dataset):
    def __init__(self, name: str, modalities: List[OD3D_FRAME_MODALITIES], path_raw: Path, path_preprocess: Path,
                 categories: List=None,
                 dict_nested_frames: Dict=None, dict_nested_frames_ban: Dict=None,
                 transform=None, index_shift=0, subset_fraction=1.):
        # crane twice:
        #   134 : BIRD
        #   517 : MACHINE
        imagenet_cats = IMAGENET_CATEOGORIES.list()
        # imagenet_cats[134], imagenet_cats[517]
        from od3d.datasets.co3d.enum import CO3D_CATEGORIES
        co3d_cats = CO3D_CATEGORIES.list()
        map_categories_co3d_to_imagenet = {}
        for co3d_cat in co3d_cats:
            map_categories_co3d_to_imagenet[co3d_cat] = list(filter(lambda cat: co3d_cat in cat, imagenet_cats))
        from od3d.datasets.co3d.enum import CO3D_CATEGORIES
        categories = categories if categories is not None else  [] # TODO: OBJECTNET3D_CATEOGORIES.list()

        super().__init__(categories=categories, dict_nested_frames=dict_nested_frames, dict_nested_frames_ban=dict_nested_frames_ban, name=name, modalities=modalities, path_raw=path_raw,
                         path_preprocess=path_preprocess, transform=transform, index_shift=index_shift,
                         subset_fraction=subset_fraction)

    def get_item(self, item):
        frame_meta = ImageNetFrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, name_unique=self.list_frames_unique[item])
        return OD3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                          meta=frame_meta, modalities=self.modalities, categories=self.categories)

    @staticmethod
    def setup(config: DictConfig):
        # logger.info(OmegaConf.to_yaml(config))
        path_raw = Path(config.path_raw)
        if path_raw.exists() and config.setup.remove_previous:
            logger.info(f"Removing previous ImageNet")
            shutil.rmtree(path_raw)

        path_raw.mkdir(parents=True, exist_ok=True)
        od3d.io.run_cmd('pip install kaggle', logger=logger, live=True)
        od3d.io.write_config_to_json_file(config=config.credentials.kaggle, fpath=Path('~/.kaggle/kaggle.json'))
        od3d.io.run_cmd(f"cd {path_raw} && kaggle competitions download -c imagenet-object-localization-challenge",
                        logger=logger, live=True)
        logger.info('extracting files...')
        od3d.io.unzip(path_raw.joinpath("imagenet-object-localization-challenge.zip"), dst=path_raw)

    @staticmethod
    def extract_meta(config: DictConfig):
        path_meta = OD3D_Dataset.get_path_meta(config=config)
        path_raw = OD3D_Dataset.get_path_raw(config=config)

        import od3d.io
        categories = od3d.io.read_str_from_file(path_raw.joinpath('LOC_synset_mapping.txt')).split('\n')
        for category_line in categories:
            #category = '_'.join(category_line.split(' ')[1:]).replace(',','').replace('-', '_').replace("'", '').replace(".", '_')
            category = category_line.split(' ')[0]
            #print(f'{category.upper()} = "{category.lower()}"')
            # logger.info(category)


        for subset in ['train_cls', 'train_loc', 'val', 'test']: # # 'test'
            logger.info(f'preprocess subset {subset}')
            if 'test' in subset:
                annots = None
            elif 'train' in subset:
                annots = sorted(od3d.io.read_str_from_file(path_raw.joinpath('LOC_train_solution.csv')).split('\n')[1:])
            else:
                annots = sorted(od3d.io.read_str_from_file(path_raw.joinpath('LOC_val_solution.csv')).split('\n')[1:])
            path_meta_subset = OD3D_FrameMeta.get_path_metas(path_meta=path_meta).joinpath(subset)
            if not path_meta_subset.exists() or config.get("extract_meta", False).get("override", False):
                fnames_lines = od3d.io.read_str_from_file(path_raw.joinpath('ILSVRC', 'ImageSets', 'CLS-LOC', f'{subset}.txt')).split('\n')
                fnames_lines = sorted(fnames_lines, key=lambda fname_line: fname_line.split(',')[0]) # .split('_')[1]
                for i in tqdm(range(len(fnames_lines))):
                    fname_line = fnames_lines[i]
                    rfpath = fname_line.split(' ')[0]
                    if 'train' in subset:
                        fname = rfpath.split('/')[1].split('.')[0]
                    else:
                        fname = rfpath.split('.')[0]

                    if 'test' in subset:
                        category = 'None'
                        l_bbox = torch.Tensor([0., 0., 0., 0.]).tolist()
                        rfpath_rgb = Path('ILSVRC').joinpath('Data', 'CLS-LOC', 'test', f'{rfpath}.JPEG')
                    elif 'train' in subset:
                        category = rfpath.split('/')[0]
                        if subset == 'train_cls':
                            l_bbox = torch.Tensor([0., 0., 0., 0.]).tolist()
                        else:
                            if not annots[i].startswith(fname):
                                logger.warning(f'skip frame {fname} as annotation and fname dont match: {annots[i]} {fname}')
                                continue
                            l_bbox = torch.LongTensor([int(a) for a in annots[i].split(' ')[1:5]]).tolist()
                        fpath_annot = path_raw.joinpath('ILSVRC', 'Annotations', 'CLS-LOC', 'train', f'{rfpath}.xml')
                        rfpath_rgb = Path('ILSVRC').joinpath('Data', 'CLS-LOC', 'train', f'{rfpath}.JPEG')
                    else:
                        if not annots[i].startswith(fname):
                            logger.warning(f'skip frame {fname} as annotation and fname dont match: {annots[i]} {fname}')
                            continue
                        annot_split = annots[i].split(' ')
                        category = annot_split[0].split(',')[1]
                        l_bbox = torch.LongTensor([int(a) for a in annot_split[1:5]]).tolist()
                        fpath_annot = path_raw.joinpath('ILSVRC', 'Annotations', 'CLS-LOC', 'val', f'{rfpath}.xml', )
                        rfpath_rgb = Path('ILSVRC').joinpath('Data', 'CLS-LOC', 'val', f'{rfpath}.JPEG')

                    img = od3d.cv.io.read_image(path=path_raw.joinpath(rfpath_rgb))
                    l_size = torch.LongTensor([img.shape[1], img.shape[2]]).tolist()

                    frame_meta = ImageNetFrameMeta(name=fname, l_bbox=l_bbox, category=category, l_size=l_size, rfpath_rgb=rfpath_rgb, subset=subset)
                    if frame_meta is not None:
                        frame_meta.save(path_meta=path_meta)
            else:
                logger.info(f'found meta subset at {path_meta_subset}')

    def preprocess(self, override: False):
        pass