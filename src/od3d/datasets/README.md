

# To add a new dataset

## 1. add a new dataset class


```
from od3d.data import ExtEnum
from od3d.datasets.dataset import OD3D_Dataset
from od3d.datasets.frame import OD3D_Frame


class EXAMPLE_CATEGORIES(str, ExtEnum):
    AEROPLANE = "aeroplane"
    BICYCLE = "bicycle"
    BOAT = "boat"
    BOTTLE = "bottle"
    ...

class EXAMPLE(OD3D_Dataset):

    def __init__(
        self,
        name: str,
        modalities: List[OD3D_FRAME_MODALITIES],
        path_raw: Path,
        path_preprocess: Path,
        categories: List[EXAMPLE_CATEGORIES] = None,
        dict_nested_frames: Dict[...] = None,
        transform=None,
        subset_fraction=1.,
        index_shift=0,
    ):
        categories = categories if categories is not None else EXAMPLE_CATEGORIES.list()
        super().__init__(categories=categories, name=name, 
                         modalities=modalities, path_raw=path_raw, 
                         path_preprocess=path_preprocess, transform=transform, 
                         subset_fraction=subset_fraction, index_shift=index_shift, 
                         dict_nested_frames=dict_nested_frames)


    def get_subset_with_dict_nested_frames(self, dict_nested_frames):
        return EXAMPLE(name=self.name, modalities=self.modalities, path_raw=self.path_raw,
                       path_preprocess=self.path_preprocess, categories=self.categories,
                       dict_nested_frames=dict_nested_frames, transform=self.transform, index_shift=self.index_shift)

    
    def get_item(self, item):
        frame_meta = EXAMPLEFrameMeta.load_from_meta_with_name_unique(path_meta=self.path_meta, 
                                                                      name_unique=self.list_frames_unique[item])
        return OD3D_Frame(path_raw=self.path_raw, path_preprocess=self.path_preprocess, path_meta=self.path_meta,
                          meta=frame_meta, modalities=self.modalities,
                          categories=self.categories)
    
    @staticmethod
    def setup(config):
        path_raw = Path(config.path_raw)

        if path_raw.exists() and config.setup_remove_previous:
            logger.info(f"Removing previous EXAMPLE")
            shutil.rmtree(path_raw)

        if path_raw.exists():
            logger.info(f"Found EXAMPLE dataset at {path_raw}")
        else:
            ...
            
    
    @staticmethod
    def preprocess_meta(config: DictConfig):
        path_raw = OD3D_Dataset.get_path_raw(config=config)
        path_meta = OD3D_Dataset.get_path_meta(config=config)
        
        if config.preprocess_meta_remove_previous:
            if path_meta.exists():
                shutil.rmtree(path_meta)
        
    
    def preprocess(override: bool = False):
        ...
```

## 2. add a new frame meta class

```
from od3d.datasets.frame import OD3D_FrameMeta, OD3D_FrameMetaXXXMixin

@dataclass
class EXAMPLEFrameMeta(OD3D_FrameMetaXXXMixin, OD3D_FrameMeta):

    @property
    def name_unique(self):
        return f'{self.subset}/{self.category}/{self.name}'

    @staticmethod
    def get_name_unique_from_category_subset_name(category, subset, name):
        return f'{subset}/{category}/{name}'

    @staticmethod
    def load_from_raw(...):
        ...



```




## 3. add a new config 

```
name: example
class_name: EXAMPLE

modalities:
    - 'rgb'
    - 'categories'
    ...
    
    
path_raw: ${platform.path_datasets}/EXAMPLE
path_preprocess: ${platform.path_datasets}/EXAMPLE_Preprocess

setup_remove_previous: False
setup_override: False

preprocess_meta_remove_previous: False
preprocess_meta_override: False
```