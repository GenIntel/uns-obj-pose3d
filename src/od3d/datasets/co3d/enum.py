from od3d.data import ExtEnum
from od3d.datasets.enum import OD3D_CATEGORIES


class CAM_TFORM_OBJ_SOURCES(str, ExtEnum):
    PCL = 'pcl'
    CO3DV1 = 'co3dv1'
    CO3D = 'co3d'
    LABELED = 'labeled'
    LABELED_CUBOID = 'labeled_cuboid'
    ALIGNED = 'aligned'
    ZSP_LABELED = 'zsp_labeled'
    ZSP_LABELED_CUBOID_REF = 'zsp_labeled_cuboid_ref'
    # FIRST_FRAME = 'first_frame'
    # FRONT_FRAME = 'front_frame'
    # FRONT_FRAME_AND_PCL = 'front_frame_and_pcl'
    # KPTS2D_ORIENT_AND_PCL = 'kpts2d_orient_and_pcl'
    # LIMITS3D = 'limits3d'
    # DROID_SLAM = 'droid_slam'
    # DROID_SLAM_ALIGNED = 'droid_slam_aligned'
    # DROID_SLAM_LABELED = 'droid_slam_labeled'
    # DROID_SLAM_ZSP = 'droid_slam_zsp'
    # DROID_SLAM_ZSP_LABELED = 'droid_slam_zsp_labeled'

class CUBOID_SOURCES(str, ExtEnum):
    FRONT_FRAME_AND_PCL = 'front_frame_and_pcl'
    KPTS2D_ORIENT_AND_PCL = 'kpts2d_orient_and_pcl'
    LIMITS3D = 'limits3d'
    DROID_SLAM = 'droid_slam'
    DEFAULT = 'default'
    ALIGNED = 'aligned'
    ZSP_REF_CUBOID = 'zsp_ref_cuboid'
    LABELED = 'labeled'

class CO3D_FRAME_TYPES(str, ExtEnum):
    DEV_KNOWN = 'dev_known'
    DEV_UNSEEN = 'dev_unseen'
    TEST_KNOWN = 'test_known'
    TEST_UNSEEN = 'test_unseen'
    TRAIN_KNOWN = 'train_known'
    TRAIN_UNSEEN = 'train_unseen'
    CO3DV1 = 'co3dv1'

class FEATURE_TYPES(str, ExtEnum):
    DINOV2_AVG = 'dinov2_avg'
    DINOV2_AVG_NORM = 'dinov2_avg_norm'
    DINOV2_ACC = 'dinov2_acc'

class REDUCE_TYPES(str, ExtEnum):
    AVG = 'avg'
    AVG50 = 'avg50'
    MIN = 'min'
    MIN_AVG = 'min_avg'

class PCL_SOURCES(str, ExtEnum):
    CO3D = 'co3d'
    CO3D_CLEAN = 'co3d_clean'
    DROID_SLAM = 'droid_slam'
    DROID_SLAM_CLEAN = 'droid_slam_clean'

class CO3D_FRAME_SPLITS(str, ExtEnum):
    MULTISEQUENCE_CAR_DEV_KNOWN = 'multisequence_car_dev_known'
    MULTISEQUENCE_CAR_DEV_UNSEEN = 'multisequence_car_dev_unseen'
    MULTISEQUENCE_CAR_TEST_KNOWN = 'multisequence_car_test_known'
    MULTISEQUENCE_CAR_TEST_UNSEEN = 'multisequence_car_test_unseen'
    MULTISEQUENCE_CAR_TRAIN_KNOWN = 'multisequence_car_train_known'
    MULTISEQUENCE_CAR_TRAIN_UNSEEN = 'multisequence_car_train_unseen'
    SINGLESEQUENCE_CAR_TEST_0_KNOWN = 'singlesequence_car_test_0_known'
    SINGLESEQUENCE_CAR_TEST_0_UNSEEN = 'singlesequence_car_test_0_unseen'

ALLOW_LIST_FRAME_TYPES = [CO3D_FRAME_TYPES.DEV_KNOWN, CO3D_FRAME_TYPES.DEV_UNSEEN,
                         CO3D_FRAME_TYPES.TRAIN_KNOWN, CO3D_FRAME_TYPES.TRAIN_UNSEEN,
                         CO3D_FRAME_TYPES.TEST_KNOWN, CO3D_FRAME_TYPES.TEST_UNSEEN]

class CO3D_CATEGORIES(str, ExtEnum):
    APPLE = "apple"
    BACKPACK = "backpack"
    BALL = "ball"
    BANANA = "banana"
    BASEBALLBAT = "baseballbat"
    BASEBALLGLOVE = "baseballglove"
    BENCH = "bench"
    BICYCLE = "bicycle"
    BOOK = "book"
    BOTTLE = "bottle"
    BOWL = "bowl"
    BROCCOLI = "broccoli"
    CAKE = "cake"
    CAR = "car"
    CARROT = "carrot"
    CELLPHONE = "cellphone"
    CHAIR = "chair"
    COUCH = "couch"
    CUP = "cup"
    DONUT = "donut"
    FRISBEE = "frisbee"
    HAIRDRYER = "hairdryer"
    HANDBAG = "handbag"
    HOTDOG = "hotdog"
    HYDRANT = "hydrant"
    KEYBOARD = "keyboard"
    KITE = "kite"
    LAPTOP = "laptop"
    MICROWAVE = "microwave"
    MOTORCYCLE = "motorcycle"
    MOUSE = "mouse"
    ORANGE = "orange"
    PARKINGMETER = "parkingmeter"
    PIZZA = "pizza"
    PLANT = "plant"
    REMOTE = "remote"
    SANDWICH = "sandwich"
    SKATEBOARD = "skateboard"
    STOPSIGN = "stopsign"
    SUITCASE = "suitcase"
    TEDDYBEAR = "teddybear"
    TOASTER = "toaster"
    TOILET = "toilet"
    TOYBUS = "toybus"
    TOYPLANE = "toyplane"
    TOYTRAIN = "toytrain"
    TOYTRUCK = "toytruck"
    TV = "tv"
    UMBRELLA = "umbrella"
    VASE = "vase"
    WINEGLASS = "wineglass"

MAP_CO3D_PASCAL3D = {
    "apple" : None,
    "backpack": None,
    "ball": None,
    "banana": None,
    "baseballbat": None,
    "baseballglove": None,
    "bench": None,
    "bicycle": "bicycle",
    "book": None,
    "bottle": "bottle",
    "bowl": None,
    "broccoli": None,
    "cake": None,
    "car": "car",
    "carrot": None,
    "cellphone": None,
    "chair": "chair",
    "couch": "sofa",
    "cup": None,
    "donut": None,
    "frisbee": None,
    "hairdryer": None,
    "handbag": None,
    "hotdog": None,
    "hydrant": None,
    "keyboard": None,
    "kite": None,
    "laptop": None,
    "microwave": None,
    "motorcycle": "motorbike",
    "mouse": None,
    "orange": None,
    "parkingmeter": None,
    "pizza": None,
    "plant": None,
    "remote": None,
    "sandwich": None,
    "skateboard": None,
    "stopsign": None,
    "suitcase": None,
    "teddybear": None,
    "toaster": None,
    "toilet": None,
    "toybus": "bus",
    "toyplane": "aeroplane",
    "toytrain": "train",
    "toytruck": None,
    "tv": "tvmonitor",
    "umbrella": None,
    "vase": None,
    "wineglass": None,
}

MAP_CO3D_PASCAL3D_NOT_NONE = { key: value for key, value in MAP_CO3D_PASCAL3D.items() if value is not None}
# map total: 10
# pascal3d total: 12
# excluded 2: boat, dining table
# questionable 2: toyplane: aeroplane, toytrain: train

MAP_CO3D_OBJECTNET3D = {
    "apple" : None,
    "backpack": "backpack",
    "ball": None,
    "banana": None,
    "baseballbat": None,
    "baseballglove": None,
    "bench": "bench",
    "bicycle": "bicycle",
    "book": None,
    "bottle": "bottle",
    "bowl": None,
    "broccoli": None,
    "cake": None,
    "car": "car",
    "carrot": None,
    "cellphone": "cellphone",
    "chair": "chair",
    "couch": "sofa",
    "cup": "cup",
    "donut": None,
    "frisbee": None,
    "hairdryer": "hair_dryer",
    "handbag": None,
    "hotdog": None,
    "hydrant": None,
    "keyboard": "keyboard",
    "kite": None,
    "laptop": "laptop",
    "microwave": "microwave",
    "motorcycle": "motorbike",
    "mouse": "mouse",
    "orange": None,
    "parkingmeter": None,
    "pizza": None,
    "plant": None,
    "remote": None,
    "sandwich": None,
    "skateboard": None,
    "stopsign": None,
    "suitcase": "suitcase",
    "teddybear": None,
    "toaster": "toaster",
    "toilet": "toilet",
    "toybus": "bus",
    "toyplane": "aeroplane",
    "toytrain": "train",
    "toytruck": None,
    "tv": "tvmonitor",
    "umbrella": None,
    "vase": None,
    "wineglass": None,
}

MAP_CO3D_OBJECTNET3D_NOT_NONE = { key: value for key, value in MAP_CO3D_OBJECTNET3D.items() if value is not None}
# map total: 22
# objectnet3d total: 100
# excluded 78:
# questionable 2: toyplane:aeroplane, toytrain: train


MAP_CO3D_COCO = {
    "apple": "apple",
    "backpack": "backpack",
    "ball": "sports ball",
    "banana": "banana",
    "baseballbat": "baseball bat",
    "baseballglove": "baseball glove",
    "bench": "bench",
    "bicycle": "bicycle",
    "book": "book",
    "bottle": "bottle",
    "bowl": "bowl",
    "broccoli": "broccoli",
    "cake": "cake",
    "car": "car",
    "carrot": "carrot",
    "cellphone": "cell phone",
    "chair": "chair",
    "couch": "couch",
    "cup": "cup",
    "donut": "donut",
    "frisbee": "frisbee",
    "hairdryer": "hair drier",
    "handbag": "handbag",
    "hotdog": "hot dog",
    "hydrant": "fire hydrant",
    "keyboard": "keyboard",
    "kite": "kite",
    "laptop": "laptop",
    "microwave": "microwave",
    "motorcycle": "motorcycle",
    "mouse": "mouse",
    "orange": "orange",
    "parkingmeter": "parking meter",
    "pizza": "pizza",
    "plant": "potted plant",
    "remote": "remote",
    "sandwich": "sandwich",
    "skateboard": "skateboard",
    "stopsign": "stop sign",
    "suitcase": "suitcase",
    "teddybear": "teddy bear",
    "toaster": "toaster",
    "toilet": "toilet",
    "toybus": "bus",
    "toyplane": "airplane",
    "toytrain": "train",
    "toytruck": "truck",
    "tv": "tv",
    "umbrella": "umbrella",
    "vase": "vase",
    "wineglass": "wine glass",
}

MAP_CO3D_COCO_NOT_NONE = { key: value for key, value in MAP_CO3D_COCO.items() if value is not None}
# map total: 51
# coco total: 80
# excluded 29:
# questionable 3: toyplane:aeroplane, toytrain: train, toytruck: truck


MAP_CATEGORIES_OD3D_TO_CO3D = {
    OD3D_CATEGORIES.APPLE: CO3D_CATEGORIES.APPLE,
    OD3D_CATEGORIES.BACKPACK: CO3D_CATEGORIES.BACKPACK,
    OD3D_CATEGORIES.BALL: CO3D_CATEGORIES.BALL,
    OD3D_CATEGORIES.BANANA: CO3D_CATEGORIES.BANANA,
    OD3D_CATEGORIES.BASEBALLBAT: CO3D_CATEGORIES.BASEBALLBAT,
    OD3D_CATEGORIES.BASEBALLGLOVE: CO3D_CATEGORIES.BASEBALLGLOVE,
    OD3D_CATEGORIES.BENCH: CO3D_CATEGORIES.BENCH,
    OD3D_CATEGORIES.BICYCLE: CO3D_CATEGORIES.BICYCLE,
    OD3D_CATEGORIES.BOOK: CO3D_CATEGORIES.BOOK,
    OD3D_CATEGORIES.BOTTLE: CO3D_CATEGORIES.BOTTLE,
    OD3D_CATEGORIES.BOWL: CO3D_CATEGORIES.BOWL,
    OD3D_CATEGORIES.BROCCOLI: CO3D_CATEGORIES.BROCCOLI,
    OD3D_CATEGORIES.CAKE: CO3D_CATEGORIES.CAKE,
    OD3D_CATEGORIES.CAR: CO3D_CATEGORIES.CAR,
    OD3D_CATEGORIES.CARROT: CO3D_CATEGORIES.CARROT,
    OD3D_CATEGORIES.CELLPHONE: CO3D_CATEGORIES.CELLPHONE,
    OD3D_CATEGORIES.CHAIR: CO3D_CATEGORIES.CHAIR,
    OD3D_CATEGORIES.COUCH: CO3D_CATEGORIES.COUCH,
    OD3D_CATEGORIES.CUP: CO3D_CATEGORIES.CUP,
    OD3D_CATEGORIES.DONUT: CO3D_CATEGORIES.DONUT,
    OD3D_CATEGORIES.FRISBEE: CO3D_CATEGORIES.FRISBEE,
    OD3D_CATEGORIES.HAIRDRYER: CO3D_CATEGORIES.HAIRDRYER,
    OD3D_CATEGORIES.HANDBAG: CO3D_CATEGORIES.HANDBAG,
    OD3D_CATEGORIES.HOTDOG: CO3D_CATEGORIES.HOTDOG,
    OD3D_CATEGORIES.HYDRANT: CO3D_CATEGORIES.HYDRANT,
    OD3D_CATEGORIES.KEYBOARD: CO3D_CATEGORIES.KEYBOARD,
    OD3D_CATEGORIES.KITE: CO3D_CATEGORIES.KITE,
    OD3D_CATEGORIES.LAPTOP: CO3D_CATEGORIES.LAPTOP,
    OD3D_CATEGORIES.MICROWAVE: CO3D_CATEGORIES.MICROWAVE,
    OD3D_CATEGORIES.MOTORCYCLE: CO3D_CATEGORIES.MOTORCYCLE,
    OD3D_CATEGORIES.MOUSE: CO3D_CATEGORIES.MOUSE,
    OD3D_CATEGORIES.ORANGE: CO3D_CATEGORIES.ORANGE,
    OD3D_CATEGORIES.PARKINGMETER: CO3D_CATEGORIES.PARKINGMETER,
    OD3D_CATEGORIES.PIZZA: CO3D_CATEGORIES.PIZZA,
    OD3D_CATEGORIES.PLANT: CO3D_CATEGORIES.PLANT,
    OD3D_CATEGORIES.REMOTE: CO3D_CATEGORIES.REMOTE,
    OD3D_CATEGORIES.SANDWICH: CO3D_CATEGORIES.SANDWICH,
    OD3D_CATEGORIES.SKATEBOARD: CO3D_CATEGORIES.SKATEBOARD,
    OD3D_CATEGORIES.STOPSIGN: CO3D_CATEGORIES.STOPSIGN,
    OD3D_CATEGORIES.SUITCASE: CO3D_CATEGORIES.SUITCASE,
    OD3D_CATEGORIES.TEDDYBEAR: CO3D_CATEGORIES.TEDDYBEAR,
    OD3D_CATEGORIES.TOASTER: CO3D_CATEGORIES.TOASTER,
    OD3D_CATEGORIES.TOILET: CO3D_CATEGORIES.TOILET,
    OD3D_CATEGORIES.BUS: CO3D_CATEGORIES.TOYBUS,
    OD3D_CATEGORIES.AIRPLANE: CO3D_CATEGORIES.TOYPLANE,
    OD3D_CATEGORIES.TRAIN: CO3D_CATEGORIES.TOYTRAIN,
    OD3D_CATEGORIES.TRUCK: CO3D_CATEGORIES.TOYTRUCK,
    OD3D_CATEGORIES.TV: CO3D_CATEGORIES.TV,
    OD3D_CATEGORIES.UMBRELLA: CO3D_CATEGORIES.UMBRELLA,
    OD3D_CATEGORIES.VASE: CO3D_CATEGORIES.VASE,
    OD3D_CATEGORIES.WINE_GLASS: CO3D_CATEGORIES.WINEGLASS,
}

# MAP_CATEGORIES_CO3D_TO_OD3D = {v: k for k, v in MAP_CATEGORIES_OD3D_TO_CO3D.items()}
MAP_CATEGORIES_CO3D_TO_OD3D = {
    CO3D_CATEGORIES.APPLE:  OD3D_CATEGORIES.APPLE,
    CO3D_CATEGORIES.BACKPACK:  OD3D_CATEGORIES.BACKPACK,
    CO3D_CATEGORIES.BALL:  OD3D_CATEGORIES.BALL,
    CO3D_CATEGORIES.BANANA:  OD3D_CATEGORIES.BANANA,
    CO3D_CATEGORIES.BASEBALLBAT:  OD3D_CATEGORIES.BASEBALLBAT,
    CO3D_CATEGORIES.BASEBALLGLOVE:  OD3D_CATEGORIES.BASEBALLGLOVE,
    CO3D_CATEGORIES.BENCH:  OD3D_CATEGORIES.BENCH,
    CO3D_CATEGORIES.BICYCLE :  OD3D_CATEGORIES.BICYCLE,
    CO3D_CATEGORIES.BOOK :  OD3D_CATEGORIES.BOOK,
    CO3D_CATEGORIES.BOTTLE :  OD3D_CATEGORIES.BOTTLE,
    CO3D_CATEGORIES.BOWL :  OD3D_CATEGORIES.BOWL,
    CO3D_CATEGORIES.BROCCOLI :  OD3D_CATEGORIES.BROCCOLI,
    CO3D_CATEGORIES.CAKE :  OD3D_CATEGORIES.CAKE,
    CO3D_CATEGORIES.CAR :  OD3D_CATEGORIES.CAR,
    CO3D_CATEGORIES.CARROT :  OD3D_CATEGORIES.CARROT,
    CO3D_CATEGORIES.CELLPHONE :  OD3D_CATEGORIES.CELLPHONE,
    CO3D_CATEGORIES.CHAIR :  OD3D_CATEGORIES.CHAIR,
    CO3D_CATEGORIES.COUCH :  OD3D_CATEGORIES.COUCH,
    CO3D_CATEGORIES.CUP :  OD3D_CATEGORIES.CUP,
    CO3D_CATEGORIES.DONUT :  OD3D_CATEGORIES.DONUT,
    CO3D_CATEGORIES.FRISBEE :  OD3D_CATEGORIES.FRISBEE,
    CO3D_CATEGORIES.HAIRDRYER :  OD3D_CATEGORIES.HAIRDRYER,
    CO3D_CATEGORIES.HANDBAG :  OD3D_CATEGORIES.HANDBAG,
    CO3D_CATEGORIES.HOTDOG :  OD3D_CATEGORIES.HOTDOG,
    CO3D_CATEGORIES.HYDRANT :  OD3D_CATEGORIES.HYDRANT,
    CO3D_CATEGORIES.KEYBOARD :  OD3D_CATEGORIES.KEYBOARD,
    CO3D_CATEGORIES.KITE :  OD3D_CATEGORIES.KITE,
    CO3D_CATEGORIES.LAPTOP :  OD3D_CATEGORIES.LAPTOP,
    CO3D_CATEGORIES.MICROWAVE :  OD3D_CATEGORIES.MICROWAVE,
    CO3D_CATEGORIES.MOTORCYCLE :  OD3D_CATEGORIES.MOTORCYCLE,
    CO3D_CATEGORIES.MOUSE :  OD3D_CATEGORIES.MOUSE,
    CO3D_CATEGORIES.ORANGE :  OD3D_CATEGORIES.ORANGE,
    CO3D_CATEGORIES.PARKINGMETER :  OD3D_CATEGORIES.PARKINGMETER,
    CO3D_CATEGORIES.PIZZA :  OD3D_CATEGORIES.PIZZA,
    CO3D_CATEGORIES.PLANT :  OD3D_CATEGORIES.PLANT,
    CO3D_CATEGORIES.REMOTE :  OD3D_CATEGORIES.REMOTE,
    CO3D_CATEGORIES.SANDWICH :  OD3D_CATEGORIES.SANDWICH,
    CO3D_CATEGORIES.SKATEBOARD :  OD3D_CATEGORIES.SKATEBOARD,
    CO3D_CATEGORIES.STOPSIGN :  OD3D_CATEGORIES.STOPSIGN,
    CO3D_CATEGORIES.SUITCASE :  OD3D_CATEGORIES.SUITCASE,
    CO3D_CATEGORIES.TEDDYBEAR :  OD3D_CATEGORIES.TEDDYBEAR,
    CO3D_CATEGORIES.TOASTER :  OD3D_CATEGORIES.TOASTER,
    CO3D_CATEGORIES.TOILET :  OD3D_CATEGORIES.TOILET,
    CO3D_CATEGORIES.TOYBUS :  OD3D_CATEGORIES.BUS,
    CO3D_CATEGORIES.TOYPLANE :  OD3D_CATEGORIES.AIRPLANE,
    CO3D_CATEGORIES.TOYTRAIN :  OD3D_CATEGORIES.TRAIN,
    CO3D_CATEGORIES.TOYTRUCK :  OD3D_CATEGORIES.TRUCK,
    CO3D_CATEGORIES.TV :  OD3D_CATEGORIES.TV,
    CO3D_CATEGORIES.UMBRELLA :  OD3D_CATEGORIES.UMBRELLA,
    CO3D_CATEGORIES.VASE :  OD3D_CATEGORIES.VASE,
    CO3D_CATEGORIES.WINEGLASS: OD3D_CATEGORIES.WINE_GLASS,
}