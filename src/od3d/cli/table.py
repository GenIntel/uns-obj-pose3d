import logging
logger = logging.getLogger(__name__)
import typer
import od3d.io
from pathlib import Path
import pandas as pd

app = typer.Typer()
from od3d.cli.benchmark import get_dataframe
from tabulate import tabulate
import re
from od3d.datasets.co3d.enum import MAP_CATEGORIES_OD3D_TO_CO3D
from od3d.datasets.objectnet3d.enum import MAP_CATEGORIES_OD3D_TO_OBJECTNET3D, MAP_CATEGORIES_OBJECTNET3D_TO_OD3D
# def get_categorical_and

TABLE_CATEGORIES_OBJECTNET3D_3 = ['cellphone', 'toilet', 'microwave', 'mean (3)']
# TABLE_CATEGORIES_OBJECTNET3D_23 = [
#     'cellphone', 'toilet', 'microwave', 'airplane', 'backpack', 'bench', 'bicycle', 'bottle', 'bus', 'car',
#     'chair', 'couch', 'cup', 'hairdryer', 'keyboard', 'laptop', 'motorcycle',
#     'mouse', 'remote', 'suitcase', 'toaster', 'train', 'tv', 'mean (23)'
# ]
TABLE_CATEGORIES_OBJECTNET3D_20 = [ # 'bottle', 'train',  'airplane',
    'cellphone', 'toilet', 'microwave', 'backpack', 'bench', 'bicycle', 'bus', 'car',
    'chair', 'couch', 'cup', 'hairdryer', 'keyboard', 'laptop', 'motorcycle',
    'mouse', 'remote', 'suitcase', 'toaster', 'tv', 'mean (20)'
]

TABLE_CATEGORIES_OBJECTNET3D_11 = [ # 'bottle', 'train',  'airplane',
    'cellphone', 'microwave', 'backpack', 'bench', 'cup', 'hairdryer', 'laptop', 'mouse', 'remote', 'toaster', 'mean (10)'
]

TABLE_CATEGORIES_OBJECTNET3D_9 = [ # 'bottle', 'train',  'airplane',
    'toilet', 'bicycle',  'bus', 'car', 'chair', 'couch',  'keyboard', 'motorcycle',  'suitcase', 'tv', 'mean (10)'
]

TABLE_CATEGORIES_CO3D_20 = ['bicycle', 'truck', 'train', 'teddybear', 'car', 'bus', 'motorcycle', 'keyboard', 'handbag', 'remote', 'airplane', 'toilet', 'hairdryer', 'mouse', 'toaster', 'hydrant', 'chair', 'laptop', 'book', 'backpack', 'mean (20)']
TABLE_CATEGORIES_CO3D_28 = ['bicycle', 'truck', 'train', 'teddybear', 'car', 'bus', 'motorcycle', 'keyboard', 'handbag', 'remote', 'airplane', 'toilet', 'hairdryer', 'mouse', 'toaster', 'hydrant', 'chair', 'laptop', 'book', 'backpack', 'cellphone', 'microwave', 'bench', 'bottle', 'couch', 'cup', 'suitcase', 'tv', 'mean (28)']

TABLE_CATEGORIES_ZSP = ['bicycle', 'hydrant', 'motorcycle', 'teddybear', 'toaster', 'mean (20)']
TABLE_CATEGORIES_YOLO = ['backpack', 'car', 'chair', 'keyboard', 'laptop', 'motorcycle', 'mean (20)'] # # B’pack Car Chair Keyboard Laptop M’cycle
TABLE_CATEGORIES_5S = ['mean (20)', 'mean (28)']
# TABLE_CATEGORIES_PASCAL3D = ['airplane', 'bicycle', 'bottle', 'bus', 'car', 'chair', 'motorcycle', 'couch', 'train', 'tv', 'mean (10)']
TABLE_CATEGORIES_PASCAL3D = ['bicycle', 'bus', 'car', 'chair', 'motorcycle', 'couch', 'tv', 'mean (7)'] # # 'bottle', 'train',  'airplane',

DATASET_PASCAL3D = 'pascal3d'
DATASET_CO3D_20 = 'co3d_20'
DATASET_CO3D_28 = 'co3d_28'
DATASET_OBJECTNET3D = 'objectnet3d'

@app.command()
def runs(runs_names_regex: str = typer.Option('.*', '-r', '--runs'),
         runs_names_regex_groups: str = typer.Option('', '-n', '--runs-names-regex-groups'),
         metrics: str = typer.Option(None, '-m', '--metrics'),
         configs: str = typer.Option(None, '-c', '--configs'),
         summary_cols: str = typer.Option(None, '-s', '--summary-cols'),
         age_in_hours_gt: int = typer.Option(0, '-g', '--age-in-hours-gt'),
         age_in_hours_lt: int = typer.Option(1000, '-l', '--age-in-hours-lt'),
         duplicates_keep: str = typer.Option('last', '-d', '--duplicates_keep'),
         show_index: bool = typer.Option(False, '-i', '--show_index'),):

    digits = 3
    logging.basicConfig(level=logging.INFO)
    if configs is not None:
        configs = configs.split(',')
    else:
        configs = []
    if metrics is not None:
        metrics = metrics.split(',')
    else:
        metrics = []
    runs_names_regex_groups = runs_names_regex_groups.split(',')
    df = get_dataframe(configs=configs, metrics=metrics, name_regex=runs_names_regex,
                       name_regex_groups = runs_names_regex_groups,
                       age_in_hours_gt = age_in_hours_gt, age_in_hours_lt = age_in_hours_lt,
                       duplicates_keep = duplicates_keep)

    if summary_cols is not None:
        for metric in metrics:
            df_metric = df.groupby(summary_cols.split(','))[metric].agg(['mean', 'std']).reset_index()
            # logger.info(tabulate(df_metric, headers='keys', tablefmt='tsv', floatfmt=f".{digits}f", showindex=False)) # latex
            logger.info(re.sub(r'[^\S\r\n]+', ', ', tabulate(df_metric, headers='keys', stralign="left", tablefmt="plain", floatfmt=f".{digits}f", showindex=show_index))) # latex

            #logger.info(df_metric)

    else:
        # logger.info(tabulate(df, headers='keys', tablefmt='csv', floatfmt=f".{digits}f")) # latex csv tsv
        logger.info(re.sub(r'[^\S\r\n]+', ', ', tabulate(df, headers='keys', stralign="left", tablefmt="plain", floatfmt=f".{digits}f", showindex=show_index))) # latex csv tsv
        # , stralign="right", numalign="right"
        # logger.info(df)
        # , stralign="left", tablefmt="plain").replace('  ', ', '))

@app.command()
def multiple(benchmark: str = typer.Option('co3d_nemo_align3d', '-b', '--benchmark'),
             ablation: str = typer.Option(None, '-a', '--ablation'),
             platform: str = typer.Option(None, '-p', '--platform'),
             age_in_hours_gt: int = typer.Option(0, '-g', '--age-in-hours-gt'),
             age_in_hours_lt: int = typer.Option(1000, '-l', '--age-in-hours-lt'),
             metrics: str = typer.Option(None, '-m', '--metrics'),
             configs: str = typer.Option(None, '-c', '--configs'),
             summary_cols: str = typer.Option(None, '-s', '--summary-cols'),
             duplicates_keep: str = typer.Option('last', '-d', '--duplicates_keep'),
             show_index: bool = typer.Option(False, '-i', '--show_index'), ):
    digits = 3
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

    df = get_dataframe_multiple(benchmark=benchmark, ablation=ablation, platform=platform,
                                age_in_hours_gt=age_in_hours_gt, age_in_hours_lt=age_in_hours_lt,
                                metrics=metrics, configs=configs, duplicates_keep=duplicates_keep,
                                add_configs_ablation=False)


    if summary_cols is not None:
        for metric in metrics:
            df_metric = df.groupby(summary_cols.split(','))[metric].agg(['mean', 'std']).reset_index()
            # logger.info(tabulate(df_metric, headers='keys', tablefmt='tsv', floatfmt=f".{digits}f", showindex=False)) # latex
            logger.info(re.sub(r'[^\S\r\n]+', ', ', tabulate(df_metric, headers='keys', stralign="left", tablefmt="plain", floatfmt=f".{digits}f", showindex=show_index))) # latex

            #logger.info(df_metric)

    else:
        # logger.info(tabulate(df, headers='keys', tablefmt='csv', floatfmt=f".{digits}f")) # latex csv tsv
        logger.info(re.sub(r'[^\S\r\n]+', ', ', tabulate(df, headers='keys', stralign="left", tablefmt="plain", floatfmt=f".{digits}f", showindex=show_index))) # latex csv tsv
        # , stralign="right", numalign="right"
        # logger.info(df)
        # , stralign="left", tablefmt="plain").replace('  ', ', '))

@app.command()
def pascal3d(
        name_regex: str = typer.Option('Pascal3D_NeMo', '-n', '--name'),
        age_in_hours: int = typer.Option(182, '-h', '--hours'),
        digits: int = typer.Option(1, '-d', '--digits')
    ):
    map_runs_names = None

    # regex either head or backbone
    #name_regex = '.*Pascal3D_NeMo_(bs.*)_slurm'
    # map_runs_names = {
    #     '11-28_14-50-50_Pascal3D_NeMo_bs1x6_slurm': '1x6',
    #     '11-28_15-53-00_Pascal3D_NeMo_bs1x12_slurm': '1x12',
    #     '11-28_14-49-42_Pascal3D_NeMo_bs1x24_slurm': '1x24',
    #     '11-28_14-50-05_Pascal3D_NeMo_bs2x6_slurm': '2x6',
    #     '11-28_14-50-16_Pascal3D_NeMo_bs2x12_slurm': '2x12',
    #     '11-28_14-51-01_Pascal3D_NeMo_bs2x24_slurm': '2x24',
    #     '11-28_14-49-53_Pascal3D_NeMo_bs4x12_slurm': '4x12',
    #     '11-28_14-50-38_Pascal3D_NeMo_bs16x12_slurm': '16x12',
    # }
    #
    # map_runs_names = {
    #     '11-29_11-59-52_Pascal3D_NeMo_bs1x6_slurm': '1x6',
    #     '11-29_11-59-29_Pascal3D_NeMo_bs1x12_slurm': '1x12',
    #     '11-29_11-58-44_Pascal3D_NeMo_bs1x24_slurm': '1x24',
    #     '11-29_11-59-06_Pascal3D_NeMo_bs2x6_slurm': '2x6',
    #     '11-29_11-59-18_Pascal3D_NeMo_bs2x12_slurm': '2x12',
    #     '11-29_12-00-03_Pascal3D_NeMo_bs2x24_slurm': '2x24',
    #     '11-29_11-58-55_Pascal3D_NeMo_bs4x12_slurm': '4x12',
    #     '11-29_11-58-32_Pascal3D_NeMo_bs8x12_slurm': '8x12',
    #     '11-29_11-59-40_Pascal3D_NeMo_bs16x12_slurm': '16x12',
    # }

    # map_runs_names = {
    #     '11-30_14-11-28_Pascal3D_NeMo_bs1x12_slurm': '1x12',
    #     '11-30_14-10-43_Pascal3D_NeMo_bs1x24_slurm': '1x24',
    #     '11-30_14-12-02_Pascal3D_NeMo_bs1x48_slurm': '1x48',
    #     '11-30_14-11-05_Pascal3D_NeMo_bs2x6_slurm': '2x6',
    #     '11-30_14-11-17_Pascal3D_NeMo_bs2x12_slurm': '2x12',
    #     '11-30_14-12-25_Pascal3D_NeMo_bs2x24_slurm': '2x24',
    #     '11-30_14-11-39_Pascal3D_NeMo_bs2x48_slurm': '2x48',
    #     '11-30_14-10-54_Pascal3D_NeMo_bs4x12_slurm': '4x12',
    #     '11-30_14-10-31_Pascal3D_NeMo_bs8x12_slurm': '8x12',
    #     '11-30_14-11-51_Pascal3D_NeMo_bs16x12_slurm': '16x12',
    # }

    # name_regex = '.*Pascal3D_NeMo_(.*epochs.*)_slurm'
    # map_runs_names = {
    #     '11-30_14-02-18_Pascal3D_NeMo_backbone_resnet_unfrozen_moving_avg_epochs10_slurm': 'ResNet50 + ResNet Head + Mov. Avg. + Epochs 10',
    #     #'11-30_11-08-22_Pascal3D_NeMo_backbone_resnet_unfrozen_moving_avg_epochs10_slurm': 'ResNet50 + ResNet Head + Mov. Avg. + Epochs 10 (2)',
    #     '11-30_14-03-49_Pascal3D_NeMo_backbone_resnet_unfrozen_epochs10_slurm': 'ResNet50 + ResNet Head + Epochs 10',
    #     #'11-30_11-08-13_Pascal3D_NeMo_backbone_resnet_unfrozen_epochs10_slurm': 'ResNet50 + ResNet Head + Epochs 10 (2)',
    #     '11-30_14-03-04_Pascal3D_NeMo_head_resnet_epochs10_slurm': 'DinoV2 + ResNet Head + epochs 10',
    #     '11-30_14-01-33_Pascal3D_NeMo_head_vit_depth_1_epochs10_slurm': 'DinoV2 + ViT Head + epochs 10',
    #     '11-30_14-02-30_Pascal3D_NeMo_backbone_resnet_unfrozen_moving_avg_epochs50_slurm': 'ResNet50 + ResNet Head + Mov. Avg. + Epochs 50',
    #     '11-30_14-04-00_Pascal3D_NeMo_backbone_resnet_unfrozen_epochs50_slurm': 'ResNet50 + ResNet Head + Epochs 50',
    #     '11-30_14-03-15_Pascal3D_NeMo_head_resnet_epochs50_slurm': 'DinoV2 + ResNet Head + Epochs 50',
    #     '11-30_14-01-45_Pascal3D_NeMo_head_vit_depth_1_epochs50_slurm': 'DinoV2 + ViT Head + Epochs 50',
    #     '11-30_14-02-41_Pascal3D_NeMo_backbone_resnet_unfrozen_moving_avg_epochs100_slurm': 'ResNet50 + ResNet Head + Mov. Avg. + Epochs 100',
    #     '11-30_14-04-12_Pascal3D_NeMo_backbone_resnet_unfrozen_epochs100_slurm': 'ResNet50 + ResNet Head + Epochs 100',
    #     '11-30_14-03-26_Pascal3D_NeMo_head_resnet_epochs100_slurm': 'DinoV2 + ResNet Head + Epochs 100',
    #     '11-30_14-01-56_Pascal3D_NeMo_head_vit_depth_1_epochs100_slurm': 'DinoV2 + ViT Head + Epochs 100',
    # }

    # name_regex = '.*Pascal3D_NeMo_(.*mask.*)_slurm'
    # map_runs_names = {
    #     '11-29_17-00-52_Pascal3D_NeMo_no_mask_slurm': 'No Mask',
    #     '11-29_17-00-41_Pascal3D_NeMo_mask_test_slurm': 'Mask Test',
    #     '11-29_17-00-30_Pascal3D_NeMo_mask_train_slurm': 'Mask Train',
    #     '11-29_17-00-18_Pascal3D_NeMo_mask_train_test_slurm': 'Mask Train Test',
    # }

    #name_regex = '.*Pascal3D_NeMo_(temp_[^b]*)_slurm'
    name_regex = '.*Pascal3D_NeMo_(head_.*|backbone_.*)_slurm'
    name_regex = '.*Pascal3D_NeMo_(epochs.*|)(head_.*|backbone_.*)_slurm'
    # name_regex = '.*Pascal3D_NeMo_(inf.*|)(head_.*|backbone_.*)_slurm'
    # moving average does not work as good for me as no moving average
    # map_runs_names = {
    #     '12-01_15-48-28_Pascal3D_NeMo_backbone_resnet_unfrozen_moving_avg_slurm': 'ResNet50 + ResNet Head + Moving Avg. + Epochs 50',
    #     '12-01_15-48-52_Pascal3D_NeMo_head_resnet_moving_avg_slurm': 'DinoV2 + ResNet Head + Moving Avg. + Epochs 50',
    #     '12-01_15-49-15_Pascal3D_NeMo_backbone_resnet_unfrozen_slurm': 'ResNet50 + ResNet Head + Epochs 50',
    #     '12-03_16-44-52_Pascal3D_NeMo_epochs100_backbone_resnet_unfrozen_moving_avg_slurm': 'ResNet50 + ResNet Head + Moving Avg. + Epochs 100',
    #     '12-03_17-10-06_Pascal3D_NeMo_epochs100_head_resnet_moving_avg_slurm': 'DinoV2 + ResNet Head + Moving Avg. + Epochs 100',
    #     '12-03_16-45-37_Pascal3D_NeMo_epochs100_backbone_resnet_unfrozen_slurm': 'ResNet50 + ResNet Head + Epochs 100',
    #     '12-03_16-45-26_Pascal3D_NeMo_epochs100_head_resnet_slurm': 'DinoV2 + ResNet Head + Epochs 100',
    # }

    # age_in_hours = 200
    # name_regex = '.*Pascal3D_NeMo_.*_slurm'
    # map_runs_names = {
    #     '12-04_11-30-32_Pascal3D_NeMo_dinov2_moving_avg_epochs50_slurm': 'DinoV2 + ResNet Head + Moving Avg. + Epochs 50',
    #     '12-04_10-40-20_Pascal3D_NeMo_head_resnet_slurm': 'DinoV2 + ResNet Head + Epochs 50',
    #     '12-04_11-30-54_Pascal3D_NeMo_dinov2_moving_avg_epochs100_slurm': 'DinoV2 + ResNet Head + Moving Avg. + Epochs 100',
    #     '12-03_16-45-26_Pascal3D_NeMo_epochs100_head_resnet_slurm': 'DinoV2 + ResNet Head + Epochs 100',
    #     '12-04_11-30-43_Pascal3D_NeMo_dinov2_moving_avg_epochs150_slurm': 'DinoV2 + ResNet Head + Moving Avg. + Epochs 150',
    # } # 12-04_11-30-43_Pascal3D_NeMo_dinov2_moving_avg_epochs150_slurm

    # age_in_hours = 24
    # name_regex = '12-1[12]_.*_Pascal3D_NeMo_epochs50'
    # map_runs_names = {
    #     '12-12_09-09-11_Pascal3D_NeMo_epochs50_uniform_refine3d_slurm': 'Uniform + Refine 3D',
    #     '12-11_19-09-51_Pascal3D_NeMo_epochs50_uniform_refine6d_slurm': 'Uniform + Refine 6D',
    #     '12-11_19-10-02_Pascal3D_NeMo_epochs50_uniform_refine3d_only_rendered_slurm': 'Uniform + Refine 3D (Only Rendered)',
    #     '12-11_19-08-54_Pascal3D_NeMo_epochs50_uniform_refine3d_no_clutter_slurm': 'Uniform + Refine 3D (No Clutter)',
    #     '12-11_19-09-28_Pascal3D_NeMo_epochs50_uniform_refine3d_only_rendered_no_clutter_slurm': 'Uniform + Refine 3D (Only Rendered + No Clutter)',
    #     '12-12_09-09-10_Pascal3D_NeMo_epochs50_uniform_depth_from_box_refine3d_slurm': 'Uniform + Depth from BBox + Refine 3D',
    #     '12-11_19-08-43_Pascal3D_NeMo_epochs50_uniform_depth_from_box_refine6d_slurm': 'Uniform + Depth from BBox + Refine 6D',
    #     '12-11_19-09-05_Pascal3D_NeMo_epochs50_epnp_refine6d_slurm': 'EPnP + Refine 6D',
    #     '12-11_19-08-32_Pascal3D_NeMo_epochs50_epnp_refine6d_epochs60_slurm': 'EPnP + 2x Refine 6D',
    #     '12-11_19-08-20_Pascal3D_NeMo_epochs50_epnp_refine3d_slurm': 'EPnP + Refine 3D',
    # }

    # age_in_hours = 24
    # name_regex = '12-1[12]_.*_Pascal3D_NeMo_epochs100'
    # map_runs_names = {
    #     '12-11_19-13-25_Pascal3D_NeMo_epochs100_uniform_refine3d_slurm': 'Uniform + Refine 3D',
    #     '12-11_19-13-36_Pascal3D_NeMo_epochs100_uniform_refine6d_slurm': 'Uniform + Refine 6D',
    #     '12-11_19-13-48_Pascal3D_NeMo_epochs100_uniform_refine3d_only_rendered_slurm': 'Uniform + Refine 3D (Only Rendered)',
    #     '12-11_19-12-40_Pascal3D_NeMo_epochs100_uniform_refine3d_no_clutter_slurm': 'Uniform + Refine 3D (No Clutter)',
    #     '12-11_19-13-14_Pascal3D_NeMo_epochs100_uniform_refine3d_only_rendered_no_clutter_slurm': 'Uniform + Refine 3D (Only Rendered + No Clutter)',
    #     '12-11_19-13-03_Pascal3D_NeMo_epochs100_uniform_depth_from_box_refine3d_slurm': 'Uniform + Depth from BBox + Refine 3D',
    #     '12-11_19-12-29_Pascal3D_NeMo_epochs100_uniform_depth_from_box_refine6d_slurm': 'Uniform + Depth from BBox + Refine 6D',
    #     '12-11_19-12-51_Pascal3D_NeMo_epochs100_epnp_refine6d_slurm': 'EPnP + Refine 6D',
    #     '12-11_19-12-17_Pascal3D_NeMo_epochs100_epnp_refine6d_epochs60_slurm': 'EPnP + 2x Refine 6D',
    #     '12-11_19-12-06_Pascal3D_NeMo_epochs100_epnp_refine3d_slurm': 'EPnP + Refine 3D',
    # }

    age_in_hours = 24
    name_regex = '12-1[12]_.*_Pascal3D_NeMo_epochs150'

    map_runs_names = {
        '12-11_19-11-32_Pascal3D_NeMo_epochs150_uniform_refine3d_slurm': 'Uniform + Refine 3D',
        '12-11_19-11-44_Pascal3D_NeMo_epochs150_uniform_refine6d_slurm': 'Uniform + Refine 6D',
        '12-11_19-11-55_Pascal3D_NeMo_epochs150_uniform_refine3d_only_rendered_slurm': 'Uniform + Refine 3D (Only Rendered)',
        '12-11_19-10-47_Pascal3D_NeMo_epochs150_uniform_refine3d_no_clutter_slurm': 'Uniform + Refine 3D (No Clutter)',
        '12-11_19-11-21_Pascal3D_NeMo_epochs150_uniform_refine3d_only_rendered_no_clutter_slurm': 'Uniform + Refine 3D (Only Rendered + No Clutter)',
        '12-11_19-11-10_Pascal3D_NeMo_epochs150_uniform_depth_from_box_refine3d_slurm': 'Uniform + Depth from BBox + Refine 3D',
        '12-11_19-10-36_Pascal3D_NeMo_epochs150_uniform_depth_from_box_refine6d_slurm': 'Uniform + Depth from BBox + Refine 6D',
        '12-11_19-10-58_Pascal3D_NeMo_epochs150_epnp_refine6d_slurm': 'EPnP + Refine 6D',
        '12-11_19-10-24_Pascal3D_NeMo_epochs150_epnp_refine6d_epochs60_slurm': 'EPnP + 2x Refine 6D',
        '12-11_19-10-13_Pascal3D_NeMo_epochs150_epnp_refine3d_slurm': 'EPnP + Refine 3D',
    }
    # vit does not head helps classfication
    # map_runs_names = {
    #     '12-04_10-40-20_Pascal3D_NeMo_head_resnet_slurm': 'DinoV2 + ResNet Head + Epochs 50',
    #     '12-01_15-48-17_Pascal3D_NeMo_head_vit_depth_1_slurm': 'DinoV2 + ViT Head + Epochs 50',
    #     '12-03_16-45-26_Pascal3D_NeMo_epochs100_head_resnet_slurm': 'DinoV2 + ResNet Head + Epochs 100',
    #     '12-03_16-44-41_Pascal3D_NeMo_epochs100_head_vit_depth_1_slurm': 'DinoV2 + ViT Head + Epochs 100',
    # }

    # name_regex = '.*Pascal3D_NeMo_(epochs.*|)(head_.*|backbone_.*)(bs.*)_slurm'
    # # batch size important matters?
    # map_runs_names = {
    #     '12-03_16-49-26_Pascal3D_NeMo_epochs100_backbone_resnet_unfrozen_bs1x6_slurm': 'ResNet50 + ResNet Head + Epochs 100 + BS 1x6',
    #     '12-03_16-49-03_Pascal3D_NeMo_epochs100_backbone_resnet_unfrozen_bs1x12_slurm': 'ResNet50 + ResNet Head + Epochs 100 + BS 1x12',
    #     '12-03_16-48-52_Pascal3D_NeMo_epochs100_backbone_resnet_unfrozen_bs1x24_slurm': 'ResNet50 + ResNet Head + Epochs 100 + BS 1x24',
    #     '12-03_16-48-41_Pascal3D_NeMo_epochs100_head_resnet_bs1x6_slurm': 'DinoV2 + ResNet Head + Epochs 100 + BS 1x6',
    #     '12-03_16-48-18_Pascal3D_NeMo_epochs100_head_resnet_bs1x12_slurm': 'DinoV2 + ResNet Head + Epochs 100 + BS 1x12',
    #     '12-03_16-48-07_Pascal3D_NeMo_epochs100_head_resnet_bs1x24_slurm': 'DinoV2 + ResNet Head + Epochs 100 + BS 1x24',
    # }

    # name_regex = '.*Pascal3D_NeMo_(epochs.*|)(head_.*|backbone_.*)_slurm'
    # map_runs_names = {
    #   '12-04_10-40-20_Pascal3D_NeMo_head_resnet_slurm': 'DinoV2 + ResNet Head + Epochs 50',
    #   '12-04_10-40-31_Pascal3D_NeMo_head_resnet_no_temp_slurm': 'DinoV2 + ResNet Head + No Temp. + Epochs 50',
    #   '12-01_15-48-17_Pascal3D_NeMo_head_vit_depth_1_slurm': 'DinoV2 + ViT Head + Epochs 50',
    #   '12-01_15-48-40_Pascal3D_NeMo_head_vit_depth_1_no_temp_slurm': 'DinoV2 + ViT Head + No Temp. + Epochs 50',
    #   '12-03_16-44-41_Pascal3D_NeMo_epochs100_head_vit_depth_1_slurm': 'DinoV2 + ViT Head + Epochs 100',
    #   '12-03_16-45-03_Pascal3D_NeMo_epochs100_head_vit_depth_1_no_temp_slurm': 'DinoV2 + ViT Head + No Temp. + Epochs 100',
    # }

    # map_runs_names = {
    #     '11-28_19-05-05_Pascal3D_NeMo_backbone_resnet_slurm': 'ResNet-50 (frozen) + 3 ResNetBlocks',
    #     '11-28_18-46-24_Pascal3D_NeMo_backbone_resnet_unfrozen_slurm': 'ResNet-50 (unfrozen) + 3 ResNetBlocks',
    #     # '11-28_19-04-42_Pascal3D_NeMo_backbone_resnet_unfrozen_moving_avg_slurm': 'ResNet-50 (unfrozen) + 3 ResNetBlocks + Moving Avg.',
    #     #'11-28_18-33-11_Pascal3D_NeMo_head_resnet_slurm': 'DinoV2 (frozen) +  3 ResNetBlocks',
    #     #'11-27_10-39-11_Pascal3D_NeMo_backbone_resnet_unfrozen_slurm': 'ResNet-50 (unfrozen) + 3 ResNetBlocks',
    #     #'11-27_10-38-23_Pascal3D_NeMo_head_resnet_slurm': 'DinoV2 (frozen) +  3 ResNetBlocks',
    #     #'11-27_09-38-56_Pascal3D_NeMo_backbone_resnet_slurm': 'ResNet-50 (frozen) + 3 ResNetBlocks',
    #     '11-28_19-04-53_Pascal3D_NeMo_head_resnet_slurm': 'DinoV2 (frozen) +  3 ResNetBlocks',
    #     '11-27_09-38-34_Pascal3D_NeMo_head_vit_depth_0_slurm': 'DinoV2 (frozen) + 1 LinearLayer',
    #     '11-27_09-38-23_Pascal3D_NeMo_head_vit_depth_1_slurm': 'DinoV2 (frozen) + 1 LinearLayer + 1 ViTBlocks',
    #     '11-27_09-38-45_Pascal3D_NeMo_head_vit_depth_2_slurm': 'DinoV2 (frozen) + 1 LinearLayer + 2 ViTBlocks',
    #     '11-27_09-38-11_Pascal3D_NeMo_head_vit_depth_5_slurm': 'DinoV2 (frozen) + 1 LinearLayer + 5 ViTBlocks',
    # }

    # name_regex = '.*Pascal3D_NeMo_(feat.*|)_(torque|slurm)'
    # map_runs_names = {
    #     '12-05_14-19-57_Pascal3D_NeMo_feat256_torque': 'Feats. Dim. 256',
    #     '12-05_14-20-23_Pascal3D_NeMo_feat128_torque': 'Feats. Dim. 128',
    #     '12-05_14-20-10_Pascal3D_NeMo_feat64_torque': 'Feats. Dim. 64',
    #     '12-06_10-06-19_Pascal3D_NeMo_feat32_slurm': 'Feats. Dim. 32',
    # }

    # name_regex = '.*Pascal3D_NeMo_(geodesic.*|)_slurm'
    # map_runs_names = {
    #     '12-06_09-52-59_Pascal3D_NeMo_geodesic_prob_sigma_0_slurm': 'Geodesic Prob. Sigma 0%',
    #     '12-06_09-52-24_Pascal3D_NeMo_geodesic_prob_sigma_0005_slurm': 'Geodesic Prob. Sigma 0.5%',
    #     '12-06_09-53-10_Pascal3D_NeMo_geodesic_prob_sigma_001_slurm': 'Geodesic Prob. Sigma 1%',
    #     '12-06_09-52-36_Pascal3D_NeMo_geodesic_prob_sigma_002_slurm': 'Geodesic Prob. Sigma 2%',
    #     '12-06_09-52-48_Pascal3D_NeMo_geodesic_prob_sigma_005_slurm': 'Geodesic Prob. Sigma 5%',
    # }

    # name_regex = '.*Pascal3D_NeMo_(uniform.*|epnp.*)_slurm'
    #
    # map_runs_names = {
    #     '12-06_09-37-36_Pascal3D_NeMo_uniform_refine3d_slurm': 'Uniform _ Refine 3D',
    #     '12-06_09-37-47_Pascal3D_NeMo_uniform_refine6d_slurm': 'Uniform + Refine 6D',
    #     '12-06_11-11-56_Pascal3D_NeMo_uniform_depth_from_box_refine3d_slurm': 'Uniform + Depth from BBox + Refine 3D',
    #     '12-06_09-37-02_Pascal3D_NeMo_uniform_depth_from_box_refine6d_slurm': 'Uniform + Depth from BBox + Refine 6D',
    #     '12-06_09-36-39_Pascal3D_NeMo_epnp_refine3d_slurm': 'EPnP + Refine 3D',
    #     '12-06_09-37-13_Pascal3D_NeMo_epnp_refine6d_slurm': 'EPnP + Refine 6D',
    #     '12-06_09-36-51_Pascal3D_NeMo_epnp_refine6d_epochs60_slurm': 'EPnP + 2x Refine 6D',
    #     '12-06_09-37-59_Pascal3D_NeMo_uniform_refine3d_only_rendered_slurm': 'Uniform + Refine 3D (Only Rendered)',
    # }


    metrics = ['test/pascal3d_test/label/acc', 'test/pascal3d_test/pose/acc_pi6', 'test/pascal3d_test/pose/acc_pi18']
    metrics_names = ['CLS [%]', '3D Pose PI/6 [%]', '3D Pose PI/18 [%]']

    df = get_dataframe(name_regex=name_regex, metrics=metrics, age_in_hours_lt=age_in_hours)
    # filter pandas df with column name and list map_runs_names.keys()
    if map_runs_names is not None:
        df = df[df['name'].isin(map_runs_names.keys())]
        df = df.replace({'name': map_runs_names})

    df = df.set_index('name')
    if map_runs_names is not None:
        df = df.reindex(map_runs_names.values())
    df = df * 100.

    # having two lists metrics, metrics_names, make a dictionary map_metrics
    map_metrics = dict(zip(metrics, metrics_names))
    df.rename(columns=map_metrics, inplace=True)
    logger.info(tabulate(df, headers='keys', tablefmt='latex', floatfmt=f".{digits}f"))

    import matplotlib.pyplot as plt
    # change from matplotlib the switch backend to interactive tkinter
    plt.switch_backend('TkAgg')
    # using matplotlib plot the dataframe transposed
    df.transpose().plot.bar(rot=0)
    # add value for each bar
    for p in plt.gca().patches:
        plt.gca().annotate(f"{p.get_height():.{digits}f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # ensure that legend is not inside bars
    plt.legend(loc='upper right') # , bbox_to_anchor=(0.0, 0.5))

    # ensure that figure is large enough to cover the legend bars, and text
    plt.gcf().set_size_inches(20, 7)

    # show the plot
    plt.show()


@app.command()
def pose_categories(
        dataset: str = typer.Option('pascal3d', '-d', '--dataset'),
        metric: str = typer.Option('pi6', '-m', '--metric'),
        name_regex: str = typer.Option('CO3D_NeMo', '-n', '--name'),
        age_in_hours: int = typer.Option(48, '-h', '--hours')):

    metric = 'pi6' # 'pi6', 'pi12', 'pi18',
    datasets = ['objectnet3d', 'pascal3d', 'co3d_20'] # 'pascal3d' 'co3d_20' 'co3d_28' 'objectnet3d'
    #datasets = ['co3d_28'] # 'pascal3d' 'co3d_20' 'co3d_28' 'objectnet3d'

    # tablefmt 'latex', 'plain', 'tsv',
    tablefmt = 'tsv'

    #name_regex = f'.*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_filtered_mesh.*'
    # name_regex = f'.*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_mesh.*'
    name_regex = f'.*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_slurm'

    name_regex = f'.*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_uniform_refine3d_slurm'
    name_regex = f'.*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_slurm'
    # name_regex = f'.*_CO3D_NeMo_uniform_refine3d_cat1_([a-z]*)_ref([0-9]*)_slurm'
    name_regex = f'.*_CO3D_NeMo_seqs5_epochs20_cat1_([a-z]*)_ref([0-9]*)_slurm'
    name_regex = f'.*_CO3D_NeMo_seqs5_epochs20_cat1_([a-z]*)_ref([0-9]*)_slurm'

    #name_regex = f'.*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_epnp_refine6d_slurm'
    #name_regex = f'.*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_epnp_refine6d_only_rendered_no_clutter_mesh_slurm'

    name_regex = f'.*_CO3D_Regression_cat1_([a-z]*)_ref([0-9]*)_slurm'

    #name_regex = f'.*_CO3D_NeMo_epnp_refine6d_cat1_([a-z]*)_ref([0-9]*)_slurm'
    name_regex = f'.*_CO3D_NeMo_gaussian_cat1_([a-z]*)_ref([0-9]*)_slurm'
    # 01-29_10-20-46_CO3D_NeMo_vMF_normalize_verts_cat1_hydrant_ref1_slurm

    name_regex = f'.*_CO3D_NeMo_vMF_normalize_cat1_([a-z]*)_ref([0-9]*)_slurm'
    name_regex = f'.*_CO3D_NeMo_vMF_normalize_verts_cat1_([a-z]*)_ref([0-9]*)_slurm'
    # name_regex = f'.*_CO3D_NeMo_vMF_normalize_img_cat1_([a-z]*)_ref([0-9]*)_slurm'
    # name_regex = f'.*_CO3D_NeMo_vMF_normalize_cat1_([a-z]*)_ref([0-9]*)_slurm'


    # cat1_hairdryer_ref0_epnp_refine6d_no_clutter_slurm
    # name_regex = f'.*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_epnp_refine6d_only_rendered_no_clutter_filtered_mesh_slurm'

    #name_regex = f'.*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_uniform_refine3d_only_rendered_slurm'
    #name_regex = f'.*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_uniform_refine3d_only_rendered_no_clutter_slurm'
    # 01-13_00-32-54_CO3D_NeMo_cat1_motorcycle_ref0_uniform_refine3d_only_rendered_no_clutter_slurm

    # only_rendered_slurm
    #name_regex = f'11-1[45].*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_filtered_mesh.*'

    # name_regex = f'.*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_.*'
    #name_regex = '11-12_15-25-58_CO3D_ZSP_local' # zsp 10s to 5s
    # name_regex = '.*CO3D_NeMo_Align3D.*'
    #name_regex = '11-14_09-03-27_CO3Dv1_NeMo_Align3D_local' # ours 10 to 10
    #name_regex = '11-13_22-16-37_CO3D_NeMo_Align3D_local' # ours 10 to 5
    #name_regex = '11-14_22-56-59_CO3Dv1_NeMo_Align3D_slurm' # ours 10 to 10
    #name_regex = '11-14_22-57-20_CO3D_NeMo_Align3D_slurm' # ours 10 to 5
    # name_regex = '11-12_21-21-36_CO3D_ZSP_cross_pascal3d_objectnet3d_local'
    # name_regex = '11-14_23-34-48_CO3D_ZSP_cross_pascal3d_objectnet3d_local'
    for dataset in datasets:
        if 'cat1' in name_regex:
            MAP_DATASET_TO_METRIC = {
                DATASET_PASCAL3D: f'test/pascal3d_test/pose/acc_{metric}',
                DATASET_OBJECTNET3D: f'test/objectnet3d_test/pose/acc_{metric}',
                DATASET_CO3D_28: f'test/co3d_no_zsp_5s_labeled_ref/pose/acc_{metric}',
                DATASET_CO3D_20: f'test/co3dv1_10s_zsp_labeled_cuboid_ref/pose/acc_{metric}'
            }
            metrics_scales = [100]
            metrics = [MAP_DATASET_TO_METRIC[dataset]]
            metrics_dfs = get_categorical_results_from_multiple_runs(metrics=metrics, metrics_scales=metrics_scales,
                                                                     age_in_hours=age_in_hours, name_regex=name_regex)
        else:
            MAP_DATASET_TO_METRIC = {
                DATASET_PASCAL3D: f'test/pascal3d_test/pose/prefix/CATEGORY_acc_{metric}',
                DATASET_OBJECTNET3D: f'test/objectnet3d_test/pose/prefix/CATEGORY_acc_{metric}',
                DATASET_CO3D_28: f'pose/prefix/CATEGORY_acc_{metric}',
                DATASET_CO3D_20: f'pose/prefix/CATEGORY_acc_{metric}',
            }

            MAP_DATASET_TO_CATEGORIES = {
                DATASET_PASCAL3D: TABLE_CATEGORIES_PASCAL3D[:-1],
                DATASET_OBJECTNET3D: TABLE_CATEGORIES_OBJECTNET3D_20[:-1],
                DATASET_CO3D_28: TABLE_CATEGORIES_CO3D_28[:-1],
                DATASET_CO3D_20:  TABLE_CATEGORIES_CO3D_20[:-1],
            }

            metrics_scales = [100]
            metrics = [MAP_DATASET_TO_METRIC[dataset]]
            categories = MAP_DATASET_TO_CATEGORIES[dataset]

            metrics_dfs = get_categorical_results_from_single_runs(metrics_templates=metrics, categories=categories,
                                                                   age_in_hours=age_in_hours, name_regex=name_regex,
                                                                   metrics_scales=metrics_scales)

        tabulate_metrics_for_dataset(metrics_dfs=metrics_dfs, datasets=[dataset], digits=1, tablefmt=tablefmt)



@app.command()
def ablation_dist():
    # CO3Dv1_NeMo, metrics

    categories = od3d.io.read_config_intern(Path('datasets/categories/zsp.yaml'))
    # categories = TABLE_CATEGORIES_CO3D_28[:-1]
    #categories = ['bottle', 'couch', 'motorcycle', 'laptop']
    align3d_1on1_name_partial = '.*NeMo_Align3D_dist_cyc.*'
    align3d_1on1_metrics = ['pose/acc_pi6', 'pose/acc_pi18']
    align3d_1on1_columns_map = {}
    align3d_1on1_columns_map[align3d_1on1_metrics[-2]] = "Acc. Pi/6. [%]"
    align3d_1on1_columns_map[align3d_1on1_metrics[-1]] = "Acc. Pi/18. [%]"
    age_in_hours = 100
    configs = ['ablation_name']
    #configs = ['method.dist_appear_weight']
    #align3d_1on1_columns_map['method.dist_appear_weight'] = 'Appear. Weight'
    align3d_1on1_columns_map['ablation_name'] = 'Name'
    for category in categories:
        align3d_1on1_metrics.append(f'pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi6')
        align3d_1on1_metrics.append(f'pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi18')
        align3d_1on1_columns_map[align3d_1on1_metrics[-2]] = category
        align3d_1on1_columns_map[align3d_1on1_metrics[-1]] = category

    align3d_1on1_df = get_dataframe(configs=configs, metrics=align3d_1on1_metrics, age_in_hours_lt=age_in_hours, name_regex=align3d_1on1_name_partial)
    #align3d_1on1_df['ablation_name'] = align3d_1on1_df['ablation_name'].values
    #align3d_1on1_df['ablation_name'] = align3d_1on1_df['ablation_name'].str
    align3d_1on1_df = align3d_1on1_df.rename(columns=align3d_1on1_columns_map)

    max_index = align3d_1on1_df['pose/acc_pi6'].idxmax()
    # Get the row with the maximum value in 'Column1'
    max_row = align3d_1on1_df.loc[max_index]

    cols_dec = ["Acc. Pi/6. [%]", 'Acc. Pi/18. [%]', 'bicycle', 'hydrant', 'motorcycle', 'teddybear', 'toaster'] # , "Acc. Pi/18. [%]"
    align3d_1on1_df[cols_dec] *= 100.
    align3d_1on1_df = align3d_1on1_df.loc[align3d_1on1_df['Name'].notna()]
    align3d_1on1_df['Name'] = [row['value'] for row in align3d_1on1_df['Name']]
    cols = ['Name', "Acc. Pi/6. [%]", 'Acc. Pi/18. [%]'] # , "Acc. Pi/18. [%]"

    cols = ['Name', "Acc. Pi/6. [%]", 'Acc. Pi/18. [%]', 'bicycle', 'hydrant', 'motorcycle', 'teddybear', 'toaster'] # , "Acc. Pi/18. [%]"

    cols = ["Acc. Pi/6. [%]", 'Acc. Pi/18. [%]', 'bicycle', 'hydrant', 'motorcycle', 'teddybear', 'toaster'] #  "Acc. Pi/18. [%]",
    #cols = ['Run', "Acc. Pi/6. [%]", 'bicycle', 'motorcycle', 'car', 'chair']

    align3d_1on1_df = align3d_1on1_df.set_index("Name")
    align3d_1on1_df = align3d_1on1_df.loc[[
        'resnet50_acc',
        'dino_vits8_acc',
        'dinov2_vitb14_acc',
        'dinov2_vits14_acc',

        # 'dinov2_dist_min',
        # 'dinov2_dist_avg',
        # 'dinov2_avg',
        # 'dinov2_avg_norm',
        # 'dinov2_dist_appear_weight_00',
        # 'dinov2_dist_appear_weight_01',
        # 'dinov2_dist_appear_weight_02',
        # 'dinov2_dist_appear_weight_03',
        # 'dinov2_dist_appear_weight_04',
        # 'dinov2_dist_appear_weight_05',
        # 'dinov2_dist_appear_weight_06',
        # 'dinov2_dist_appear_weight_07',
        # 'dinov2_dist_appear_weight_08',
        # 'dinov2_dist_appear_weight_09',
        # 'dinov2_dist_appear_weight_10',
    ]]

    df = align3d_1on1_df[cols]
    logger.info(tabulate(df, headers='keys', tablefmt='latex',  floatfmt=".1f")) # 'github', 'tsv', 'latex', 'latex_raw'
    logger.info(tabulate(align3d_1on1_df, headers='keys', tablefmt='latex',  floatfmt=".1f")) # 'github', 'tsv', 'latex', 'latex_raw'

from typing import List
def get_categorical_results_from_multiple_runs(metrics, age_in_hours: float, configs=[], name_regex=f'.*cat1_([a-z]*)_ref([0-9]*)_.*', metrics_scales=None):
    rows = []
    COLUMN_CATEGORY = "category"
    COLUMN_REFERENCE = "ref"
    COLUMN_CATEGORY_MEAN = "mean"
    COLUMN_INDEX = "index"
    df = get_dataframe(configs=configs, metrics=metrics, age_in_hours_lt=age_in_hours,
                       name_regex=name_regex, name_regex_groups=[COLUMN_CATEGORY, COLUMN_REFERENCE], filter_runs_with_metrics=False)
    metrics_dfs = []
    for m, metric in enumerate(metrics):
        if metrics_scales is not None and len(metrics_scales) > m:
            metric_scale = metrics_scales[m]
        else:
            metric_scale = 1.
        metric_df = df[df[metric].notnull()]
        metric_df = metric_df[[metric, COLUMN_CATEGORY, COLUMN_REFERENCE]]
        metric_df = metric_df.drop_duplicates(subset=[COLUMN_CATEGORY, COLUMN_REFERENCE], keep="first")
        metric_df_mean_over_refs = metric_df.groupby(COLUMN_CATEGORY)[metric].mean(numeric_only=False) * metric_scale
        # metric_df.groupby(COLUMN_CATEGORY).count()
        metric_df_mean_over_refs[COLUMN_CATEGORY_MEAN] = metric_df_mean_over_refs.mean()
        metric_df_std_over_refs = metric_df.groupby(COLUMN_CATEGORY)[metric].std(numeric_only=False) * metric_scale
        metric_df_std_over_refs[COLUMN_CATEGORY_MEAN] = metric_df_std_over_refs.mean()
        metric_df = pd.DataFrame({'mean': metric_df_mean_over_refs, 'std': metric_df_std_over_refs}).reset_index()
        metric_df = metric_df.set_index(COLUMN_CATEGORY).transpose()
        metrics_dfs.append(metric_df)

    return metrics_dfs

def get_categorical_results_from_single_runs(metrics_templates, categories, age_in_hours: float, configs=[], name_regex='.*_CO3D_NeMo_ref.*', metrics_scales=None, map_od3d_to_datasets=None):
    metrics_df = []
    for m, metric in enumerate(metrics_templates):
        columns_map = {}
        metrics = []  # ['pose/acc_pi6', 'pose/acc_pi6_std']
        columns_std = []
        columns_mean = []
        columns_mean_map = {}
        columns_std_map = {}

        if map_od3d_to_datasets is None or len(map_od3d_to_datasets) <= m:
            map_od3d_to_dataset = MAP_CATEGORIES_OD3D_TO_CO3D
        else:
            map_od3d_to_dataset = map_od3d_to_datasets[m]
        for category in categories:
            if category not in map_od3d_to_dataset:
                map_od3d_to_dataset[category] = category # f'pose/prefix/{map_od3d_to_dataset[category]}_acc_pi6' # f'pose/prefix/{map_od3d_to_dataset[category]}_acc_pi6_std'
            metrics.append(metric.replace('CATEGORY', map_od3d_to_dataset[category]))
            columns_map[metrics[-1]] = category + '_mean'
            columns_mean.append(columns_map[metrics[-1]])
            columns_mean_map[columns_map[metrics[-1]]] = category
            metrics.append(metric.replace('CATEGORY', map_od3d_to_dataset[category]) + '_std')
            columns_map[metrics[-1]] = category + '_std'
            columns_std.append(columns_map[metrics[-1]])
            columns_std_map[columns_map[metrics[-1]]] = category

        df = get_dataframe(configs=configs, metrics=metrics, age_in_hours_lt=age_in_hours, name_regex=name_regex)
        df = df.rename(columns=columns_map)

        if metrics_scales is not None and len(metrics_scales) > m:
            metric_scale = metrics_scales[m]
        else:
            metric_scale = 1.
        df = df * metric_scale
        df = df.iloc[-1:] # [:1] or [-1:]
        df = pd.concat([df[columns_mean].rename(columns=columns_mean_map), df[columns_std].rename(columns=columns_std_map)])
        df['category'] = ['mean', 'std']
        df = df.set_index('category')
        metrics_df.append(df)
        # metric_df = metric_df.set_index(COLUMN_CATEGORY).transpose()
    return metrics_df


@app.command()
def pose_pi6_categories_separate():
    COLUMN_ACC_PI6 = "Acc. Pi/6. [%]"
    logging.basicConfig(level=logging.INFO)
    #import wandb
    categories20 = od3d.io.read_config_intern(Path('datasets/categories/zsp.yaml'))
    categories28 = od3d.io.read_config_intern(Path('datasets/categories/cross.yaml'))
    from od3d.datasets.co3d.enum import MAP_CATEGORIES_OD3D_TO_CO3D
    from od3d.datasets.pascal3d.enum import MAP_CATEGORIES_OD3D_TO_PASCAL3D
    #categories_co3d = [MAP_CATEGORIES_OD3D_TO_CO3D[cat] for cat in categories]
    #categories_pascal3d = [MAP_CATEGORIES_OD3D_TO_PASCAL3D[cat] for cat in categories]
    #config = od3d.io.load_hierarchical_config()

    # 08-14_10-02-12_CO3D_NeMo_use_mask_rgb_and_object_slurm
    # 08-14_09-05-31_CO3D_NeMo_moving_average_slurm
    # 08-11_20-47-23_CO3D_NeMo_cross_entropy_bank_loss_gradient_slurm
    # voge:    bed shelf     calculator cellphone computer cabinet        guitar iron knife oven      pen pot rifle slipper stove toilet tub wheelchair
    # starmap: bed bookshelf calculator cellphone computer filing cabinet guitar iron knife microwave pen pot rifle slipper stove toilet tub wheelchair
                # aero bike boat bottle bus car chair table mbike sofa train tv mean
    from od3d.datasets.co3d.enum import MAP_CATEGORIES_OD3D_TO_CO3D


    # age_in_hours = 20 # 17
    # configs = []
    #
    # # ##### FROM MULTIPLE RUNS
    metrics_dataset = [DATASET_PASCAL3D, DATASET_OBJECTNET3D, DATASET_CO3D_28, DATASET_CO3D_20]
    metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/objectnet3d_test/pose/acc_pi6', 'test/co3d_no_zsp_5s_labeled_ref/pose/acc_pi6', 'test/co3dv1_10s_zsp_labeled_cuboid_ref/pose/acc_pi6']

    # metrics = ['test/pascal3d_test/pose/acc_pi18', 'test/objectnet3d_test/pose/acc_pi18', 'test/co3d_no_zsp_5s_labeled_ref/pose/acc_pi18', 'test/co3dv1_10s_zsp_labeled_cuboid_ref/pose/acc_pi18']

    metrics_scales = [100, 100, 100, 100]
    metrics_names = ['PASCAL3D [%]', 'ObjectNet3D [%]', 'CO3D 5s [%]', 'CO3D ZSP 10s [%]']
    name_partial = '_CO3D_NeMo_'
    name_regex = f'.*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_mesh.*'
    name_regex = f'.*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_filtered_mesh.*'
    #name_regex = f'.*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_filtered_cuboid.*'
    #name_regex = f'.*_CO3D_NeMo_cat1_([a-z]*)_ref([0-9]*)_.*'
    metrics_dfs = get_categorical_results_from_multiple_runs(metrics=metrics, metrics_scales=metrics_scales, age_in_hours=age_in_hours, configs=configs, name_regex=name_regex)

    # #
    # #
    # # # ###### FROM SINGLE RUNS
    metrics = ['pose/prefix/CATEGORY_acc_pi6']
    metrics = ['test/pascal3d_test/pose/prefix/CATEGORY_acc_pi6']
    # metrics = ['test/objectnet3d_test/pose/prefix/CATEGORY_acc_pi6']

    metrics_dataset = [DATASET_CO3D_20]
    #metrics_dataset = [DATASET_CO3D_28]
    metrics_dataset = [DATASET_PASCAL3D]
    #metrics_dataset = [DATASET_OBJECTNET3D]

    metrics_scales = [100]
    categories = TABLE_CATEGORIES_CO3D_20[:-1]
    categories = TABLE_CATEGORIES_CO3D_28[:-1]
    categories = TABLE_CATEGORIES_PASCAL3D[:-1]
    # categories = TABLE_CATEGORIES_OBJECTNET3D_11[:-1]
    # categories = TABLE_CATEGORIES_OBJECTNET3D_9[:-1]
    # categories = TABLE_CATEGORIES_OBJECTNET3D_20[:-1]
    # #categories = TABLE_CATEGORIES_OBJECTNET3D_3[:-1]

    map_od3d_to_datasets = [MAP_CATEGORIES_OD3D_TO_CO3D]

    name_partial = '_CO3Dv1_NeMo_Align3D_'
    name_partial = '_CO3D_NeMo_Align3D_'
    name_partial = '11-02_00-32-29_CO3D_ZSP_src_5s_ref_5s_local'
    name_partial = '11-02_22-00-14_CO3D_ZSP_z_cross_pascal3d_objectnet3d_local'
    name_partial = '11-05_17-18-19_CO3D_ZSP_z_cross_pascal3d_objectnet3d_local'
    #name_partial = '11-03_08-26-52_CO3D_ZSP_src_5s_ref_5s_local'
    name_partial = '11-11_13-11-29_CO3Dv1_NeMo_Align3D_local' # 1 to 1
    name_partial = '11-11_13-09-40_CO3Dv1_NeMo_Align3D_local' # 10 to 1
    name_regex = '11-12_10-11-07_CO3D_NeMo_Align3D_local' #  ours 10s to 5
    name_regex = '11-12_15-25-58_CO3D_ZSP_local' # zsp 10s to 5s
    # name_regex = '.*CO3D_NeMo_Align3D.*'
    #name_regex = '11-14_09-03-27_CO3Dv1_NeMo_Align3D_local' # ours 10 to 10
    #name_regex = '11-13_22-16-37_CO3D_NeMo_Align3D_local' # ours 10 to 5
    #name_regex = '11-14_22-56-59_CO3Dv1_NeMo_Align3D_slurm' # ours 10 to 10
    #name_regex = '11-14_22-57-20_CO3D_NeMo_Align3D_slurm' # ours 10 to 5
    name_regex = '11-12_21-21-36_CO3D_ZSP_cross_pascal3d_objectnet3d_local'
    age_in_hours = 230
    metrics_dfs = get_categorical_results_from_single_runs(metrics_templates=metrics, categories=categories, age_in_hours=age_in_hours, name_regex=name_regex, metrics_scales=metrics_scales, map_od3d_to_datasets=map_od3d_to_datasets)


def tabulate_metrics_for_dataset(metrics_dfs, datasets: List[str], digits=2, tablefmt='latex'):
    ALLOW_CATEGORIES = None
    # ALLOW_CATEGORIES = ['hairdryer', 'bicycle', 'suitcase']
    BAN_CATEGORIES = [] #  ['tv', 'motorcycle']
    _TABLE_CATEGORIES_PASCAL3D = list(filter(lambda category: category not in BAN_CATEGORIES, TABLE_CATEGORIES_PASCAL3D))
    #_TABLE_CATEGORIES_OBJECTNET3D_3 = list(filter(lambda category: category not in BAN_CATEGORIES, TABLE_CATEGORIES_OBJECTNET3D_3))
    _TABLE_CATEGORIES_OBJECTNET3D_20 = list(filter(lambda category: category not in BAN_CATEGORIES, TABLE_CATEGORIES_OBJECTNET3D_20))
    _TABLE_CATEGORIES_OBJECTNET3D_11 = list(filter(lambda category: category not in BAN_CATEGORIES, TABLE_CATEGORIES_OBJECTNET3D_11))
    _TABLE_CATEGORIES_OBJECTNET3D_9 = list(filter(lambda category: category not in BAN_CATEGORIES, TABLE_CATEGORIES_OBJECTNET3D_9))
    _TABLE_CATEGORIES_CO3D_20 = list(filter(lambda category: category not in BAN_CATEGORIES, TABLE_CATEGORIES_CO3D_20))
    _TABLE_CATEGORIES_CO3D_28 = list(filter(lambda category: category not in BAN_CATEGORIES, TABLE_CATEGORIES_CO3D_28))
    _TABLE_CATEGORIES_ZSP = list(filter(lambda category: category not in BAN_CATEGORIES, TABLE_CATEGORIES_ZSP))
    _TABLE_CATEGORIES_YOLO = list(filter(lambda category: category not in BAN_CATEGORIES, TABLE_CATEGORIES_YOLO))
    if ALLOW_CATEGORIES is not None:
        _TABLE_CATEGORIES_PASCAL3D = list(
            filter(lambda category: category in ALLOW_CATEGORIES or category.startswith('mean'), _TABLE_CATEGORIES_PASCAL3D))
        _TABLE_CATEGORIES_OBJECTNET3D_20 = list(
            filter(lambda category: category in ALLOW_CATEGORIES or category.startswith('mean'), _TABLE_CATEGORIES_OBJECTNET3D_20))
        _TABLE_CATEGORIES_OBJECTNET3D_9 = list(
            filter(lambda category: category in ALLOW_CATEGORIES or category.startswith('mean'), _TABLE_CATEGORIES_OBJECTNET3D_9))
        _TABLE_CATEGORIES_OBJECTNET3D_11 = list(
            filter(lambda category: category in ALLOW_CATEGORIES or category.startswith('mean'), _TABLE_CATEGORIES_OBJECTNET3D_11))
        _TABLE_CATEGORIES_CO3D_20 = list(
            filter(lambda category: category in ALLOW_CATEGORIES or category.startswith('mean'), _TABLE_CATEGORIES_CO3D_20))
        _TABLE_CATEGORIES_CO3D_28 = list(
            filter(lambda category: category in ALLOW_CATEGORIES or category.startswith('mean'), _TABLE_CATEGORIES_CO3D_28))
        _TABLE_CATEGORIES_ZSP = list(filter(lambda category: category in ALLOW_CATEGORIES or category.startswith('mean'), _TABLE_CATEGORIES_ZSP))
        _TABLE_CATEGORIES_YOLO = list(filter(lambda category: category in ALLOW_CATEGORIES or category.startswith('mean'), _TABLE_CATEGORIES_YOLO))


    for m, metric_df in enumerate(metrics_dfs):
        if datasets[m] == DATASET_PASCAL3D:
            logger.info(f'PASCAL3D {len(_TABLE_CATEGORIES_PASCAL3D[:-1])}')
            metric_df[_TABLE_CATEGORIES_PASCAL3D[-1]] = [metric_df[_TABLE_CATEGORIES_PASCAL3D[:-1]].loc['mean'].mean(),
                                                        metric_df[_TABLE_CATEGORIES_PASCAL3D[:-1]].loc['std'].mean()]

            logger.info(tabulate(metric_df[_TABLE_CATEGORIES_PASCAL3D], headers='keys', tablefmt=tablefmt, floatfmt=f".{digits}f"))
        elif datasets[m] == DATASET_OBJECTNET3D:
            zeros_df = pd.DataFrame(columns=_TABLE_CATEGORIES_OBJECTNET3D_20[:-1], index=['mean', 'std'], data=0.)
            metric_df = pd.merge(metric_df, zeros_df[zeros_df.columns.difference(metric_df.columns)], left_index=True, right_index=True, how='outer')
            logger.info('ObjectNet3D')
            for object_net3d_table in [_TABLE_CATEGORIES_OBJECTNET3D_20, _TABLE_CATEGORIES_OBJECTNET3D_11, _TABLE_CATEGORIES_OBJECTNET3D_9]:
                metric_df[object_net3d_table[-1]] = [metric_df[object_net3d_table[:-1]].loc['mean'].mean(),
                                                     metric_df[object_net3d_table[:-1]].loc['std'].mean()]

                logger.info(f'ObjectNet3D {len(object_net3d_table[:-1])} ')

                logger.info(tabulate(metric_df[object_net3d_table], headers='keys', tablefmt=tablefmt, floatfmt=f".{digits}f"))
                #
                # logger.info(f'ObjectNet3D {len(object_net3d_table[:-1])} ')
                # metric_df[object_net3d_table[-1]] = [
                #     metric_df[object_net3d_table[:-1]].loc['mean'].mean(),
                #     metric_df[object_net3d_table[:-1]].loc['std'].mean()
                # ]
                # logger.info(tabulate(metric_df[object_net3d_table], headers='keys', tablefmt='latex', floatfmt=".2f"))

        elif datasets[m] == DATASET_CO3D_20:
            logger.info('CO3D 20')

            metric_df[_TABLE_CATEGORIES_CO3D_20[-1]] = [metric_df[_TABLE_CATEGORIES_CO3D_20[:-1]].loc['mean'].mean(),
                                                       metric_df[_TABLE_CATEGORIES_CO3D_20[:-1]].loc['std'].mean()]
            logger.info('Dataset: ZSP, Categories: ZSP')
            logger.info(tabulate(metric_df[_TABLE_CATEGORIES_ZSP], headers='keys', tablefmt=tablefmt, floatfmt=f".{digits}f"))
            logger.info('Dataset: ZSP, Categories: YOLO')
            logger.info(tabulate(metric_df[_TABLE_CATEGORIES_YOLO], headers='keys', tablefmt=tablefmt, floatfmt=f".{digits}f"))
            logger.info(f'Dataset: ZSP, Categories: {len(_TABLE_CATEGORIES_CO3D_20[:-1])}')
            logger.info(tabulate(metric_df[_TABLE_CATEGORIES_CO3D_20], headers='keys', tablefmt=tablefmt, floatfmt=f".{digits}f"))

        elif datasets[m] == DATASET_CO3D_28:
            metric_df[_TABLE_CATEGORIES_CO3D_20[-1]] = [metric_df[_TABLE_CATEGORIES_CO3D_20[:-1]].loc['mean'].mean(),
                                                       metric_df[_TABLE_CATEGORIES_CO3D_20[:-1]].loc['std'].mean()]
            metric_df[_TABLE_CATEGORIES_CO3D_28[-1]] = [metric_df[_TABLE_CATEGORIES_CO3D_28[:-1]].loc['mean'].mean(),
                                                       metric_df[_TABLE_CATEGORIES_CO3D_28[:-1]].loc['std'].mean()]

            logger.info('Dataset: Ours, Categories: ZSP')
            logger.info(tabulate(metric_df[_TABLE_CATEGORIES_ZSP], headers='keys', tablefmt=tablefmt, floatfmt=f".{digits}f"))
            logger.info('Dataset: Ours, Categories: YOLO')
            logger.info(tabulate(metric_df[_TABLE_CATEGORIES_YOLO], headers='keys', tablefmt=tablefmt, floatfmt=f".{digits}f"))
            logger.info(f'Dataset: Ours, Categories: {len(_TABLE_CATEGORIES_CO3D_28[:-1])}')
            logger.info(tabulate(metric_df[_TABLE_CATEGORIES_CO3D_28], headers='keys', tablefmt=tablefmt, floatfmt=f".{digits}f"))

        else:
            #logger.info(metric_df.to_latex(float_format=".2f"))
            logger.info(tabulate(metric_df, headers='keys', tablefmt=tablefmt, floatfmt=f".{digits}f"))


@app.command()
def pose_pi6_categories_align3d_co3dv1():
    from od3d.datasets.co3d.enum import MAP_CATEGORIES_OD3D_TO_CO3D

    # CO3Dv1_NeMo, metrics
    age_in_hours = 24
    configs = []
    #  ZSP v1
    # CO3Dv1_NeMo, metrics
    name_regex = '10-29_00-20-19_CO3Dv1_NeMo_Align3D_local' # 10-29_00-20-19_CO3Dv1_NeMo_Align3D_local
    columns_map = {}
    metrics = [] #  ['pose/acc_pi6', 'pose/acc_pi6_std']
    columns_std = []
    columns_mean = []
    columns_mean_map = {}
    columns_std_map = {}
    for category in TABLE_CATEGORIES_CO3D_20[:-1]:
        metrics.append(f'pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi6')
        columns_map[metrics[-1]] = category + '_mean'
        columns_mean.append(columns_map[metrics[-1]])
        columns_mean_map[columns_map[metrics[-1]]] = category
        metrics.append(f'pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi6_std')
        columns_map[metrics[-1]] = category + '_std'
        columns_std.append(columns_map[metrics[-1]])
        columns_std_map[columns_map[metrics[-1]]] = category

    columns_mean_map = dict(zip(columns_mean, columns_map.values()))
    columns_std_map = dict(zip(columns_std, columns_map.values()))

    df = get_dataframe(configs=configs, metrics=metrics, age_in_hours_lt=age_in_hours, name_regex=name_regex)
    df = df.rename(columns=columns_map)
    metric_df = pd.concat([df[columns_mean].rename(columns=columns_mean_map), df[columns_std].rename(columns=columns_std_map)])
    metric_df = metric_df * 100.
    metric_df['category'] = ['mean', 'std']
    return metric_df

    #metric_df= metric_df.set_index('category')
    #metric_df = pd.concat([df[columns_mean].rename(columns=columns_mean_map), df[columns_std].rename(columns=columns_std_map)])


@app.command()
def pose_pi6_categories():
    logging.basicConfig(level=logging.INFO)
    #import wandb
    categories = od3d.io.read_config_intern(Path('datasets/categories/zsp.yaml'))

    from od3d.datasets.co3d.enum import MAP_CATEGORIES_OD3D_TO_CO3D
    from od3d.datasets.pascal3d.enum import MAP_CATEGORIES_OD3D_TO_PASCAL3D
    #categories_co3d = [MAP_CATEGORIES_OD3D_TO_CO3D[cat] for cat in categories]
    #categories_pascal3d = [MAP_CATEGORIES_OD3D_TO_PASCAL3D[cat] for cat in categories]
    #config = od3d.io.load_hierarchical_config()

    # 08-14_10-02-12_CO3D_NeMo_use_mask_rgb_and_object_slurm
    # 08-14_09-05-31_CO3D_NeMo_moving_average_slurm
    # 08-11_20-47-23_CO3D_NeMo_cross_entropy_bank_loss_gradient_slurm

    age_in_hours = 24
    configs = []
    #  ZSP v1
    # CO3Dv1_NeMo, metrics
    pascal3d_nemo_name_partial = '10-29_00-20-19_CO3Dv1_NeMo_Align3D_local' # 10-29_00-20-19_CO3Dv1_NeMo_Align3D_local

    pascal3d_nemo_metrics = []
    pascal3d_nemo_columns_map = {}
    #pascal3d_nemo_metrics = ['test/pascal3d_test/pose/acc_pi6']
    #pascal3d_nemo_columns_map[pascal3d_nemo_metrics[-1]] = "Acc. Pi/6. [%]"
    for category in categories:
        pascal3d_nemo_metrics.append(f'test/pascal3d_test/pose/prefix/{category}_acc_pi6')
        pascal3d_nemo_columns_map[pascal3d_nemo_metrics[-1]] = category
    pascal3d_nemo_df = get_dataframe(configs=configs, metrics=pascal3d_nemo_metrics, age_in_hours_lt=age_in_hours, name_regex=pascal3d_nemo_name_partial)
    pascal3d_nemo_df = pascal3d_nemo_df.rename(columns=pascal3d_nemo_columns_map)

    # CO3Dv1_NeMo_Align3D, metrics
    align3d_name_partial = 'CO3Dv1_NeMo_Align3D'
    align3d_metrics = ['only_to_ref/pose/acc_pi6']
    align3d_columns_map = {}
    align3d_columns_map[align3d_metrics[-1]] = "Acc. Pi/6. [%]"
    for category in categories:
        align3d_metrics.append(f'only_to_ref/pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi6')
        align3d_columns_map[align3d_metrics[-1]] = category

    # CO3Dv1_NeMo_Align3D, metrics
    align3d_1on1_name_partial = 'CO3Dv1_NeMo_Align3D'
    align3d_1on1_metrics = ['pose/acc_pi6']
    align3d_1on1_columns_map = {}
    align3d_1on1_columns_map[align3d_1on1_metrics[-1]] = "Acc. Pi/6. [%]"
    for category in categories:
        align3d_1on1_metrics.append(f'pose/prefix/{MAP_CATEGORIES_OD3D_TO_CO3D[category]}_acc_pi6')
        align3d_1on1_columns_map[align3d_1on1_metrics[-1]] = category


    nemo_df = get_dataframe(configs=configs, metrics=nemo_metrics, age_in_hours_lt=age_in_hours, name_partial=nemo_name_partial)
    nemo_df = nemo_df.rename(columns=nemo_columns_map)
    align3d_df = get_dataframe(configs=configs, metrics=align3d_metrics, age_in_hours_lt=age_in_hours, name_partial=align3d_name_partial)
    align3d_df = align3d_df.rename(columns=align3d_columns_map)
    align3d_1on1_df = get_dataframe(configs=configs, metrics=align3d_1on1_metrics, age_in_hours_lt=age_in_hours, name_partial=align3d_1on1_name_partial)
    align3d_1on1_df = align3d_1on1_df.rename(columns=align3d_1on1_columns_map)

    df = pd.concat([nemo_df, align3d_df, align3d_1on1_df, pascal3d_nemo_df])
    cols = ['Run', "Acc. Pi/6. [%]", 'bicycle', 'hydrant', 'motorcycle', 'teddybear', 'toaster']
    cols = ['Run', "Acc. Pi/6. [%]", 'bicycle', 'motorcycle', 'car', 'chair']

    df = df[cols]
    logger.info(tabulate(df, headers='keys', tablefmt='latex',  floatfmt=".3f")) # 'github', 'tsv', 'latex', 'latex_raw'
    # logger.info('\n' + my_df.to_csv(sep='\t', index=False, float_format="%.3f"))
    #logger.info('\n' + my_df.to_csv(sep=',', index=False, float_format="%.3f"))
