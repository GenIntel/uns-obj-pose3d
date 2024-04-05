import numpy
import pandas
import typer
import od3d.io
from omegaconf import OmegaConf
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
from od3d.benchmark.run import bench_single_method_local, bench_single_method_local_separate_venv, bench_single_method_local_docker, torque_run_method_or_cmd, slurm_run_method_or_cmd
import json
app = typer.Typer()
import subprocess
from omegaconf import open_dict
import time

from od3d.cli._platform import get_slurm_jobs_ids

import datetime
import pandas as pd
from pygit2 import Repository

from tabulate import tabulate
import re
import od3d.io

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def get_timestamp_as_string():
    now = datetime.datetime.now()
    timestamp = now.strftime("%m-%d_%H-%M-%S")
    return timestamp

def get_timestamp_from_string(string):
    now = datetime.datetime.now()
    year = now.strftime("%Y")
    timestamp = datetime.datetime.strptime('_'.join(f'{year}-{string}'.split('_')[:2]), "%Y-%m-%d_%H-%M-%S")
    return timestamp

def get_nested_value(data, key):
    keys = key.split('.')  # Split the string key into a list of keys
    value = data
    last_key_is_method = False
    for k in keys:
        if isinstance(value, dict) and 'value' in value and k not in value:
            value = value['value']
        if k in value:
            value = value[k]
        else:
            if last_key_is_method:
                continue
            return None  # Key not found

        if k == 'method':
            last_key_is_method = True
        else:
            last_key_is_method = False

    if isinstance(value, dict) and 'value' in value:
        value = value['value']
    return value

def get_runs_multiple(benchmark: str=None, platform: str = None, ablation:str = None,
                      age_in_hours_gt: int = 0, age_in_hours_lt: int = 1000, state=None):
    run_name_regex = get_run_name_regex(ablation=ablation, platform=platform, benchmark=benchmark)
    return get_runs(name_regex=run_name_regex, age_in_hours_gt=age_in_hours_gt, age_in_hours_lt=age_in_hours_lt, state=state)


def get_runs(name_regex='.*', age_in_hours_gt=0, age_in_hours_lt=1000, state=None):
    logging.basicConfig(level=logging.INFO)
    config = od3d.io.load_hierarchical_config()

    import wandb
    # Initialize wandb
     # wandb.init(project=config.logger.wandb_project_name)

    # Access the API
    api = wandb.Api()

    timestamp_created_lt = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=age_in_hours_gt)).isoformat()
    timestamp_created_gt = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=age_in_hours_lt)).isoformat()

    # config.logger.wandb_project_name
    # Fetch all the runs in your project
    if state is not None:
        runs = api.runs(config.logger.wandb_project_name, filters={
                                "display_name": {"$regex": name_regex},
                                "$and": [{
                                    'created_at': {
                                        "$lt": timestamp_created_lt,
                                        "$gt": timestamp_created_gt,
                                    },
                                "state": {"$regex": state}
                                }]}
                        )
    else:
        runs = api.runs(config.logger.wandb_project_name, filters={
                                "display_name": {"$regex": name_regex},
                                "$and": [{
                                    'created_at': {
                                        "$lt": timestamp_created_lt,
                                        "$gt": timestamp_created_gt,
                                    },
                                }]}
                        )
    return runs

def get_dataframe_multiple(ablation: str=None, platform: str=None, benchmark: str = None,
                           age_in_hours_gt=0, age_in_hours_lt=1000, configs=None, metrics=[],
                           duplicates_keep='all', state=None, add_configs_ablation=True, add_configs_default=True):

    run_name_regex = get_run_name_regex(ablation=ablation, platform=platform, benchmark=benchmark)
    if configs is None:
        configs = []

    if add_configs_ablation:
        configs += get_ablations_configs(ablation=ablation)
    if add_configs_default:
        configs += ['platform.link', 'train_datasets.labeled.class_name', 'method.class_name']

    if ablation is not None:
        ablation_regex_groups = ablation.split(',')
    else:
        ablation_regex_groups = []
    return get_dataframe(configs=configs, metrics=metrics, name_regex=run_name_regex,
                         name_regex_groups=['bench', 'method'] + ablation_regex_groups,
                         age_in_hours_gt=age_in_hours_gt, age_in_hours_lt=age_in_hours_lt,
                         duplicates_keep=duplicates_keep, state=state)


def get_dataframe(configs=[], metrics=[], name_regex='.*', name_regex_groups=[],
                  age_in_hours_lt=1000, age_in_hours_gt=0, name_partial_ban=None,
                  filter_runs_with_metrics=True, duplicates_keep='all', state=None):

    # Initialize wandb
     # wandb.init(project=config.logger.wandb_project_name)

    # # Access the API
    # api = wandb.Api()
    #
    # timestamp_created_gt = (datetime.datetime.now(datetime.timezone.utc) -datetime.timedelta(hours=age_in_hours)).isoformat()
    # # config.logger.wandb_project_name
    # # Fetch all the runs in your project
    # runs = api.runs(config.logger.wandb_project_name, filters={
    #                         "display_name": {"$regex": name_regex},
    #                         "$and": [{
    #                             'created_at': {
    #                                 # "$lt": '2022-03-09T10',
    #                                 "$gt": timestamp_created_gt
    #                             }
    #                         }]}
    #                 )


    runs = get_runs(name_regex=name_regex, age_in_hours_lt=age_in_hours_lt, age_in_hours_gt=age_in_hours_gt,
                    state=state)

    # if name_regex is not None:
    #     #runs_names_regex_matches = [re.match(name_regex, run.name) for run in runs]
    #     #runs = [runs[i] if len(runs_names_regex_matches[i].groups()) >= len(name_regex_groups) else None for i in range(len(runs))]
    #     runs = list(filter(lambda run: re.match(name_regex, run.name) and len(re.match(name_regex, run.name).groups()) >= len(name_regex_groups), runs))
    #     logger.info(f'after filtering name regex {len(runs)}, regex {name_regex}')
    #     #logger.info(runs)

    if name_partial_ban is not None:
        for n in name_partial_ban:
            runs = list(filter(lambda run: n not in run.name, runs))
        #logger.info('after filtering name partial ban...')
        logger.info(f'after filtering name partial ban {len(runs)}')
        #logger.info(runs)

    if filter_runs_with_metrics and metrics is not None:
        #logger.info(metrics)
        #for run in runs:
        #    logger.info(list(run.summary.keys()))
        runs = list(filter(lambda run: all([metric in list(run.summary.keys()) for metric in metrics]), runs))
        logger.info(f'after filtering metrics {len(runs)}')
        #logger.info('after filtering metrics...')
        #logger.info(runs)

    logger.info(f'found {len(runs)} runs')
    runs_names = [run.name for run in runs]
    #logger.info(f'runs: \n{runs_names}')
    rows = []
    for run in runs:
        try:
            run_summary = run.summary
            row = [run.name]
            if len(configs) > 0:
                json_config = json.loads(run.json_config)
            for config in configs:
                row.append(get_nested_value(json_config, key=config))
            for metric in metrics:
                if metric in run_summary:
                    row.append(run_summary[metric])
                else:
                    row.append(numpy.nan)
            if name_regex is not None and len(name_regex_groups) > 0:
                re_matched_groups = re.match(name_regex, run.name).groups()
                for i, name_regex_group in enumerate(name_regex_groups):
                    row.append(re_matched_groups[i])
            else:
                name_regex_groups = []

            rows.append(row)
        except Exception as e:

            logger.warning(f'skipping run {run.name} due to {e}')
    # logger.info(rows)

    #cols = ['name'] + metrics

    cols = ['name'] + configs + metrics + name_regex_groups

    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values('name')
    if duplicates_keep is not None and (duplicates_keep == 'first' or duplicates_keep == 'last'):
        subset_cols = configs + name_regex_groups
        logger.info(subset_cols)
        df = df.drop_duplicates(subset=subset_cols, keep=duplicates_keep)

    #logger.info(tabulate(rows, headers=cols, tablefmt='github',  floatfmt=".3f")) # 'github', 'tsv'
    #logger.info(tabulate(rows, headers=cols, tablefmt='html',  floatfmt=".3f")) # 'github', 'tsv'

    # cols_renames = {
    #     'name': "Run",
    #     'test/pascal3d_test/pose/acc_pi6': "Acc. Pi/6. [%]",
    #     'test/pascal3d_test/pose/acc_pi18': "Acc. Pi/18. [%]",
    #     'test/pascal3d_test/pose/err_median': "Median [deg.]",
    #     'test/pascal3d_test/pose/err_mean': "Mean [deg.]",
    #     'test/pascal3d_test/time_pose': 'Inference Duration [s]',
    #     'test/co3d_5s_test/pose/acc_pi6': "Acc. Pi/6. [%]",
    #     'test/co3d_5s_test/pose/acc_pi18': "Acc. Pi/18. [%]",
    #     'test/co3d_5s_test/pose/err_median': "Median [deg.]",
    #     'test/co3d_5s_test/pose/err_mean': "Mean [deg.]",
    #     'test/co3d_5s_test/time_pose': 'Duration [s]',
    #     'test/co3d_50s_test/pose/acc_pi6': "Acc. Pi/6. [%]",
    #     'test/co3d_50s_test/pose/acc_pi18': "Acc. Pi/18. [%]",
    #     'test/co3d_50s_test/pose/err_median': "Median [deg.]",
    #     'test/co3d_50s_test/pose/err_mean': "Mean [deg.]",
    #     'test/co3d_50s_test/time_pose': 'Inference Duration [s]',
    #     'test/co3d/pose/acc_pi6': "Acc. Pi/6. [%]",
    #     'test/co3d/pose/acc_pi18': "Acc. Pi/18. [%]",
    #     'test/co3d/pose/err_median': "Median [deg.]",
    #     'test/co3d/pose/err_mean': "Mean [deg.]",
    #     'test/co3d/time_pose': 'Inference Duration [s]',
    # }
    #
    # cols_scales = {
    #     'test/pascal3d_test/pose/acc_pi6': 100.,
    #     'test/pascal3d_test/pose/acc_pi18': 100.,
    #     'test/co3d_5s_test/pose/acc_pi6': 100.,
    #     'test/co3d_5s_test/pose/acc_pi18': 100.,
    #     'test/co3d_50s_test/pose/acc_pi6': 100.,
    #     'test/co3d_50s_test/pose/acc_pi18': 100.,
    #     'test/co3d/pose/acc_pi6': 100.,
    #     'test/co3d/pose/acc_pi18': 100.,
    # }
    #
    # for col in cols_scales.keys():
    #     if col in df:
    #         df[col] = df[col] * cols_scales[col]
    # df = df.rename(columns=cols_renames)


    return df
@app.command()
def table():
    logging.basicConfig(level=logging.INFO)
    #import wandb
    # config = od3d.io.load_hierarchical_config()

    # 08-14_10-02-12_CO3D_NeMo_use_mask_rgb_and_object_slurm
    # 08-14_09-05-31_CO3D_NeMo_moving_average_slurm
    # 08-11_20-47-23_CO3D_NeMo_cross_entropy_bank_loss_gradient_slurm

    metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/pascal3d_test/pose/acc_pi18', 'test/pascal3d_test/pose/err_median', 'test/pascal3d_test/pose/err_mean', 'test/pascal3d_test/time_pose']
    #metrics = ['test/co3d_5s_test/pose/acc_pi6', 'test/co3d_5s_test/pose/acc_pi18', 'test/co3d_5s_test/pose/err_median', 'test/co3d_5s_test/pose/err_mean', 'test/co3d_5s_test/time_pose']

    #metrics = ['test/co3d_50s_test/pose/acc_pi6', 'test/co3d_50s_test/pose/acc_pi18', 'test/co3d_50s_test/pose/err_median', 'test/co3d_50s_test/pose/err_mean', 'test/co3d_50s_test/time_pose']
    name_partial = '_car_1s_' # None, 'inference', 'split', 'render'
    # configs = ['method.value.multiview.type', 'method.value.multiview.batch_size']
    age_in_hours = 250
    configs = []

    my_df = get_dataframe(configs=configs, metrics=metrics, age_in_hours_lt=age_in_hours, name_partial=name_partial)


    # logger.info(tabulate(my_df, headers='keys', tablefmt='tsv',  floatfmt=".3f")) # 'github', 'tsv'
    # logger.info('\n' + my_df.to_csv(sep='\t', index=False, float_format="%.3f"))
    logger.info('\n' + my_df.to_csv(sep=',', index=False, float_format="%.3f"))


    # my_df.to_csv('output.csv', index=False, header=False, float_format='%.3f')

@app.command()
def table_multiple_sequences():
    logging.basicConfig(level=logging.INFO)

    metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/pascal3d_test/pose/acc_pi18', 'test/pascal3d_test/pose/err_median', 'test/pascal3d_test/pose/err_mean']
    #metrics = ['test/co3d_5s_test/pose/acc_pi6', 'test/co3d_5s_test/pose/acc_pi18', 'test/co3d_5s_test/pose/err_median', 'test/co3d_5s_test/pose/err_mean']
    #metrics = ['test/co3d_50s_test/pose/acc_pi6', 'test/co3d_50s_test/pose/acc_pi18', 'test/co3d_50s_test/pose/err_median', 'test/co3d_50s_test/pose/err_mean']
    #metrics = ['test/co3d/pose/acc_pi6', 'test/co3d/pose/acc_pi18',
    #           'test/co3d/pose/err_median', 'test/co3d/pose/err_mean']

    name_partial = '_car_1s_' # _1s_ 'multiview' _mv6_
    name_partial_ban = ['45s']
    configs = ['method.class_name', 'train_datasets.labeled.categories', 'method.value.multiview.type', 'method.value.multiview.batch_size', 'train_datasets.labeled.dict_nested_frames']
    age_in_hours = 24 * 10

    my_df = get_dataframe(configs=configs, metrics=metrics, age_in_hours_lt=age_in_hours, name_partial=name_partial, name_partial_ban=name_partial_ban)

    my_df['sequence_nth'] = my_df["Run"].str.split('_1s_').str[1:2].str.join('_').str[:3]
    my_df['type'] = my_df["Run"].str[14:].str.split('_1s_').str[0] + my_df["Run"].str.split('_1s_').str[1:2].str.join('_').str[3:]
    my_df['type'] = my_df['type'].str.replace('_slurm', '')
    my_df['type'] = my_df['type'].str.replace('_incr', '')
    my_df['type'] = my_df['type'].str.replace('Incremental', 'Incr.')

    my_df['type'] = my_df['type'].str.replace('_', '\n')
    metrics_new_names = ['Acc. Pi/6. [%]', 'Acc. Pi/18. [%]', 'Median [deg.]', 'Mean [deg.]']
    #my_df[metrics[0]] *= 100
    #my_df[metrics[1]] *= 100

    map_columns = {
        metrics[0]: metrics_new_names[0],
        metrics[1]: metrics_new_names[1],
        metrics[2]: metrics_new_names[2],
        metrics[3]: metrics_new_names[3]
    }

    my_df = my_df.rename(columns=map_columns)
    my_df = my_df.sort_values(by=['type', 'sequence_nth'])
    my_df = my_df.reset_index(drop=True)

    for i, metric in enumerate(metrics_new_names):
        #my_df = my_df.sort_values(by=['category', metric])
        mv_plot = sns.barplot(data=my_df, x="type", y=metric, errorbar=('ci', 100))
        #
        # mv_plot = sns.catplot(
        #     x="type",  # x variable name
        #     y=metric,  # y variable name
        #     hue="method",  # group variable name
        #     data=my_df,  # dataframe to plot
        #     kind="bar",
        #     errorbar=('ci', 100)
        # )

        plt.savefig(f"seqs_type_{metrics[i].replace('/', '_')}.png", bbox_inches='tight')
        plt.clf()

@app.command()
def table_multiple_categories_multiview_incremental():

    logging.basicConfig(level=logging.INFO)

    #metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/pascal3d_test/pose/acc_pi18', 'test/pascal3d_test/pose/err_median', 'test/pascal3d_test/pose/err_mean']
    #metrics = ['test/co3d_5s_test/pose/acc_pi6', 'test/co3d_5s_test/pose/acc_pi18', 'test/co3d_5s_test/pose/err_median', 'test/co3d_5s_test/pose/err_mean']
    #metrics = ['test/co3d_50s_test/pose/acc_pi6', 'test/co3d_50s_test/pose/acc_pi18', 'test/co3d_50s_test/pose/err_median', 'test/co3d_50s_test/pose/err_mean']
    metrics = ['test/co3d/pose/acc_pi6', 'test/co3d/pose/acc_pi18',
               'test/co3d/pose/err_median', 'test/co3d/pose/err_mean']

    name_partial = '_1s_' # _1s_ 'multiview' _mv6_
    name_partial_ban = ['45s']
    configs = ['method.class_name', 'train_datasets.labeled.categories', 'method.value.multiview.type', 'method.value.multiview.batch_size', 'train_datasets.labeled.dict_nested_frames']
    age_in_hours = 24 * 10

    my_df = get_dataframe(configs=configs, metrics=metrics, age_in_hours_lt=age_in_hours, name_partial=name_partial, name_partial_ban=name_partial_ban)

    my_df['sequence_nth'] = my_df["Run"].str.split('_1s_').str[1:2].str.join('_').str[:3]
    my_df['train_datasets.labeled.categories'] = my_df['train_datasets.labeled.categories'].str[0]
    my_df = my_df.groupby(['train_datasets.labeled.categories', 'method.class_name']).head(3)

    # NeMo, NeMo_MultiView, NeMo_Incremental

    metrics_new_names = ['Acc. Pi/6. [%]', 'Acc. Pi/18. [%]', 'Median [deg.]', 'Mean [deg.]']
    #my_df[metrics[0]] *= 100
    #my_df[metrics[1]] *= 100

    map_columns = {
        'train_datasets.labeled.categories': 'category',
        'method.class_name': 'method',
        metrics[0]: metrics_new_names[0],
        metrics[1]: metrics_new_names[1],
        metrics[2]: metrics_new_names[2],
        metrics[3]: metrics_new_names[3]
    }

    my_df = my_df.rename(columns=map_columns)
    my_df = my_df.sort_values(by=['category', 'method', 'sequence_nth'])
    my_df = my_df.reset_index(drop=True)

    for i, metric in enumerate(metrics_new_names):
        #my_df = my_df.sort_values(by=['category', metric])

        mv_plot = sns.catplot(
            x="category",  # x variable name
            y=metric,  # y variable name
            hue="method",  # group variable name
            data=my_df,  # dataframe to plot
            kind="bar",
            errorbar=('ci', 100)
        )

        plt.savefig(f"method_cats_{metrics[i].replace('/', '_')}.png")
        plt.clf()



    my_df.to_csv('output.csv', index=False, header=False)

def save_category_sequence_images_as_one(df: pandas.DataFrame):
    from od3d.datasets.meta import OD3D_Meta
    # df has to contain 'train_datasets.labeled.dict_nested_frames', 'category', 'sequence_nth'
    config = od3d.io.load_hierarchical_config()

    def get_first_non_none_value(dictionary):
        if dictionary is None:
            return None
        for key, value in dictionary.items():
            if value is not None:
                return f'{key}/{list(value.keys())[0]}'
        return None

    sequences_name = [get_first_non_none_value(config) for config in df['train_datasets.labeled.dict_nested_frames']]
    sequences_path_imgs = [Path(config.platform_local.path_datasets).joinpath('CO3D', sequence_name, 'images') if sequence_name is not None else None for sequence_name in sequences_name]
    sequences_fpath_first_imgs = [sorted(list(sequence_path_imgs.iterdir()), key=lambda f: [OD3D_Meta.atoi(val) for val in re.split(r'(\d+)', f.stem)]) if sequence_path_imgs is not None else None for sequence_path_imgs in sequences_path_imgs]
    #sequences_fpath_first_img = [sequence_fpath_first_imgs[0] for sequence_fpath_first_imgs in sequences_fpath_first_imgs]
    df['sequence'] = sequences_name
    df['sequences_fpath_first_imgs'] = sequences_fpath_first_imgs

    df_category_sequence_unique = df.groupby(['category', 'sequence_nth']).head(1)
    df_category_sequence_unique = df_category_sequence_unique.reset_index(drop=True)
    from od3d.cv.visual.show import fpaths_to_rgb, show_img, imgs_to_img
    from od3d.cv.visual.draw import draw_text_in_rgb
    import torch
    category_sequence_rgbs = []
    for i, sequence_fpath_first_imgs in enumerate(df_category_sequence_unique['sequences_fpath_first_imgs']):
        ids = np.linspace(0, len(sequence_fpath_first_imgs)-1, 3, dtype=int)
        category_sequence_rgbs.append(draw_text_in_rgb(fpaths_to_rgb(fpaths=[sequence_fpath_first_imgs[id] for id in ids], H=512, W=512), text=f"\n\n\n{df_category_sequence_unique['category'][i]} {df_category_sequence_unique['sequence_nth'][i]}"))
        # , fpath=f"{df_category_sequence_unique['category'][i]}_{df_category_sequence_unique['sequence_nth'][i]}.png"
    show_img(imgs_to_img(torch.stack(category_sequence_rgbs, dim=0)[:, None]), fpath='category_sequences.png')

@app.command()
def table_multiple_categories():
    logging.basicConfig(level=logging.INFO)
    # config = od3d.io.load_hierarchical_config()

    #metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/pascal3d_test/pose/acc_pi18', 'test/pascal3d_test/pose/err_median', 'test/pascal3d_test/pose/err_mean']
    #metrics = ['test/co3d_5s_test/pose/acc_pi6', 'test/co3d_5s_test/pose/acc_pi18', 'test/co3d_5s_test/pose/err_median', 'test/co3d_5s_test/pose/err_mean']
    #metrics = ['test/co3d_50s_test/pose/acc_pi6', 'test/co3d_50s_test/pose/acc_pi18', 'test/co3d_50s_test/pose/err_median', 'test/co3d_50s_test/pose/err_mean']
    metrics = ['test/co3d/pose/acc_pi6', 'test/co3d/pose/acc_pi18',
               'test/co3d/pose/err_median', 'test/co3d/pose/err_mean']

    name_partial = '_1s_' # _1s_ 'multiview' _mv6_
    name_partial_ban = ['MultiView', 'Incremental']
    configs = ['train_datasets.labeled.categories', 'method.value.multiview.type', 'method.value.multiview.batch_size']
    age_in_hours = 24 * 4

    my_df = get_dataframe(configs=configs, metrics=metrics, age_in_hours_lt=age_in_hours, name_partial=name_partial, name_partial_ban=name_partial_ban)

    import seaborn as sns
    import matplotlib.pyplot as plt

    my_df['train_datasets.labeled.categories'] = my_df['train_datasets.labeled.categories'].str[0]
    my_df = my_df.groupby(['train_datasets.labeled.categories']).head(3)

    metrics_new_names = ['Acc. Pi/6. [%]', 'Acc. Pi/18. [%]', 'Median [deg.]', 'Mean [deg.]']
    #my_df[metrics[0]] *= 100
    #my_df[metrics[1]] *= 100

    map_columns = {
        'train_datasets.labeled.categories': 'category',
        metrics[0]: metrics_new_names[0],
        metrics[1]: metrics_new_names[1],
        metrics[2]: metrics_new_names[2],
        metrics[3]: metrics_new_names[3]
    }

    my_df = my_df.rename(columns=map_columns)
    my_df = my_df.sort_values(by=['category'])
    import numpy as np

    for i, metric in enumerate(metrics_new_names):
        my_df = my_df.sort_values(by=['category', metric])

        yerr = np.stack([my_df.groupby('category')[metric].min().to_numpy(), my_df.groupby('category')[metric].max().to_numpy()])

        mv_plot = sns.barplot(data=my_df, x="category", y=metric, errorbar=('ci', 100))

        #mv_plot = sns.catplot(
        #    x="train seqs.",  # x variable name
        #    y=metric,  # y variable name
        #    hue="inference type",  # group variable name
        #    data=my_df,  # dataframe to plot
        #    kind="bar",
        #)

        plt.savefig(f"single_{metrics[i].replace('/', '_')}.png")
        plt.clf()



    my_df.to_csv('output.csv', index=False, header=False)


@app.command()
def table_multiview_sequences():
    logging.basicConfig(level=logging.INFO)
    # config = od3d.io.load_hierarchical_config()

    #metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/pascal3d_test/pose/acc_pi18', 'test/pascal3d_test/pose/err_median', 'test/pascal3d_test/pose/err_mean']
    #metrics = ['test/co3d_5s_test/pose/acc_pi6', 'test/co3d_5s_test/pose/acc_pi18', 'test/co3d_5s_test/pose/err_median', 'test/co3d_5s_test/pose/err_mean']
    metrics = ['test/co3d_50s_test/pose/acc_pi6', 'test/co3d_50s_test/pose/acc_pi18', 'test/co3d_50s_test/pose/err_median', 'test/co3d_50s_test/pose/err_mean']
    name_partial = 'multiview'
    configs = ['method.value.multiview.type', 'method.value.multiview.batch_size', 'method.value.inference.refine.dims_detached']
    age_in_hours = 6

    my_df = get_dataframe(configs=configs, metrics=metrics, age_in_hours_lt=age_in_hours, name_partial=name_partial)

    import seaborn as sns
    import matplotlib.pyplot as plt

    metrics_new_names = ['Acc. Pi/6. [%]', 'Acc. Pi/18. [%]', 'Median [deg.]', 'Mean [deg.]']
    #my_df[metrics[0]] *= 100
    #my_df[metrics[1]] *= 100

    #my_df['seqs'] = my_df["Run"].str.split('s_').str[0:1]
    map_seqs = {
        '_1st_slurm': '1',
        '_2nd_slurm': '1',
        '_3rd_slurm': '1',
        's8_2_slurm': '2',
        's8_3_slurm': '3',
    }
    my_df.loc[my_df['method.value.inference.refine.dims_detached'].map(len) == 0, 'method.value.multiview.type'] = 'multiview+translation'
    my_df['seqs'] = my_df["Run"].str.split('s_').str[0:2].str.join('_').str[-10:].replace(map_seqs)
    my_df = my_df.loc[my_df['seqs'] != '_bs8_slurm']
    #my_df = my_df.groupby(['method.value.multiview.type', 'seqs']).head(1)
    my_df = my_df.rename(columns={'seqs': 'train seqs.', 'method.value.multiview.type': 'inference type', 'method.value.multiview.batch_size': 'multiview #frames', metrics[0]: metrics_new_names[0], metrics[1]: metrics_new_names[1], metrics[2]: metrics_new_names[2], metrics[3]: metrics_new_names[3]})
    my_df = my_df.sort_values(by=['train seqs.', 'inference type'])

    for i, metric in enumerate(metrics_new_names):
        mv_plot = sns.catplot(
            x="train seqs.",  # x variable name
            y=metric,  # y variable name
            hue="inference type",  # group variable name
            data=my_df,  # dataframe to plot
            kind="bar",
        )
        #fig = mv_plot.get_figure()
        plt.savefig(f"multiview_{metrics[i].replace('/', '_')}.png")

    my_df.to_csv('output.csv', index=False, header=False)


@app.command()
def table_multiview():
    logging.basicConfig(level=logging.INFO)
    # config = od3d.io.load_hierarchical_config()

    #metrics = ['test/pascal3d_test/pose/acc_pi6', 'test/pascal3d_test/pose/acc_pi18', 'test/pascal3d_test/pose/err_median', 'test/pascal3d_test/pose/err_mean']
    #metrics = ['test/co3d_5s_test/pose/acc_pi6', 'test/co3d_5s_test/pose/acc_pi18', 'test/co3d_5s_test/pose/err_median', 'test/co3d_5s_test/pose/err_mean']
    metrics = ['test/co3d_50s_test/pose/acc_pi6', 'test/co3d_50s_test/pose/acc_pi18', 'test/co3d_50s_test/pose/err_median', 'test/co3d_50s_test/pose/err_mean']
    name_partial = 'multiview'
    configs = ['method.value.multiview.type', 'method.value.multiview.batch_size']
    age_in_hours = 2

    my_df = get_dataframe(configs=configs, metrics=metrics, age_in_hours_lt=age_in_hours, name_partial=name_partial)

    import seaborn as sns
    import matplotlib.pyplot as plt

    metrics_new_names = ['Acc. Pi/6. [%]', 'Acc. Pi/18. [%]', 'Median [deg.]', 'Mean [deg.]']
    #my_df[metrics[0]] *= 100
    #my_df[metrics[1]] *= 100

    my_df = my_df.groupby(['method.value.multiview.type', 'method.value.multiview.batch_size']).head(1)
    my_df = my_df.rename(columns={'method.value.multiview.type': 'type', 'method.value.multiview.batch_size': 'multiview #frames', metrics[0]: metrics_new_names[0], metrics[1]: metrics_new_names[1], metrics[2]: metrics_new_names[2], metrics[3]: metrics_new_names[3]})
    my_df = my_df.sort_values(by=['type'])

    for i, metric in enumerate(metrics_new_names):
        mv_plot = sns.catplot(
            x="multiview #frames",  # x variable name
            y=metric,  # y variable name
            hue="type",  # group variable name
            data=my_df,  # dataframe to plot
            kind="bar",
        )
        #fig = mv_plot.get_figure()
        plt.savefig(f"multiview_{metrics[i].replace('/', '_')}.png")

    my_df.to_csv('output.csv', index=False, header=False)

def get_run_name(bench_name: str, method_name: str, platform_name: str, ablation_name: str= None, without_timestamp: bool = False):
    if ablation_name is not None:
        run_name = f'{bench_name}_{method_name}_{ablation_name}_{platform_name}'
    else:
        run_name = f'{bench_name}_{method_name}_{platform_name}'
    if without_timestamp is False:
        run_name = f'{get_timestamp_as_string()}_{run_name}'
    return run_name

def get_run_name_regex(benchmark: str=None, platform: str = None, ablation:str = None):

    ablations_regex = get_ablations_regex(ablation=ablation)

    if benchmark is not None:
        cfg_platform = platform if platform is not None else 'local'
        cfg = od3d.io.load_hierarchical_config(benchmark=benchmark, platform=cfg_platform)
        bench_regex = f'({cfg.train_datasets.labeled.class_name}.*)'
        method_regex = f'({"|".join([method_cfg.class_name for method_cfg in cfg.method.values()])})'
    else:
        bench_regex = '(.*)'
        method_regex = '(.*)'

    platform_regex = platform if platform is not None else '.*'
    run_name_regex = get_run_name(bench_name=bench_regex,
                                  method_name=method_regex,
                                  platform_name=platform_regex,
                                  ablation_name=ablations_regex, without_timestamp=True)
    run_name_regex = f'.*{run_name_regex}'

    logger.info(run_name_regex)
    return run_name_regex

def get_ablations_root_dir():
    file_fpath = Path(__file__).parent.resolve()
    config_dir_rel = "../../../config"
    config_dir_abs = file_fpath.joinpath(config_dir_rel)
    ablations_root_dir = config_dir_abs.joinpath("ablations")
    return ablations_root_dir

def get_ablations_fpaths(ablation: str= None):
    ablations_fpaths_rel = get_ablations_fpaths_rel(ablation=ablation)
    ablations_root_dir = get_ablations_root_dir()
    ablations_fpaths = []
    for ablation_fpaths_rel in ablations_fpaths_rel:
        ablations_fpaths.append([])
        for ablation_fpath_rel in ablation_fpaths_rel:
            ablations_fpaths[-1].append(ablations_root_dir.joinpath(f'{ablation_fpath_rel}.yaml'))
    return ablations_fpaths

def get_ablations_fpaths_rel(ablation:str=None):
    if ablation is not None:
        ablations_root_dir = get_ablations_root_dir()

        # create one config per ablation
        ablation_dirs = [ablations_root_dir.joinpath(a) for a in ablation.split(',')]
        #ablation_dir = ablations_root_dir.joinpath(ablation)

        ablation_fpaths_rel = []
        for ablation_dir in ablation_dirs:
            ablation_fpaths_rel.append([])
            logger.info(f'crwaling ablation directory {ablation_dir}')
            for ablation_file_fpath in ablation_dir.iterdir():
                if ablation_file_fpath.is_dir():
                    continue
                ablation_fpath_rel = ablation_file_fpath.relative_to(ablations_root_dir).with_suffix('')
                if not ablation_fpath_rel.name.startswith("_"):
                    ablation_fpaths_rel[-1].append(ablation_fpath_rel)
    else:
        ablation_fpaths_rel = []

    return ablation_fpaths_rel

def get_ablations_configs(ablation:str=None):
    from od3d.io import read_yaml
    from od3d.data.ext_dicts import unroll_nested_dict

    ablations_configs_fpaths = get_ablations_fpaths(ablation=ablation)
    ablations_configs = []
    for ablation_configs_fpaths in ablations_configs_fpaths:
        for fpath in ablation_configs_fpaths:
            config = dict(read_yaml(fpath, resolve=False)) # problem: ablation not interpolatable
            #logger.info(type(config))
            config = unroll_nested_dict(config, separator='.')
            #logger.info(config.keys())
            ablations_configs +=config.keys()

    ablations_configs = list(set(ablations_configs))
    if 'defaults' in ablations_configs:
        ablations_configs.remove('defaults')

    return ablations_configs

def get_ablations_regex(ablation:str=None):
    ablations_fpaths_rel = get_ablations_fpaths_rel(ablation=ablation)

    ablation_regex = ")_(".join(["|".join([ablation_fpath_rel.stem for ablation_fpath_rel in ablation_fpaths_rel]) for ablation_fpaths_rel in ablations_fpaths_rel])
    if len(ablation_regex) > 0:
        ablation_regex = f'({ablation_regex})'

    return ablation_regex

def get_ablations_fpaths_rel_comb(ablation:str=None):
    import itertools
    ablation_fpaths_rel = get_ablations_fpaths_rel(ablation=ablation)
    combinations_ablation_fpaths_rel = list(itertools.product(*ablation_fpaths_rel))

    logger.info(f'loading {len(combinations_ablation_fpaths_rel)} hierarchical ablations...')
    return combinations_ablation_fpaths_rel

def get_run_name_without_timestamp(run_name: str):
    return run_name[15:]

def get_run_name_without_ts_and_platform(run_name: str):
    return run_name[15:].rsplit('_', 1)[0]

@app.command()
def multiple(benchmark: str = typer.Option('co3d_nemo', '-b', '--benchmark'),
        ablation: str = typer.Option(None, '-a', '--ablation'),
        platform: str = typer.Option('local', '-p', '--platform'),
        age_in_hours_lt: int = typer.Option(24, '-l', '--age-in-hours-lt'),
        sleep_in_mins: int = typer.Option(60, '-s', '--sleep')):
    logging.basicConfig(level=logging.INFO)

    if ablation is None:
        cfgs = [od3d.io.load_hierarchical_config(benchmark=benchmark, platform=platform)]
    else:
        combinations_ablations_fpaths_rel = get_ablations_fpaths_rel_comb(ablation=ablation)
        logger.info(f'loading {len(combinations_ablations_fpaths_rel)} hierarchical ablations...')
        cfgs = od3d.io.load_multiple_hierarchical_configs(benchmark=benchmark, platform=platform,
                                                          multiple_ablations=combinations_ablations_fpaths_rel)

    # create one config per method
    logger.info(f'creating one config per method for {len(cfgs)} configs')
    methods_cfgs = []
    for i, cfg in tqdm(enumerate(cfgs)):
        #logger.info(f'{i} of {len(cfgs)}')
        methods_keys = cfg.method.keys()
        for key in methods_keys:
            method_cfg = cfg.copy()
            method_cfg.method = cfg.method[key]
            # note: not sure why I checked this, this slows down everything tremendously, and
            #       paralellization not straightforward
            # method_cfg_exists = False
            # for prev_method_cfg in methods_cfgs:
            #     if method_cfg == prev_method_cfg:
            #         method_cfg_exists = True
            # if not method_cfg_exists:
            #     methods_cfgs.append(method_cfg)
            methods_cfgs.append(method_cfg)

    print(f"{len(methods_cfgs)} configs with single method.")

    current_branch = Repository('.').head.shorthand  # 'master'

    if ablation is not None:
        prev_runs_lt = get_runs_multiple(benchmark=benchmark, ablation=ablation, age_in_hours_lt=age_in_hours_lt,
                                         state='(finished|running)')
        prev_runs_names_without_ts_and_platform = [get_run_name_without_ts_and_platform(run.name) for run in prev_runs_lt]
    else:
        prev_runs_names_without_ts_and_platform = []

    prev_runs_all = get_dataframe_multiple(benchmark=benchmark, ablation=ablation,
                                           duplicates_keep='last', state='(finished|running)',
                                           add_configs_default=True, add_configs_ablation=False)
    prev_runs_all_names = prev_runs_all['name'].tolist()
    prev_runs_all_names_without_ts_and_platform = \
        [get_run_name_without_ts_and_platform(run_name) for run_name in prev_runs_all_names]
    prev_runs_all_names = prev_runs_all_names[::-1]
    prev_runs_all_names_without_ts_and_platform = prev_runs_all_names_without_ts_and_platform[::-1] # reverse to order from latest to oldest

    started_runs = 0
    for i, method_cfg in tqdm(enumerate(methods_cfgs)):
        #logger.info(f'{i} of {len(methods_cfgs)}')
        with open_dict(method_cfg):
            if method_cfg.get('branch', None) is None:
                method_cfg.branch = current_branch

            ablation_name = method_cfg.get("ablation_name", None)
            run_name = get_run_name(bench_name=method_cfg.train_datasets.labeled.class_name,
                                               method_name=method_cfg.method.class_name,
                                               platform_name=method_cfg.platform.link,
                                               ablation_name=ablation_name)
            run_name_without_ts_and_platform = get_run_name_without_ts_and_platform(run_name)
            if run_name_without_ts_and_platform in prev_runs_names_without_ts_and_platform:
                logger.info(f'{run_name} already exists. Skipping...')
                continue

            checkpoint = method_cfg.method.get('checkpoint', None)
            if checkpoint is not None and '__LAST_RUN__' in checkpoint:
                if run_name_without_ts_and_platform in prev_runs_all_names_without_ts_and_platform:
                    method_cfg.method.checkpoint = method_cfg.method.checkpoint.replace('__LAST_RUN__', prev_runs_all_names[prev_runs_all_names_without_ts_and_platform.index(run_name_without_ts_and_platform)])
                    logger.info(f'Found last checkpoint for {run_name} in {method_cfg.method.checkpoint}')
                else:
                    logger.warning(f'Could not find last checkpoint for {run_name}. Running anyway...')

            method_cfg.run_name = run_name

        if method_cfg.platform.link == 'local':
            bench_single_method_local(method_cfg)
        elif method_cfg.platform.link == 'local-separate-venv':
            bench_single_method_local_separate_venv(method_cfg)
        elif method_cfg.platform.link == 'local-docker':
            bench_single_method_local_docker(method_cfg)
        elif method_cfg.platform.link == 'torque':
            torque_run_method_or_cmd(method_cfg)
        elif method_cfg.platform.link == 'slurm':
            slurm_run_method_or_cmd(method_cfg)
            if (started_runs+1) % 40 == 0:
                time.sleep(sleep_in_mins * 60)
        started_runs += 1

        time.sleep(10)


def get_failed_runs(name_regex='.*', age_in_hours=1000):
    logging.basicConfig(level=logging.INFO)
    runs = get_runs(name_regex=name_regex, age_in_hours_lt=age_in_hours)
    runs = list(filter(lambda run: run.state =='failed' or run.state=='crashed', runs)) #  or run.state =='running'
    # runs_states = [run.state for run in runs]
    runs_names = [run.name for run in runs]
    logger.info(f'found {len(runs)} failed or crashed runs')
    return runs_names

@app.command()
def recent(age_in_hours: int = typer.Option(1000, '-h', '--hours'),
                  name_regex: str = typer.Option('.*', '-n', '--name')):
    runs = get_runs(name_regex=name_regex, age_in_hours_lt=age_in_hours)
    # runs_states = [run.state for run in runs]
    for run in runs:
        logger.info(f'{run.name} {run.state}')

@app.command()
def delete_wandb(age_in_hours_gt: int = typer.Option(0, '-g', '--greater'),
                 age_in_hours_lt: int = typer.Option(1000, '-l', '--lower'),
                 name_regex: str = typer.Option('.*', '-n', '--name')):

    logging.basicConfig(level=logging.INFO)
    runs = get_runs(name_regex=name_regex, age_in_hours_lt=age_in_hours_lt, age_in_hours_gt=age_in_hours_gt)
    logger.info(f'deleting following runs: ')
    for run in runs:
        logger.info(run.name)
        run.delete()

@app.command()
def delete_wandb_failed(age_in_hours_gt: int = typer.Option(0, '-g', '--greater'),
                        age_in_hours_lt: int = typer.Option(1000, '-l', '--lower'),
                        name_regex: str = typer.Option('.*', '-n', '--name')):

    logging.basicConfig(level=logging.INFO)
    runs_failed = get_runs(name_regex=name_regex, age_in_hours_lt=age_in_hours_lt, age_in_hours_gt=age_in_hours_gt, state='(failed|crashed)')

    logger.info(f'deleting following failed runs: ')
    for run in runs_failed:
        logger.info(run.name)
        run.delete()

@app.command()
def delete_wandb_running(age_in_hours_gt: int = typer.Option(0, '-g', '--greater'),
                         age_in_hours_lt: int = typer.Option(1000, '-l', '--lower'),
                         name_regex: str = typer.Option('.*', '-n', '--name')):

    logging.basicConfig(level=logging.INFO)
    runs_running = get_runs(name_regex=name_regex, age_in_hours_lt=age_in_hours_lt, age_in_hours_gt=age_in_hours_gt, state='running')

    logger.info(f'deleting following runs: ')
    for run in runs_running:
        logger.info(run.name)
        run.delete()

@app.command()
def restart_slurm(age_in_hours: int = typer.Option(1000, '-h', '--hours'),
                  name_regex: str = typer.Option('.*', '-n', '--name')):
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    runs_names = get_failed_runs(age_in_hours=age_in_hours, name_regex=name_regex)

    cfg = od3d.io.read_config_intern(rfpath=Path('platform/local.yaml'))
    for run_name in runs_names:
        # cfg.path_home
        local_tmp_config_fpath = Path(cfg.path_home).joinpath('tmp', f'config_{run_name}.yaml')
        try:
            cfg_old = od3d.io.read_config_extern(local_tmp_config_fpath)
            timestamp_str = get_timestamp_as_string()
            cfg_old.run_name = timestamp_str + run_name[len(timestamp_str):]

            logger.info(f'restarting {run_name}...')
            slurm_run_method_or_cmd(cfg_old)
        except Exception as e:
            logger.info(e)
    logger.info(runs_names)

@app.command()
def single_local(config_fpath: str = typer.Option(None, '-c', '--config')):
    logging.basicConfig(level=logging.INFO)
    method_cfg = OmegaConf.load(config_fpath)
    bench_single_method_local(method_cfg)


@app.command()
def test(benchmark: str = typer.Option('timeseries_internal', '-b', '--benchmark'),
         mode: str = typer.Option('local', '-m', '--mode'),
         tasks: str = typer.Option(None, '-t', '--tasks'),
         constraint: str = typer.Option('4h8c', '-c', '--constraint'),
         frameworks: str = typer.Option(None, '-f', '--frameworks'),
         localcode: str = typer.Option(None, '-l', '--localcode')):
    logging.basicConfig(level=logging.DEBUG)
    logger.info("test")
@app.command()
def info_slurm():
    'scontrol show job'

    'srun -p lmb_gpu-rtx2080 -w dagobert --pty bash'
    pass



@app.command()
def rsync(platform_source: str = typer.Option('slurm', '-s', '--source'),
          platform_target: str = typer.Option('local', '-t', '--target'),
          run: str = typer.Option(None, '-r', '--run')):
    logging.basicConfig(level=logging.INFO)

    if run is None:
        logger.warning('Please specify a run.')
        return

    config_source = od3d.io.load_hierarchical_config(platform=platform_source)
    config_target = od3d.io.load_hierarchical_config(platform=platform_target)
    source_link = f'{config_source.platform.link}:' if config_source.platform.link != 'local' else ''
    target_link = f'{config_target.platform.link}:' if config_target.platform.link != 'local' else ''

    path_source = Path(config_source.platform.path_exps).joinpath(run)
    path_target = Path(config_target.platform.path_exps).joinpath(run)
    od3d.io.run_cmd(cmd=f'rsync -avrzP {source_link}{path_source} {target_link}{path_target.parent}', live=True, logger=logger)

@app.command()
def rsync_multiple(platform_source: str = typer.Option('slurm', '-s', '--source'),
                   platform_target: str = typer.Option('local', '-t', '--target'),
                   benchmark: str = typer.Option('co3d_nemo', '-b', '--benchmark'),
                   ablation: str = typer.Option(None, '-a', '--ablation'),
                   platform: str = typer.Option(None, '-p', '--platform'),
                   age_in_hours_lt: int = typer.Option(24, '-l', '--age-in-hours-lt'),
                   age_in_hours_gt: int = typer.Option(0, '-g', '--age-in-hours-gt'),
                   duplicates_keep: str = typer.Option('last', '-d', '--duplicates_keep')):
    logging.basicConfig(level=logging.INFO)
    prev_runs_all = get_dataframe_multiple(benchmark=benchmark, ablation=ablation, platform=platform,
                                           duplicates_keep=duplicates_keep, state='(finished|running)',
                                           add_configs_default=True, add_configs_ablation=False,
                                           age_in_hours_lt=age_in_hours_lt, age_in_hours_gt=age_in_hours_gt)
    prev_runs_all_names = prev_runs_all['name'].tolist()

    for run in prev_runs_all_names:
        logger.info(f'rsync run {run}')

        config_source = od3d.io.load_hierarchical_config(platform=platform_source)
        config_target = od3d.io.load_hierarchical_config(platform=platform_target)
        source_link = f'{config_source.platform.link}:' if config_source.platform.link != 'local' else ''
        target_link = f'{config_target.platform.link}:' if config_target.platform.link != 'local' else ''

        path_source = Path(config_source.platform.path_exps).joinpath(run)
        path_target = Path(config_target.platform.path_exps).joinpath(run)
        od3d.io.run_cmd(cmd=f'rsync -avrzP {source_link}{path_source} {target_link}{path_target.parent}', live=True, logger=logger)


@app.command()
def status_slurm():
    logging.basicConfig(level=logging.INFO)
    # 60j = 60 characters
    format = '"%.18i %.9P %.60j %.8u %.8T %.10M %.9l %.6D %R"'
    slurm_result = subprocess.run(f"ssh slurm 'squeue --me --format={format}'", capture_output=True, shell=True)
    slurm_jobs = slurm_result.stdout.decode("utf-8").split("\n")
    for slurm_job in slurm_jobs:
        logger.info(slurm_job)
@app.command()
def status_torque():
    logging.basicConfig(level=logging.INFO)

    torque_result = subprocess.run(f'ssh torque "qstat -a -u $(whoami)"', capture_output=True, shell=True)
    torque_jobs = torque_result.stdout.decode("utf-8").split("\n")
    for torque_job in torque_jobs:
        logger.info(torque_job)
@app.command()
def stop_torque(job: str = typer.Option(None, '-j', '--job')):
    logging.basicConfig(level=logging.INFO)

    jobs = job.split(',')
    for job in jobs:
        torque_result = subprocess.run(f'ssh torque "qdel {job}"', capture_output=True, shell=True)
        for line in torque_result.stdout.decode("utf-8").split("\n"):
            logger.info(line)

@app.command()
def stop_slurm(job: str = typer.Option(None, '-j', '--job')):
    logging.basicConfig(level=logging.INFO)

    jobs = job.split(',')
    for job in jobs:
        if job.startswith('l'):
            slurm_jobs_ids = get_slurm_jobs_ids(int(job[1:]))
            logger.info(f'stop slurm job ids {slurm_jobs_ids}')
        else:
            slurm_jobs_ids = [int(job)]

        for job_id in slurm_jobs_ids:
            slurm_result = subprocess.run(f'ssh slurm "scancel {str(job_id)}"', capture_output=True, shell=True)
            for line in slurm_result.stdout.decode("utf-8").split("\n"):
                logger.info(line)

