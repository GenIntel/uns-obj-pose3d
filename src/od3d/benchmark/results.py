import logging

from sympy import yn

logger = logging.getLogger(__name__)
import torch
from typing import Dict, List, Union
import wandb
import math

from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.metrics import RocCurveDisplay
import numpy as np
from pathlib import Path


class OD3D_Results(Dict[str, Union[torch.Tensor, List]]):
    def __init__(self, device: torch.device='cpu', init_dict: Dict[str, Union[torch.Tensor, List]]=None, logging_dir=None):
        super().__init__()
        self.mean_blocklist = ['label_gt', 'label_pred', 'rot_diff_rad', 'name_unique', 'item_id', 'cam_tform4x4_obj', 'label_names', 'pi6_pr_vs_sim_geo_and_appear', 'noise2d']
        self.log_blocklist = ['name_unique', 'item_id', 'cam_tform4x4_obj', 'noise2d']
        self.device = device
        self.logging_dir = logging_dir

        if init_dict is not None:
            self.__add__(other=init_dict)

    def __add__(self, other: Dict[str, torch.Tensor]):
        for key, val in other.items():
            # logger.info(key)
            if isinstance(val, torch.Tensor):
                if val.dim() == 0:
                    val = val[None,]
                if key not in self.keys():
                    self[key] = val.to(device=self.device) # torch.Tensor(size=val.shape, device=self.device, dtype=val.dtype)
                else:
                    self[key] = torch.cat([self[key], val.to(device=self.device)], dim=0)
            elif isinstance(val, List):
                if key not in self.keys():
                    self[key] = []
                self[key] += val
            else:
                self[key] = val
        return self

    def add_prefix(self, prefix: str):
        res = {}
        for key, val in self.items():
            res[f'{prefix}_{key}'] = val
        return OD3D_Results(init_dict=res)

    def mean(self):
        res = {}
        for key, val in self.items():
            if not any(s in key for s in self.mean_blocklist): #  key not in self.mean_blocklist:
                res[key] = val.mean(dim=0)


        for k in self.keys():
            if 'label_gt' in k:
                prefix = k[:k.find('label_gt')]
                prefix_saved = f'prefix/{prefix}' if len(prefix) > 0 else ''
                if f'{prefix}label_pred' in self.keys():
                    if f'{prefix}label_names' in self.keys():
                        label_names = self[f'{prefix}label_names']
                    else:
                        label_names = [str(i) for i in range(max(set(self[f'{prefix}label_gt'] + self[f'{prefix}label_pred'])) + 1)]
                    res[f'label/{prefix_saved}acc'] = (self[f'{prefix}label_gt'] == self[f'{prefix}label_pred']).to(dtype=float).mean(dim=0)
                    res[f'label/{prefix_saved}confusion'] = wandb.plot.confusion_matrix(probs=None,
                                                                                        y_true=self[f'{prefix}label_gt'].numpy(),
                                                                                        preds=self[f'{prefix}label_pred'].numpy(),
                                                                                        class_names=label_names)

            if 'rot_diff_rad' in k:
                prefix = k[:k.find('rot_diff_rad')]
                prefix_saved = f'prefix/{prefix}' if len(prefix) > 0 else ''
                rot_diff_rad = self[f'{prefix}rot_diff_rad']
                res[f'pose/{prefix_saved}acc_pi6'] = (rot_diff_rad < math.pi / 6.).to(dtype=float).mean()
                res[f'pose/{prefix_saved}acc_pi12'] = (rot_diff_rad < math.pi / 12.).to(dtype=float).mean()
                res[f'pose/{prefix_saved}acc_pi18'] = (rot_diff_rad < math.pi / 18.).to(dtype=float).mean()

                if rot_diff_rad.dim() == 2:
                    res[f'pose/{prefix_saved}acc_pi6_std'] = (rot_diff_rad < math.pi / 6.).to(
                        dtype=float).mean(dim=0).std(dim=-1)
                    res[f'pose/{prefix_saved}acc_pi12_std'] = (rot_diff_rad < math.pi / 12.).to(
                        dtype=float).mean(dim=0).std(dim=-1)
                    res[f'pose/{prefix_saved}acc_pi18_std'] = (rot_diff_rad < math.pi / 18.).to(
                        dtype=float).mean(dim=0).std(dim=-1)
                else:
                    logger.warning(f'Unexpected dimensions of rot_diff_rad {rot_diff_rad.dim()}')
                # if rot_diff_rad.dim() == 3:
                #     res[f'pose/{prefix_saved}acc_pi6_std'] = (rot_diff_rad < math.pi / 6.).to(
                #         dtype=float).mean(dim=-1).std(dim=-1).mean(dim=-1)
                #     res[f'pose/{prefix_saved}acc_pi12_std'] = (rot_diff_rad < math.pi / 12.).to(
                #         dtype=float).mean(dim=-1).std(dim=-1).mean(dim=-1)
                #     res[f'pose/{prefix_saved}acc_pi18_std'] = (rot_diff_rad < math.pi / 18.).to(
                #         dtype=float).mean(dim=-1).std(dim=-1).mean(dim=-1)

                res[f'pose/{prefix_saved}err_median'] = 180 / math.pi * rot_diff_rad.median()
                res[f'pose/{prefix_saved}err_mean'] = 180 / math.pi * rot_diff_rad.mean()

                if f'{prefix}pose_sim_geo' in self.keys() and f'{prefix}pose_sim_appear' in self.keys():
                    res[f'pose/pr/{prefix_saved}pi6_pr_vs_sim_geo_and_appear'] = self.get_pr_3d(
                        ground_truth=(rot_diff_rad.flatten() < math.pi / 6.).detach().cpu().numpy().astype(int),
                        sim_1st_dim=self[f'{prefix}pose_sim_geo'].flatten().detach().cpu().numpy(),
                        sim_2nd_dim=self[f'{prefix}pose_sim_appear'].flatten().detach().cpu().numpy(),
                        title=f"PI/6={res[f'pose/{prefix_saved}acc_pi6']:.2f}")

                if f'{prefix}sim' in self.keys():
                    res[f'pose/pr/{prefix_saved}pi6'] = self.get_pr(ground_truth=(rot_diff_rad.flatten() < math.pi / 6.).detach().cpu().numpy().astype(int), predictions=self[f'{prefix}sim'].flatten().detach().cpu().numpy(), title=f"PI/6={res[f'pose/{prefix_saved}acc_pi6']:.2f}")
                    res[f'pose/pr/{prefix_saved}pi18'] = self.get_pr(ground_truth=(rot_diff_rad.flatten() < math.pi / 18.).detach().cpu().numpy().astype(int), predictions=self[f'{prefix}sim'].flatten().detach().cpu().numpy(), title=f"PI/18={res[f'pose/{prefix_saved}acc_pi18']:.2f}")

        return OD3D_Results(init_dict=res)

    def get_pr_3d(self, ground_truth, sim_1st_dim, sim_2nd_dim, title):
        # Make data.
        X = np.arange(-0.01, 1.01, 0.01)
        Y = np.arange(-0.01, 1.01, 0.01)
        X, Y = np.meshgrid(X, Y)

        mask_xy = (sim_1st_dim[:, None, None] >= X[None,]) * (sim_2nd_dim[:, None, None] >= Y[None,])
        Z_precision = (ground_truth[:, None, None] * mask_xy).sum(axis=0) / (mask_xy.sum(axis=0))
        Z_precision[mask_xy.sum(axis=0) == 0] = 1.

        Z_recall = (ground_truth[:, None, None] * mask_xy).sum(axis=0) / (ground_truth.sum())
        if ground_truth.sum() == 0:
            Z_recall[:, :] = 0.

        Z_f1 = 2 * (Z_recall * Z_precision) / (Z_recall + Z_precision)
        Z_f1[Z_recall + Z_precision == 0] = 0.

        Z_f1_sub = []
        X_sub = []
        Y_sub = []
        text_sub = []
        stepsize=40

        for x_step in range(0, math.ceil(len(X) / stepsize)):
            for y_step in range(0, math.ceil(len(Y) / stepsize)):
                Z_f1_step = Z_f1[(x_step) * stepsize: (x_step+1) * stepsize, (y_step) * stepsize: (y_step+1) * stepsize, ]
                y_amax_step = Z_f1_step.argmax(axis=-1)
                x_amax_step = Z_f1_step[:, y_amax_step].diagonal().argmax(axis=0)
                y_amax_step = y_amax_step[x_amax_step]
                x_amax = (x_step) * stepsize + x_amax_step
                y_amax = (y_step) * stepsize + y_amax_step
                X_sub.append(X[x_amax, y_amax])
                Y_sub.append(Y[x_amax, y_amax])
                text_sub.append(f'Precision={Z_precision[x_amax, y_amax]:.2f}<br>Recall={Z_recall[x_amax, y_amax]:.2f}')
                Z_f1_sub.append(Z_f1_step[x_amax_step, y_amax_step])

        #fig2 = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16], z=[], color="red")

        # colorscale: 'Electric' 'Viridis', 'Blues', 'Greens'
        # showscale: True/False
        import plotly.graph_objects as go
        fig = go.Figure(data=[
            go.Scatter3d(x=X_sub, y=Y_sub, z=Z_f1_sub, mode="markers+text", name="F1", text=text_sub,
                         textposition="top center",),
            go.Surface(z=Z_precision, x=X, y=Y, colorscale='Greens', name='Precision', hoverinfo='skip', opacity=0.5,
                       showscale=False),
            go.Surface(z=Z_recall, x=X, y=Y, colorscale='Blues', name='Recall', hoverinfo='skip', opacity=0.5,
                       showscale=False),
        ])
        #,  x='min. sim. appearance', y='min. sim. geometry'
        fig.update_layout(title=title,
                          #xaxis_title="X Axis Title",
                          #yaxis_title="Y Axis Title",
                          autosize=False,
                          width=500, height=500,
                          margin=dict(l=65, r=50, b=65, t=90),
                          scene=dict(
                              xaxis_title='Min. Sim. Geometry',
                              yaxis_title='Min. Sim. Appearance'),
                          )

        #fig.show()
        return wandb.Plotly(fig)

    def get_pr(self, ground_truth, predictions, title):


        precision, recall, thresholds = metrics.precision_recall_curve(ground_truth, predictions)

        if len(thresholds) == 0:
            return self.get_empty_figure(title=title)

        display = metrics.PrecisionRecallDisplay.from_predictions(
            ground_truth, predictions, name=title,
        )
        display.figure_.set_figwidth(4)
        display.figure_.set_figheight(4)

        thresholdsLength = len(thresholds)
        colorMap = plt.get_cmap('jet', thresholdsLength)
        thresholds_every = int(thresholdsLength // 10) + 1
        for i in range(0, thresholdsLength, thresholds_every):
            if np.isfinite(recall[i]) and np.isfinite(precision[i]):
                display.ax_.plot(recall[i], precision[i], "o", label=f"sim >= {thresholds[i]:.3f}", color=colorMap(i / thresholdsLength))

        display.ax_.set_xlabel('Recall')
        display.ax_.set_ylabel('Precision')

        #display.ax_.set_title(title)
        display.ax_.legend()
        display.ax_.axis("square")
        display.ax_.set_xlim([-0.1, 1.1])
        display.ax_.set_ylim([-0.1, 1.1])
        display.ax_.axis('on')

        #from od3d.cv.visual.show import get_img_from_plot
        #from od3d.cv.io import image_as_wandb_image
        #img = get_img_from_plot(ax=display.ax_, fig=display.figure_, axis_off=False)
        #return image_as_wandb_image(img, caption=title)
        return display.figure_


    def get_empty_figure(self, title):
        plt.ioff()
        # Create a Figure object
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label=title)
        ax.legend()
        ax.axis("square")
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])

        return fig


    def get_roc(self, ground_truth, predictions, title):
        """
            Args:
                ground_truth (np.ndarray): (n_samples,), int {0, 1}
                predictions (np.ndarray): (n_samples,), float
            Returns:
                figure (matplotlib figure)
        """
        fp, tp, thresholds = metrics.roc_curve(ground_truth, predictions, drop_intermediate=False)

        if len(thresholds) == 0:
            return self.get_empty_figure(title=title)

        display = RocCurveDisplay.from_predictions(
            ground_truth,
            predictions,
            name=title,
            drop_intermediate=False,
        )
        display.figure_.set_figwidth(4)
        display.figure_.set_figheight(4)


        thresholdsLength = len(thresholds)
        colorMap = plt.get_cmap('jet', thresholdsLength)
        thresholds_every = int(thresholdsLength // 10) + 1
        for i in range(0, thresholdsLength, thresholds_every):
            if np.isfinite(tp[i]) and np.isfinite(fp[i]):
                display.ax_.plot(fp[i], tp[i], "o", label=f"sim >= {thresholds[i]:.2f}",
                                 color=colorMap(i / thresholdsLength))

        display.ax_.set_xlabel('False Positive Rate')
        display.ax_.set_ylabel('True Positive Rate')

        #display.ax_.set_title(title)
        display.ax_.legend()
        display.ax_.axis("square")
        display.ax_.set_xlim([-0.1, 1.1])
        display.ax_.set_ylim([-0.1, 1.1])
        display.ax_.axis('on')


        #from od3d.cv.visual.show import get_img_from_plot
        #from od3d.cv.io import image_as_wandb_image
        #img = get_img_from_plot(ax=display.ax_, fig=display.figure_, axis_off=False)
        #return image_as_wandb_image(img, caption=title)
        return display.figure_

    # def get_roc(self, ground_truth, predictions, title): #labels, predictions, positive_label, thresholds_every=10, title=''):
    #
    #     # fp: false positive rates. tp: true positive rates
    #     fp, tp, thresholds = metrics.roc_curve(ground_truth, predictions, drop_intermediate=False)
    #     roc_auc = metrics.auc(fp, tp)
    #
    #     plt.ioff()
    #     # Create a Figure object
    #     fig, ax = plt.subplots(figsize=(4, 4))
    #     # fig = plt.figure(figsize=(16, 16))
    #
    #     # Add a subplot (1 row, 1 column, first subplot)
    #     #ax = fig.add_subplot(1, 1, 1)
    #
    #     ax.axis("square")
    #     ax.plot(fp, tp, label='ROC curve (area = %0.2f)' % roc_auc, linewidth=2, color='darkorange')
    #     ax.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=2)
    #     ax.set_xlabel('False positives rate')
    #     ax.set_ylabel('True positives rate')
    #     ax.set_xlim([-0.03, 1.0])
    #     ax.set_ylim([0.0, 1.03])
    #     ax.set_title(title)
    #     ax.legend()
    #     # ax.legend(loc="lower right")
    #     # ax.grid(True)
    #
    #     # plot some thresholds
    #     thresholdsLength = len(thresholds)
    #     colorMap = plt.get_cmap('jet', thresholdsLength)
    #     thresholds_every = int(thresholdsLength // 10) + 1
    #     for i in range(0, thresholdsLength, thresholds_every):
    #         if np.isfinite(fp[i]) and np.isfinite(tp[i]):
    #             threshold_value_with_max_four_decimals = str(thresholds[i])[:5]
    #             ax.text(fp[i] - 0.03, tp[i] + 0.005, threshold_value_with_max_four_decimals, fontdict={'size': 10},
    #                      color=colorMap(i / thresholdsLength))
    #
    #     from od3d.cv.visual.show import get_img_from_plot
    #     from od3d.cv.io import image_as_wandb_image
    #     img = get_img_from_plot(ax=ax, fig=fig, axis_off=False)
    #     plt.close(fig)
    #     return image_as_wandb_image(img, caption=title)

    #
    # def get_roc(self, ground_truth, predictions, title):
    #     """
    #         Args:
    #             ground_truth (np.ndarray): (n_samples,), int {0, 1}
    #             predictions (np.ndarray): (n_samples,), float
    #         Returns:
    #             figure (matplotlib figure)
    #     """
    #     display = RocCurveDisplay.from_predictions(
    #         ground_truth,
    #         predictions,
    #         name=title,
    #         drop_intermediate=False,
    #     )
    #     display.ax_.axis("square")
    #     display.ax_.set_xlabel("False Positive Rate")
    #     display.ax_.set_ylabel("True Positive Rate")
    #     display.ax_.set_title(title)
    #     display.ax_.legend()
    #     return display.figure_
    #
    #
    #     # table = wandb.Table(data=data, columns=["false positive rate", "true positive rate"])
    #     # return wandb.plot.line_series(
    #     #     xs=[0, 1, 2, 3, 4],
    #     #     ys=[[10, 20, 30, 40, 50], [0.5, 11, 72, 3, 41]],
    #     #     keys=["metric Y", "metric Z"],
    #     #     title=title,
    #     #     xname="false positive rate",
    #     #     yn="true positive rate")
    #
    #     # return wandb.plot.line(table, "false positive rate", "true positive rate", title=title)


    def get_filtered_log_results(self):
        res = {}
        for key, val in self.items():
            if not any(s in key for s in self.log_blocklist): #  key not in self.log_blocklist: # any(substring in s for s in string_list)
                res[key] = val
        return res

    def get_log_results(self):
        res = {}
        for key, val in self.items():
            # if isinstance(val, torch.Tensor):
            #     res[key] = val.detach().cpu().tolist()
            # else:
            res[key] = val
        return res

    def log(self):
        filtered_log_results = self.get_filtered_log_results()
        wandb.log(filtered_log_results)

    def log_with_prefix(self, prefix: str, prefix_append_char='/'):
        filtered_log_results = self.get_filtered_log_results()
        filtered_log_results_with_prefix = {prefix + f'{prefix_append_char}' + k: v for k, v in filtered_log_results.items()}
        wandb.log(filtered_log_results_with_prefix)


    def save_visual(self, prefix: str):
        for key, val in self.items():
            if isinstance(val, wandb.data_types.Image):
                fpath = self.logging_dir.joinpath(f'{prefix}/{key}.png')
                fpath.parent.mkdir(parents=True, exist_ok=True)
                val.image.save(fpath)

        #fpath.parent.mkdir(parents=True, exist_ok=True)
        #torch.save(obj=self.get_log_results(), f=fpath)

    def save_with_dataset(self, prefix: str, dataset, _dict: dict=None):
        if _dict is None:
            _dict = self.get_log_results()
        fpath = self.logging_dir.joinpath(f'{prefix}/{dataset.name}/results.pt')
        fpath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj=_dict, f=fpath)
        dataset.save_to_config(fpath=self.logging_dir.joinpath(f'{prefix}/{dataset.name}/config.yaml'))

    @classmethod
    def read_from_local(cls, logging_dir: Path, dataset_rpath: Path):
        fpath = logging_dir.joinpath(f'{dataset_rpath}/results.pt')
        _dict = torch.load(fpath)
        return cls(logging_dir=logging_dir, init_dict=_dict)
