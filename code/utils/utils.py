from pathlib import Path
from abc import ABC, abstractclassmethod
import torch
import torchvision.transforms as T
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from torchmetrics.classification.accuracy import Accuracy

from pytorch_lightning import LightningDataModule, seed_everything, Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.callbacks import LearningRateMonitor, BatchSizeFinder, DeviceStatsMonitor
from typing import Any, Dict, List, Optional, Type
import shutil

from matplotlib import pyplot as plt
plt.style.use('tableau-colorblind10')
import pandas as pd
import json
import pprint
import seaborn as sns

import numpy as np

import torchmetrics
from torchmetrics import PrecisionRecallCurve, ROC
from torchmetrics.functional.classification import binary_roc, binary_auroc, multiclass_auroc, binary_precision_recall_curve, multiclass_precision_recall_curve, confusion_matrix
from torchmetrics.utilities.compute import _auc_compute_without_check, _auc_compute

LEGEND_SIZE = 50
AXIS_SIZE = 40


LABEL_MAP = {
    # 'bin': {'0': 0, '1': 1, '2': 1, '3': 1, '4': 1, '5': None},
    # 'tcmr_viral': {'0': None, '1': 0, '2': None, '3': None, '4': 1, '5': None},
    # 'no_viral': {'0': 0, '1': 1, '2': 2, '3': 3, '4': None, '5': None},
    'no_other': {'0': 'Normal', '1': 'TCMR', '2': 'ABMR', '3': 'Mixed', '4': 'Viral'},
    # 'no_stable': {'0': None, '1': 1, '2': 2, '3': 3, '4': None, '5': None},
    # 'all': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5},
    'rejections': {'0': 'TCMR', '1': 'ABMR', '2': 'Mixed'},
    'norm_rest': {'0': 'Normal', '1': 'Disease'},
    'rej_rest': {'0': 'Rejection', '1': 'Other'},
    'rest_rej': {'0': 'Other', '1': 'Rejection'},
    'norm_rej_rest': {'0': 'Normal', '1': 'Rejection', '2': 'Other'},
    'big_three': {'0': 'ccRCC', '1': 'papRCC', '2': 'chRCC'},
    'tcmr_abmr': {'0': 'TCMR', '1': 'ABMR'},
    'tcmr': {'0': 'Other', '1': 'TCMR'},

}
COLOR_MAP = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
COLOR_MAP = ['#377eb8', '#ff7f00', '#f781bf']


#---->read yaml
import yaml
from addict import Dict
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)

#---->load Loggers
from pytorch_lightning import loggers as pl_loggers

def load_loggers(cfg):

    # log_path = cfg.General.log_path
    # Path(log_path).mkdir(exist_ok=True, parents=True)
    # log_name = str(Path(cfg.config).parent) + f'_{cfg.Model.backbone}' + f'_{cfg.Loss.base_loss}'
    # version_name = Path(cfg.config).name[:-5]
    
    
    #---->TensorBoard
    if cfg.General.server == 'train':
        # print(cfg.log_path)
        if cfg.fine_tune:
            cfg.log_path = Path(cfg.log_path)
            print('cfg.log_path: ', cfg.log_path)

            tb_logger = pl_loggers.TensorBoardLogger(cfg.log_path,
                                                    version = cfg.version,
                                                    sub_dir = f'ft_epoch_{cfg.epoch}',
                                                    log_graph = True, default_hp_metric = False)
        
        else:
            tb_logger = pl_loggers.TensorBoardLogger(cfg.log_path,
                                                    # version = f'fold{cfg.Data.fold}'
                                                    log_graph = True, default_hp_metric = False)
        # print(tb_logger.version)
        version = tb_logger.version
        #---->CSV
        csv_logger = pl_loggers.CSVLogger(cfg.log_path, version = version
                                        ) # version = f'fold{cfg.Data.fold}', 
        # print(csv_logger.version)
        # wandb_logger = pl_loggers.WandbLogger(project=f'{cfg.Model.name}_{cfg.task}', name=f'{cfg.log_name}', save_dir=cfg.log_path)
        return [tb_logger, csv_logger]
    else:  
        if cfg.from_finetune:
            prefix = 'test_ft_epoch'
        else:
            prefix = 'test_epoch'
        cfg.log_path = Path(cfg.log_path)
        print('cfg.log_path: ', cfg.log_path)

        tb_logger = pl_loggers.TensorBoardLogger(cfg.log_path,
                                                version = cfg.version,
                                                sub_dir = f'{prefix}_{cfg.epoch}',
                                                log_graph = True, default_hp_metric = False)
        #---->CSV
        # for some reason this creates the save path.
        version = tb_logger.version
        csv_logger_path = Path(cfg.log_path) / 'lightning_logs' / f'version_{cfg.version}' / f'test_epoch_{cfg.epoch}'
        csv_logger = pl_loggers.CSVLogger(csv_logger_path,
                                        version = cfg.version)
        # wandb_logger = pl_loggers.WandbLogger(project=f'{cfg.task}_{cfg.log_name}')                      
    
        # print(f'---->Log dir: {cfg.log_path}')

    # return tb_logger
        # return [tb_logger]
        return [tb_logger, csv_logger]
    # return wandb_logger
    # return [tb_logger, csv_logger, wandb_logger]


#---->load Callback
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, StochasticWeightAveraging
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def load_callbacks(cfg, save_path):

    Mycallbacks = []
    # Make output path
    output_path = save_path / 'checkpoints' 
    output_path.mkdir(exist_ok=True, parents=True)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=cfg.General.patience,
        verbose=True,
        mode='min'
    )

    Mycallbacks.append(early_stop_callback)
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description='green_yellow',
            progress_bar='green1',
            progress_bar_finished='green1',
            batch_progress='green_yellow',
            time='grey82',
            processing_speed='grey82',
            metrics='grey82'
        )
    )
    Mycallbacks.append(progress_bar)

    if cfg.General.server == 'train':
        # save_path = Path(cfg.log_path) / 'lightning_logs' / f'version_{cfg.resume_version}' / last.ckpt
        if cfg.fine_tune:
            Mycallbacks.append(ModelCheckpoint(monitor = 'val_loss',
                                            dirpath = str(output_path),
                                            
                                            filename = 'ft_{epoch:02d}-{val_loss:.4f}-{val_auc: .4f}-{val_patient_auc:.4f}',
                                            verbose = True,
                                            save_last = True,
                                            save_top_k = 1,
                                            mode = 'min',
                                            save_weights_only = True))
            # Mycallbacks.append(ModelCheckpoint(monitor = 'val_auc',
            #                                 dirpath = str(output_path),
            #                                 filename = 'ft_{epoch:02d}-{val_loss:.4f}-{val_auc:.4f}-{val_patient_auc: .4f}',
            #                                 verbose = True,
            #                                 save_last = True,
            #                                 save_top_k = 1,
            #                                 mode = 'max',
            #                                 save_weights_only = True))
            # Mycallbacks.append(ModelCheckpoint(monitor = 'val_patient_auc',
            #                                 dirpath = str(output_path),
            #                                 filename = 'ft_{epoch:02d}-{val_loss:.4f}-{val_auc:.4f}-{val_patient_auc:.4f}',
            #                                 verbose = True,
            #                                 save_last = True,
            #                                 save_top_k = 1,
            #                                 mode = 'max',
            #                                 save_weights_only = True))
        else:
            Mycallbacks.append(ModelCheckpoint(monitor = 'val_loss',
                                            dirpath = str(output_path),
                                            filename = '{epoch:02d}-{val_loss:.4f}-{val_accuracy:.2f}-{val_auc: .2f}-{val_patient_auc: .2f}',
                                            verbose = True,
                                            save_last = True,
                                            save_top_k = 3,
                                            mode = 'min',
                                            save_weights_only = True))
            Mycallbacks.append(ModelCheckpoint(monitor = 'val_auc',
                                            dirpath = str(output_path),
                                            filename = '{epoch:02d}-{val_loss:.4f}-{val_accuracy:.2f}-{val_auc: .2f}-{val_patient_auc: .2f}',
                                            verbose = True,
                                            save_last = True,
                                            save_top_k = 1,
                                            mode = 'max',
                                            save_weights_only = True))
            Mycallbacks.append(ModelCheckpoint(monitor = 'val_accuracy',
                                            dirpath = str(output_path),
                                            filename = '{epoch:02d}-{val_loss:.4f}-{val_accuracy:.2f}-{val_auc: .2f}-{val_patient_auc: .2f}',
                                            verbose = True,
                                            save_last = True,
                                            save_top_k = 3,
                                            mode = 'max',
                                            save_weights_only = True))
            # Mycallbacks.append(ModelCheckpoint(monitor = 'val_patient_auc',
            #                                 dirpath = str(output_path),
            #                                 filename = '{epoch:02d}-{val_loss:.4f}-{val_auc:.4f}-{val_patient_auc:.4f}',
            #                                 verbose = True,
            #                                 save_last = True,
            #                                 save_top_k = 3,
            #                                 mode = 'max',
            #                                 save_weights_only = True))

    swa = StochasticWeightAveraging(swa_lrs=1e-2)
    Mycallbacks.append(swa)

    # device_stats = DeviceStatsMonitor(cpu_stats=True)
    # Mycallbacks.append(device_stats)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    Mycallbacks.append(lr_monitor)

    return Mycallbacks

#---->val loss
import torch
import torch.nn.functional as F
def cross_entropy_torch(x, y):
    x_softmax = [F.softmax(x[i], dim=0) for i in range(len(x))]
    x_log = torch.tensor([torch.log(x_softmax[i][y[i]]) for i in range(len(y))])
    # x_log = torch.tensor([torch.log(x_softmax[i][y[i]]) for i in range(y.shape[0])])
    loss = - torch.sum(x_log) / y.shape[0]
    return loss

#-----> convert labels for task


def convert_labels_for_task(task, label):

    return LABEL_MAP[task][label]


def get_optimal_operating_point(probs, target):
# def get_optimal_operating_point(fpr, tpr, thresholds):
    '''
    Returns: 
        optimal_fpr [Tensor]
        optimal_tpr [Tensor]
        optimal_threshold [Float]
    '''

    fpr, tpr, thresholds = binary_roc(probs, target)

    youden_j = tpr - fpr
    optimal_idx = torch.argmax(youden_j)
    # print(youden_j[optimal_idx])
    optimal_threshold = thresholds[optimal_idx].item()
    # print(optimal_threshold)
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]

    return optimal_fpr, optimal_tpr, optimal_threshold


def get_roc_curve_2(probs, target, mean_auroc, task, ax=None, target_label=0, target_class=''):

    

    if type(probs) is np.ndarray:
        probs = torch.from_numpy(probs)
    if type(target) is np.ndarray:
        target = torch.from_numpy(target)
    
    task_label_map = LABEL_MAP[task]
    n_classes = len([v for v in task_label_map.values() if v != None])
    print(n_classes)
    
    if n_classes <= 2:
        auroc_score = binary_auroc(probs, target)
        ROC = torchmetrics.ROC(task='binary')
    else:
        # n_classes = 3
        # print(n_classes)
        # print(probs)
        # print(target)
        auroc_score = multiclass_auroc(probs, target, num_classes=n_classes, average=None)
        ROC = torchmetrics.ROC(task='multiclass', num_classes=n_classes)

    fpr_list, tpr_list, thresholds = ROC(probs, target)

    plots = []
    if n_classes > 2:
        # print(target_label)
        color = COLOR_MAP[target_label]
        
        fpr = fpr_list[target_label].cpu().numpy()
        tpr = tpr_list[target_label].cpu().numpy()
        df = pd.DataFrame(data = {'fpr': fpr, 'tpr': tpr})
        # print(df)
        # color = COLOR_MAP[0]
        line_plot = sns.lineplot(data=df, x='fpr', y='tpr', label=f'{auroc_score[target_label]:.3f}', legend='full', color=color, linewidth=5, errorbar=('ci', 95), ax=ax)
        line_plot.set(xlabel=None)
        line_plot.set(ylabel=None)
        ax.legend(title=f'MEAN: {mean_auroc:.3f}', title_fontsize=50, loc='lower right', fontsize=50, frameon=False, borderpad=0, alignment='right')
        # print(ax.get_ymajorticklabels())
        

    else: 
        color = COLOR_MAP[target_label]
        
        optimal_fpr, optimal_tpr, optimal_threshold = get_optimal_operating_point(probs, target)
        fpr = fpr_list.cpu().numpy()
        tpr = tpr_list.cpu().numpy()
        optimal_fpr = optimal_fpr.cpu().numpy()
        optimal_tpr = optimal_tpr.cpu().numpy()

        df = pd.DataFrame(data = {'fpr': fpr, 'tpr': tpr})
        line_plot = sns.lineplot(data=df, x='fpr', y='tpr', label=f'{auroc_score:.3f}', legend='full', color=color, linewidth=5, errorbar=('ci', 95), ax = ax) #AUROC
        line_plot.set(xlabel=None)
        line_plot.set(ylabel=None)
        ax.legend(loc='lower right', fontsize=LEGEND_SIZE, frameon=False, handlelength=0)
        
    ax.plot([0,1], [0,1], linestyle='--', color='red')
    ax.set_xlim([0.0,1.0])
    ax.set_ylim([0.0,1.0])
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.5, 1.0])
    # ax.set_xlabel('FPR (1-specificity)', fontsize=50, labelpad=20) #False positive rate (1-specificity)
    # ax.set_ylabel('TPR (sensitivity)', fontsize=50, labelpad=20)
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)
    # print(ax.get_ymajorticklabels())
    # print(ax.get_ymajorticklabels()[0])
    # ax.xaxis.set_ticks(np.arange(0.0, 1.0, 0.2))
    # ax.yaxis.set_ticks(np.arange(0.0, 1.0, 0.2))
    # ax.get_ymajorticklabels()[0].set_visible(False)
    # ax.legend(loc='lower right', fontsize=LEGEND_SIZE, frameon=False, handlelength=0)

def get_pr_curve_2(probs, target, mean_precision, task, ax=None, target_label=0, target_class=''):

    if type(probs) is np.ndarray:
        probs = torch.from_numpy(probs)
    if type(target) is np.ndarray:
        target = torch.from_numpy(target)
    
    task_label_map = LABEL_MAP[task]
    n_classes = len([v for v in task_label_map.values() if v != None])
    
    if n_classes <= 2:
        precision, recall, thresholds = binary_precision_recall_curve(probs, target)
    else:
        # auroc_score = multiclass_auroc(probs, target, num_classes=n_classes, average=None)
        precision, recall, thresholds = multiclass_precision_recall_curve(probs, target, num_classes=n_classes)
        

    # fpr_list, tpr_list, thresholds = ROC(probs, target)

    plots = []
    if n_classes > 2:
        color = COLOR_MAP[target_label]
        
        re = recall[target_label]
        pr = precision[target_label]
        # no_skill = len(target[target==target_label]) / len(target)
        
        partial_auc = _auc_compute(re, pr, 1.0)
        df = pd.DataFrame(data = {'re': re.cpu().numpy(), 'pr': pr.cpu().numpy()})
        line_plot = sns.lineplot(data=df, x='re', y='pr', label=f'{partial_auc:.3f}', legend='full', errorbar=('ci', 95), color=color, linewidth=5, ax = ax)
        line_plot.set(xlabel=None)
        line_plot.set(ylabel=None)
    else: 
        color = COLOR_MAP[0]
        # precision, recall, thresholds = binary_precision_recall_curve(probs, target)
        baseline = len(target[target==1]) / len(target)
        
        pr = precision
        re = recall
        partial_auc = _auc_compute(re, pr, 1.0) # - baseline
        # ax.plot(re, pr)
        df = pd.DataFrame(data = {'re': re.cpu().numpy(), 'pr': pr.cpu().numpy()})
        line_plot = sns.lineplot(data=df, x='re', y='pr', label=f'{partial_auc:.3f}', legend='full', color=color, linewidth=5, ax = ax)
    
    # ax.plot([0,1], [mean_precision, mean_precision], linestyle='--', color='red') #label=f'Baseline={baseline:.3f}', 
    # ax.annotate(f'Average Precision={mean_precision:.2f}', (0.5, mean_precision), textcoords='offset points', xytext=(0,-30), ha='right', fontsize=30)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    # ax.set_xlabel('Recall', fontsize=AXIS_SIZE, labelpad=20)
    # ax.set_ylabel('Precision', fontsize=AXIS_SIZE, labelpad=20)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.5, 1.0])
    ax.tick_params(axis='x', labelsize=50)
    ax.tick_params(axis='y', labelsize=50)
    # ax.get_ymajorticklabels()[0].set_visible(False)
    ax.legend(loc='lower left', fontsize=50, frameon=False)

def get_roc_curve(probs, target, task, model, ax=None, separate=True):

    if type(probs) is np.ndarray:
        probs = torch.from_numpy(probs)
    if type(target) is np.ndarray:
        target = torch.from_numpy(target)
        
    task_label_map = LABEL_MAP[task]
    
    if len(probs.shape) == 1:
        n_classes = 2
        ROC = torchmetrics.ROC(task='binary')

    else:
        n_classes = 3
        ROC = torchmetrics.ROC(task='multiclass', num_classes=n_classes)
        
    fpr_list, tpr_list, thresholds = ROC(probs, target)

    plots = []
    if n_classes > 2:
        auroc_score = multiclass_auroc(probs, target, num_classes=n_classes, average=None)
        for i in range(len(fpr_list)):
            fig, ax = plt.subplots(figsize=(10,10))
            # fig = plt.figure(figsize=(6,6))

            class_label = task_label_map[str(i)]
            # color = COLOR_MAP[0]
            color = COLOR_MAP[i]
            
            fpr = fpr_list[i].cpu().numpy()
            tpr = tpr_list[i].cpu().numpy()
            # ax.plot(fpr, tpr, label=f'class_{i}, AUROC={auroc_score[i]}')
            df = pd.DataFrame(data = {'fpr': fpr, 'tpr': tpr})
            # line_plot = sns.lineplot(data=df, x='fpr', y='tpr', label=f'{auroc_score[i]:.3f}', legend='full', color=color, linewidth=5)

            ### temporary!!!
            if separate:
                color = COLOR_MAP[0]
                line_plot = sns.lineplot(data=df, x='fpr', y='tpr', label=f'{auroc_score[i]:.3f}', legend='full', color=color, linewidth=5, )
                add_on = i
                # output_dir = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/results/{model}/'
                output_dir = f'/homeStor1/ylan/DeepGraft_project/DeepGraft_Draft/figures/{model}'

                ax.plot([0,1], [0,1], linestyle='--', color='red')
                ax.set_xlim([0,1])
                ax.set_ylim([0,1])
                ax.set_xlabel('', fontsize=18)
                # 
                ax.set_ylabel('True positive rate (sensitivity)', fontsize=AXIS_SIZE, labelpad=20)

                # if i == 2:
                ax.set_xlabel('False positive rate (1-specificity)', fontsize=AXIS_SIZE, labelpad=20)
                # else:
                    # ax.set_xlabel('', fontsize=AXIS_SIZE)
                ax.tick_params(axis='x', labelsize=30)
                ax.tick_params(axis='y', labelsize=30)
                # ax.set_yticklabels(fontsize=15)
                # ax.set_title('ROC curve')
                ax.legend(loc='lower right', fontsize=LEGEND_SIZE, frameon=False, handlelength=0)
                ax.get_ymajorticklabels()[0].set_visible(False)
                line_plot.figure.savefig(f'{output_dir}/{model}_{task}_{add_on}_roc.png', dpi=400)
                line_plot.figure.savefig(f'{output_dir}/{model}_{task}_{add_on}_roc.svg', format='svg')
            #     # plt.show()

                line_plot.figure.clf()
            #     plots.append(fig)
            # else:

    else: 
        # fig, ax = plt.subplots(figsize=(10,10))
        auroc_score = binary_auroc(probs, target)
        color = COLOR_MAP[0]
        
        optimal_fpr, optimal_tpr, optimal_threshold = get_optimal_operating_point(probs, target)
        fpr = fpr_list.cpu().numpy()
        tpr = tpr_list.cpu().numpy()
        optimal_fpr = optimal_fpr.cpu().numpy()
        optimal_tpr = optimal_tpr.cpu().numpy()

        df = pd.DataFrame(data = {'fpr': fpr, 'tpr': tpr})
        line_plot = sns.lineplot(data=df, x='fpr', y='tpr', label=f'{auroc_score:.3f}', legend='full', color=color, linewidth=5, errorbar=('ci', 95), ax = ax) #AUROC
        # ax.plot([0, 1], [optimal_tpr, optimal_tpr], linestyle='--', color='black', label=f'OOP={optimal_threshold:.3f}')
        # ax.plot([optimal_fpr, optimal_fpr], [0, 1], linestyle='--', color='black')

    # for fig in plots:
        # ax = fig.add_subplot(111)
    ax.plot([0,1], [0,1], linestyle='--', color='red')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    # ax.set_xlabel('', fontsize=18)
    ax.set_xlabel('False positive rate (1-specificity)', fontsize=AXIS_SIZE, labelpad=20)
    ax.set_ylabel('True positive rate (sensitivity)', fontsize=AXIS_SIZE, labelpad=20)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    # ax.set_yticklabels(fontsize=15)
    # ax.set_title('ROC curve')
    ax.get_ymajorticklabels()[0].set_visible(False)
    ax.legend(loc='lower right', fontsize=LEGEND_SIZE, frameon=False, handlelength=0)

    # return plots
    # return line_plot

def get_pr_curve(probs, target, task, model, ax=None, target_label=1):

    if type(probs) is np.ndarray:
        probs = torch.from_numpy(probs)
    if type(target) is np.ndarray:
        target = torch.from_numpy(target)

    if task == 'norm_rest' or task == 'rej_rest' or task == 'rest_rej':
        n_classes = 2 
        # PRC = torchmetrics.PrecisionRecallCurve(task='binary')
        # ROC = torchmetrics.ROC(task='binary')
    else: 
        n_classes = 3
        # PRC = torchmetrics.PrecisionRecallCurve(task='multiclass', num_classes = n_classes)
        # ROC = torchmetrics.ROC(task='multiclass', num_classes=n_classes)
    
    
    # fig, ax = plt.subplots(figsize=(10,10))


    
    if n_classes > 2:

        precision, recall, thresholds = multiclass_precision_recall_curve(probs, target, num_classes=n_classes)
        task_label_map = LABEL_MAP[task]
        
        for i in range(len(precision)):

            fig, ax = plt.subplots(figsize=(10,10))

            class_label = task_label_map[str(i)]
            color = COLOR_MAP[0]

            re = recall[i]
            pr = precision[i]
            
            baseline = len(target[target==i]) / len(target)
            partial_auc = _auc_compute(re, pr, 1.0) - baseline
            df = pd.DataFrame(data = {'re': re.cpu().numpy(), 'pr': pr.cpu().numpy()})
            line_plot = sns.lineplot(data=df, x='re', y='pr', label=f'{partial_auc:.3f}', legend='full', color=color, linewidth=5, ax = ax)

            # baseline = len(target[target==i]) / len(target)
            print(baseline)
            ax.plot([0,1],[baseline, baseline], linestyle='--', color=color)

            add_on = i
            # output_dir = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/results/{model}/'
            output_dir = f'/homeStor1/ylan/DeepGraft_project/DeepGraft_Draft/figures/{model}'
            
            # ax.plot([0,1], [0,1], linestyle='--', color='red')
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            # 
            ax.set_xlabel('Precision', fontsize=AXIS_SIZE, labelpad=20)
            ax.set_ylabel('Recall', fontsize=AXIS_SIZE, labelpad=20)
            ax.tick_params(axis='x', labelsize=30)
            ax.tick_params(axis='y', labelsize=30)
            ax.get_ymajorticklabels()[0].set_visible(False)
            # ax.set_yticklabels(fontsize=15)
            # ax.set_title('ROC curve')
            ax.legend(loc='lower right', fontsize=LEGEND_SIZE, frameon=False, handlelength=0)
            

            line_plot.figure.savefig(f'{output_dir}/{model}_{task}_{add_on}_pr.png', dpi=400)
            line_plot.figure.savefig(f'{output_dir}/{model}_{task}_{add_on}_pr.svg', format='svg')
            # plt.show()

            line_plot.figure.clf()

    else: 
        # print(fpr_list)
        
        color = COLOR_MAP[0]
        precision, recall, thresholds = binary_precision_recall_curve(probs, target)
        baseline = len(target[target==target_label]) / len(target)
        
        pr = precision
        re = recall
        partial_auc = _auc_compute(re, pr, 1.0) - baseline
        # ax.plot(re, pr)
        df = pd.DataFrame(data = {'re': re.cpu().numpy(), 'pr': pr.cpu().numpy()})
        line_plot = sns.lineplot(data=df, x='re', y='pr', label=f'{partial_auc:.3f}', legend='full', color=color, ax = ax)
    
        ax.plot([0,1], [baseline, baseline], linestyle='--', color=color) #label=f'Baseline={baseline:.3f}', 

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('Recall', fontsize=AXIS_SIZE, labelpad=20)
    ax.set_ylabel('Precision', fontsize=AXIS_SIZE, labelpad=20)
    # ax.yticks(fontsize=25)
    # ax.set_title('PR curve')
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.get_ymajorticklabels()[0].set_visible(False)
    ax.legend(loc='lower right', fontsize=LEGEND_SIZE, frameon=False, handlelength=0)

    # return line_plot

def get_confusion_matrix(probs, target, task, optimal_threshold, ax=None, comment='patient', stage='test'): # threshold

    if type(probs) is np.ndarray:
        probs = torch.from_numpy(probs)
    if type(target) is np.ndarray:
        target = torch.from_numpy(target)

    task_label_map  = LABEL_MAP[task]
    n_classes = len([v for v in task_label_map.values() if v != None])

    if n_classes <= 2:
        ROC = torchmetrics.ROC(task='binary')
        confmat = confusion_matrix(probs, target, task='binary', threshold=optimal_threshold, num_classes=n_classes)
    # if task == 'norm_rest' or task == 'rej_rest' or task == 'rest_rej':
        # n_classes = 2 
        # ROC = torchmetrics.ROC(task='binary')
        # confmat = confusion_matrix(probs, target, task='binary', threshold=optimal_threshold, num_classes=n_classes)
    else: 
        ROC = torchmetrics.ROC(task='multiclass', num_classes=n_classes)
        confmat = confusion_matrix(probs, target, task='multiclass', num_classes=n_classes, threshold=optimal_threshold)



    # preds = torch.argmax(probs, dim=1)
    # if self.n_classes <= 2:
    #     probs = probs[:,1] 

    # read threshold file
    # threshold_csv_path = f'{self.loggers[0].log_dir}/val_thresholds.csv'
    # if not Path(threshold_csv_path).is_file():
    #     # thresh_dict = {'index': ['train', 'val'], 'columns': , 'data': [[0.5, 0.5], [0.5, 0.5]]}
    #     thresh_df = pd.DataFrame({'slide': [0.5], 'patient': [0.5]})
    #     thresh_df.to_csv(threshold_csv_path, index=False)
    # else:  
    # thresh_df = pd.read_csv(threshold_csv_path, index_col=False)
    # optimal_threshold = thresh_df['patient'].values[0]
    # print(optimal_threshold)
    # if stage != 'test':
    #     if n_classes <= 2:
    #         fpr_list, tpr_list, thresholds = ROC(probs, target)
    #         optimal_fpr, optimal_tpr, optimal_threshold = get_optimal_operating_point(fpr_list, tpr_list, thresholds)
    #         # print(f'Optimal Threshold {stage} {comment}: ', optimal_threshold)
    #         thresh_df.at[0, comment] =  optimal_threshold
    #         thresh_df.to_csv(threshold_csv_path, index=False)
    #     else: 
    #         optimal_threshold = 0.5
    # elif stage == 'test': 
    # if n_classes > 2:
    #     optimal_threshold=1/n_classes
    # if n_classes == 2:    
    #     optimal_fpr, optimal_tpr, optimal_threshold = get_optimal_operating_point(probs, target)

        # fpr_list, tpr_list, thresholds = ROC(probs, target)
        # optimal_fpr, optimal_tpr, optimal_threshold = get_optimal_operating_point(fpr_list, tpr_list, thresholds)
    # else:
    #     optimal_threshold = 0.5
    # optimal_threshold = thresh_df.at[0, comment]

    # print(f'Optimal Threshold {stage} {comment}: ', optimal_threshold)
        # optimal_threshold = 0.5 # manually change to val_optimal_threshold for testing

    # print(confmat)
    # confmat = self.confusion_matrix(preds, target, threshold=optimal_threshold)
    # if n_classes == 2:
    #     confmat = confusion_matrix(probs, target, task='binary', threshold=optimal_threshold, num_classes=n_classes)
    # elif n_classes > 2: 
    #     confmat = confusion_matrix(probs, target, task='multiclass', num_classes=n_classes, threshold=optimal_threshold)

    cm_labels = LABEL_MAP[task].values()

    # fig, ax = plt.subplots()
    # figsize=plt.rcParams.get('figure.figsize')
    # plt.figure(figsize=(10, 10))

    # df_cm = pd.DataFrame(confmat.cpu().numpy(), index=range(self.n_classes), columns=range(self.n_classes))
    df_cm = pd.DataFrame(confmat.cpu().numpy(), index=cm_labels, columns=cm_labels)
    print(df_cm)
    # fig_ = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Spectral').get_figure()
    # sns.set(font_scale=1.5)
    cm_plot = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'fontsize': 25, 'multialignment':'center'}, ax=ax) #
    cm_plot.set_xticklabels(cm_plot.get_xmajorticklabels(), fontsize = 30)
    cm_plot.set_yticklabels(cm_plot.get_ymajorticklabels(), fontsize = 30)
    # cm_plot.xaxis.tick_top()
    # cm_plot.set_yticklabels(fontsize=30)
    # sns.set(font_scale=1.3)

    plt.yticks(va='center')
    plt.ylabel('True', fontsize=AXIS_SIZE, labelpad=20)
    plt.xlabel('Prediction', fontsize=AXIS_SIZE, labelpad=20)


    
    
    
    # cm_plot = 
    # if stage == 'train':
    #     self.loggers[0].experiment.add_figure(f'{stage}/Confusion matrix', cm_plot.figure, self.current_epoch)
    #     if len(self.loggers) > 2:
    #         self.loggers[2].log_image(key=f'{stage}/Confusion matrix', images=[cm_plot.figure], caption=[self.current_epoch])
    #     # self.loggers[0].experiment.add_figure(f'{stage}/Confusion matrix', cm_plot.figure, self.current_epoch)
    # else:
    #     ax.set_title(f'{stage}_{comment}')
    #     if comment: 
    #         stage += f'_{comment}'
    #     # fig_.savefig(f'{self.loggers[0].log_dir}/cm_{stage}.png', dpi=400)
    #     cm_plot.figure.savefig(f'{self.loggers[0].log_dir}/{stage}_cm.png', dpi=400)

    # # fig.clf()
    # cm_plot.figure.clf()
    # return cm_plot, optimal_threshold


if __name__ == '__main__':

    print(LABEL_MAP)