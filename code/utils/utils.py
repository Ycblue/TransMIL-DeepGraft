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
# from pytorch_lightning.loops.base import Loop
# from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.callbacks import LearningRateMonitor
from typing import Any, Dict, List, Optional, Type
import shutil

from matplotlib import pyplot as plt
plt.style.use('tableau-colorblind10')
import pandas as pd
import json
import pprint
import seaborn as sns

import torchmetrics
from torchmetrics import PrecisionRecallCurve, ROC
from torchmetrics.functional.classification import binary_auroc, multiclass_auroc, binary_precision_recall_curve, multiclass_precision_recall_curve, confusion_matrix
from torchmetrics.utilities.compute import _auc_compute_without_check, _auc_compute

LABEL_MAP = {
    # 'bin': {'0': 0, '1': 1, '2': 1, '3': 1, '4': 1, '5': None},
    # 'tcmr_viral': {'0': None, '1': 0, '2': None, '3': None, '4': 1, '5': None},
    # 'no_viral': {'0': 0, '1': 1, '2': 2, '3': 3, '4': None, '5': None},
    'no_other': {'0': 'Normal', '1': 'TCMR', '2': 'ABMR', '3': 'Mixed', '4': 'Viral'},
    # 'no_stable': {'0': None, '1': 1, '2': 2, '3': 3, '4': None, '5': None},
    # 'all': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5},
    'rejections': {'0': 'TCMR', '1': 'ABMR', '2': 'Mixed'},
    'norm_rest': {'0': 'Normal', '1': 'Disease'},
    'rej_rest': {'0': 'Rejection', '1': 'Rest'},
    'rest_rej': {'0': 'Rest', '1': 'Rejection'},
    'norm_rej_rest': {'0': 'Normal', '1': 'Rejection', '2': 'Rest'},

}
COLOR_MAP = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


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
        # version = tb_logger.version
        csv_logger_path = Path(cfg.log_path) / 'lightning_logs' / f'version_{cfg.version}' / f'test_epoch_{cfg.epoch}'
        csv_logger = pl_loggers.CSVLogger(csv_logger_path,
                                        version = cfg.version)
        # wandb_logger = pl_loggers.WandbLogger(project=f'{cfg.task}_{cfg.log_name}')                      
    
    print(f'---->Log dir: {cfg.log_path}')

    # return tb_logger
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
                                            filename = '{epoch:02d}-{val_loss:.4f}-{val_auc: .4f}-{val_patient_auc:.4f}',
                                            verbose = True,
                                            save_last = True,
                                            save_top_k = 3,
                                            mode = 'min',
                                            save_weights_only = True))
            Mycallbacks.append(ModelCheckpoint(monitor = 'val_accuracy',
                                            dirpath = str(output_path),
                                            filename = '{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}-{val_patient_auc: .4f}',
                                            verbose = True,
                                            save_last = True,
                                            save_top_k = 3,
                                            mode = 'max',
                                            save_weights_only = True))
            Mycallbacks.append(ModelCheckpoint(monitor = 'val_patient_auc',
                                            dirpath = str(output_path),
                                            filename = '{epoch:02d}-{val_loss:.4f}-{val_auc:.4f}-{val_patient_auc:.4f}',
                                            verbose = True,
                                            save_last = True,
                                            save_top_k = 3,
                                            mode = 'max',
                                            save_weights_only = True))

    swa = StochasticWeightAveraging(swa_lrs=1e-2)
    Mycallbacks.append(swa)

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


def get_optimal_operating_point(fpr, tpr, thresholds):
    '''
    Returns: 
        optimal_fpr [Tensor]
        optimal_tpr [Tensor]
        optimal_threshold [Float]
    '''

    youden_j = tpr - fpr
    optimal_idx = torch.argmax(youden_j)
    # print(youden_j[optimal_idx])
    optimal_threshold = thresholds[optimal_idx].item()
    # print(optimal_threshold)
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]

    return optimal_fpr, optimal_tpr, optimal_threshold



def get_roc_curve(probs, target, task):
        
    task_label_map = LABEL_MAP[task]

    if task == 'norm_rest' or task == 'rej_rest' or task == 'rest_rej':

        n_classes = 2
        # PRC = torchmetrics.PrecisionRecallCurve(task='binary')
        ROC = torchmetrics.ROC(task='binary')
    else: 
        n_classes = 3
        # PRC = torchmetrics.PrecisionRecallCurve(task='multiclass', num_classes = n_classes)
        ROC = torchmetrics.ROC(task='multiclass', num_classes=n_classes)

    fpr_list, tpr_list, thresholds = ROC(probs, target)

    # self.AUROC(out_probs, target.squeeze())

    fig, ax = plt.subplots(figsize=(6,6))

    if n_classes > 2:
        auroc_score = multiclass_auroc(probs, target, num_classes=n_classes, average=None)
        for i in range(len(fpr_list)):

            class_label = task_label_map[str(i)]
            color = COLOR_MAP[i]
            
            fpr = fpr_list[i].cpu().numpy()
            tpr = tpr_list[i].cpu().numpy()
            # ax.plot(fpr, tpr, label=f'class_{i}, AUROC={auroc_score[i]}')
            df = pd.DataFrame(data = {'fpr': fpr, 'tpr': tpr})
            line_plot = sns.lineplot(data=df, x='fpr', y='tpr', label=f'{class_label}={auroc_score[i]:.3f}', legend='full', color=color)
        
    else: 
        auroc_score = binary_auroc(probs, target)
        color = COLOR_MAP[0]
        
        optimal_fpr, optimal_tpr, optimal_threshold = get_optimal_operating_point(fpr_list, tpr_list, thresholds)
        fpr = fpr_list.cpu().numpy()
        tpr = tpr_list.cpu().numpy()
        optimal_fpr = optimal_fpr.cpu().numpy()
        optimal_tpr = optimal_tpr.cpu().numpy()

        df = pd.DataFrame(data = {'fpr': fpr, 'tpr': tpr})
        line_plot = sns.lineplot(data=df, x='fpr', y='tpr', label=f'{auroc_score:.3f}', legend='full', color=color) #AUROC
        # ax.plot([0, 1], [optimal_tpr, optimal_tpr], linestyle='--', color='black', label=f'OOP={optimal_threshold:.3f}')
        # ax.plot([optimal_fpr, optimal_fpr], [0, 1], linestyle='--', color='black')
    ax.plot([0,1], [0,1], linestyle='--', color='red')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('False positive rate (1-specificity)', fontsize=18)
    ax.set_ylabel('True positive rate (sensitivity)', fontsize=18)
    # ax.set_title('ROC curve')
    ax.legend(loc='lower right', fontsize=15)

    return line_plot

def get_pr_curve(probs, target, task):

    if task == 'norm_rest' or task == 'rej_rest' or task == 'rest_rej':
        n_classes = 2 
        PRC = torchmetrics.PrecisionRecallCurve(task='binary')
        # ROC = torchmetrics.ROC(task='binary')
    else: 
        n_classes = 3
        PRC = torchmetrics.PrecisionRecallCurve(task='multiclass', num_classes = n_classes)
        # ROC = torchmetrics.ROC(task='multiclass', num_classes=n_classes)
    
    
    fig, ax = plt.subplots(figsize=(6,6))


    
    if n_classes > 2:

        precision, recall, thresholds = multiclass_precision_recall_curve(probs, target, num_classes=n_classes)
        task_label_map = LABEL_MAP[task]
        
        for i in range(len(precision)):

            class_label = task_label_map[str(i)]
            color = COLOR_MAP[i]

            re = recall[i]
            pr = precision[i]
            
            partial_auc = _auc_compute(re, pr, 1.0)
            df = pd.DataFrame(data = {'re': re.cpu().numpy(), 'pr': pr.cpu().numpy()})
            line_plot = sns.lineplot(data=df, x='re', y='pr', label=f'{class_label}={partial_auc:.3f}', legend='full', color=color)

            baseline = len(target[target==i]) / len(target)
            ax.plot([0,1],[baseline, baseline], linestyle='--', label=f'Baseline={baseline:.3f}', color=color)

    else: 
        # print(fpr_list)
        color = COLOR_MAP[0]
        precision, recall, thresholds = binary_precision_recall_curve(probs, target)
        baseline = len(target[target==1]) / len(target)
        
        pr = precision
        re = recall
        partial_auc = _auc_compute(re, pr, 1.0)
        # ax.plot(re, pr)
        df = pd.DataFrame(data = {'re': re.cpu().numpy(), 'pr': pr.cpu().numpy()})
        line_plot = sns.lineplot(data=df, x='re', y='pr', label=f'{partial_auc:.3f}', legend='full', color=color)
        
    
        ax.plot([0,1], [baseline, baseline], linestyle='--', label=f'Baseline={baseline:.3f}', color=color)

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('Recall', fontsize=18)
    ax.set_ylabel('Precision', fontsize=18)
    # ax.set_title('PR curve')
    ax.legend(loc='lower right', fontsize=15)

    return line_plot

def get_confusion_matrix(probs, target, task, threshold_csv_path, comment='patient', stage='test'): # threshold

        
        if task == 'norm_rest' or task == 'rej_rest' or task == 'rest_rej':

            n_classes = 2 
            ROC = torchmetrics.ROC(task='binary')
        else: 
            n_classes = 3
            ROC = torchmetrics.ROC(task='multiclass', num_classes=n_classes)


        # preds = torch.argmax(probs, dim=1)
        # if self.n_classes <= 2:
        #     probs = probs[:,1] 

        # read threshold file
        # threshold_csv_path = f'{self.loggers[0].log_dir}/val_thresholds.csv'
        # if not Path(threshold_csv_path).is_file():
        #     # thresh_dict = {'index': ['train', 'val'], 'columns': , 'data': [[0.5, 0.5], [0.5, 0.5]]}
        #     thresh_df = pd.DataFrame({'slide': [0.5], 'patient': [0.5]})
        #     thresh_df.to_csv(threshold_csv_path, index=False)

        # thresh_df = pd.read_csv(threshold_csv_path)
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

        if n_classes == 2:    
            fpr_list, tpr_list, thresholds = ROC(probs, target)
            optimal_fpr, optimal_tpr, optimal_threshold = get_optimal_operating_point(fpr_list, tpr_list, thresholds)
        else:
            optimal_threshold = 0.5
        # optimal_threshold = thresh_df.at[0, comment]

        print(f'Optimal Threshold {stage} {comment}: ', optimal_threshold)
            # optimal_threshold = 0.5 # manually change to val_optimal_threshold for testing

        # print(confmat)
        # confmat = self.confusion_matrix(preds, target, threshold=optimal_threshold)
        if n_classes == 2:
            confmat = confusion_matrix(probs, target, task='binary', threshold=optimal_threshold)
        elif n_classes > 2: 
            confmat = confusion_matrix(probs, target, task='multiclass', num_classes=n_classes)

        cm_labels = LABEL_MAP[task].values()

        # fig, ax = plt.subplots()
        figsize=plt.rcParams.get('figure.figsize')
        plt.figure(figsize=figsize)

        # df_cm = pd.DataFrame(confmat.cpu().numpy(), index=range(self.n_classes), columns=range(self.n_classes))
        df_cm = pd.DataFrame(confmat.cpu().numpy(), index=cm_labels, columns=cm_labels)
        # fig_ = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Spectral').get_figure()
        # sns.set(font_scale=1.5)
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'fontsize': 'x-large', 'multialignment':'center'})

        plt.yticks(va='center')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


        
        
        
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
        return plt


if __name__ == '__main__':

    print(LABEL_MAP)