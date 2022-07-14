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
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from typing import Any, Dict, List, Optional, Type
import shutil

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
    if cfg.stage != 'test':
        
        tb_logger = pl_loggers.TensorBoardLogger(cfg.log_path,
                                                  # version = f'fold{cfg.Data.fold}'
                                                log_graph = True, default_hp_metric = False)
        # print(tb_logger.version)
        version = tb_logger.version
        #---->CSV
        csv_logger = pl_loggers.CSVLogger(cfg.log_path, version = version
                                        ) # version = f'fold{cfg.Data.fold}', 
        # print(csv_logger.version)
    else:  
        cfg.log_path = Path(cfg.log_path) / f'test'
        tb_logger = pl_loggers.TensorBoardLogger(cfg.log_path,
                                                version = f'test',
                                                log_graph = True, default_hp_metric = False)
        #---->CSV
        csv_logger = pl_loggers.CSVLogger(cfg.log_path,
                                        version = f'test', )
                              
    
    print(f'---->Log dir: {cfg.log_path}')

    # return tb_logger
    return [tb_logger, csv_logger]


#---->load Callback
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
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

    if cfg.General.server == 'train' :
        # save_path = Path(cfg.log_path) / 'lightning_logs' / f'version_{cfg.resume_version}' / last.ckpt
        Mycallbacks.append(ModelCheckpoint(monitor = 'val_loss',
                                         dirpath = str(output_path),
                                         filename = '{epoch:02d}-{val_loss:.4f}-{val_auc: .4f}',
                                         verbose = True,
                                         save_last = True,
                                         save_top_k = 1,
                                         mode = 'min',
                                         save_weights_only = True))
        Mycallbacks.append(ModelCheckpoint(monitor = 'val_auc',
                                         dirpath = str(output_path),
                                         filename = '{epoch:02d}-{val_loss:.4f}-{val_auc:.4f}',
                                         verbose = True,
                                         save_last = True,
                                         save_top_k = 1,
                                         mode = 'max',
                                         save_weights_only = True))
    return Mycallbacks

#---->val loss
import torch
import torch.nn.functional as F
def cross_entropy_torch(x, y):
    x_softmax = [F.softmax(x[i], dim=0) for i in range(len(x))]
    x_log = torch.tensor([torch.log(x_softmax[i][y[i]]) for i in range(y.shape[0])])
    loss = - torch.sum(x_log) / y.shape[0]
    return loss

#-----> convert labels for task
label_map = {
    'bin': {'0': 0, '1': 1, '2': 1, '3': 1, '4': 1, '5': None},
    'tcmr_viral': {'0': None, '1': 0, '2': None, '3': None, '4': 1, '5': None},
    'no_viral': {'0': 0, '1': 1, '2': 2, '3': 3, '4': None, '5': None},
    'no_other': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': None},
    'no_stable': {'0': None, '1': 1, '2': 2, '3': 3, '4': None, '5': None},
    'all': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5},

}
def convert_labels_for_task(task, label):

    return label_map[task][label]


