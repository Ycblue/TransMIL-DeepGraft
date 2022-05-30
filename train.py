import argparse
from pathlib import Path
import numpy as np
import glob

from sklearn.model_selection import KFold

from datasets.data_interface import DataInterface, MILDataModule
from models.model_interface import ModelInterface
import models.vision_transformer as vits
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loops import KFoldLoop
import torch

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='DeepGraft/TransMIL.yaml',type=str)
    parser.add_argument('--gpus', default = 2, type=int)
    parser.add_argument('--loss', default = 'CrossEntropyLoss', type=str)
    parser.add_argument('--fold', default = 0)
    parser.add_argument('--bag_size', default = 1024, type=int)
    args = parser.parse_args()
    return args

#---->main
def main(cfg):

    torch.set_num_threads(16)

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    #---->load loggers
    cfg.load_loggers = load_loggers(cfg)
    # print(cfg.load_loggers)

    #---->load callbacks
    cfg.callbacks = load_callbacks(cfg)

    #---->Define Data 
    # DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
    #             'train_num_workers': cfg.Data.train_dataloader.num_workers,
    #             'test_batch_size': cfg.Data.test_dataloader.batch_size,
    #             'test_num_workers': cfg.Data.test_dataloader.num_workers,
    #             'dataset_name': cfg.Data.dataset_name,
    #             'dataset_cfg': cfg.Data,}
    # dm = DataInterface(**DataInterface_dict)
    home = Path.cwd().parts[1]
    DataInterface_dict = {
                'data_root': cfg.Data.data_dir,
                'label_path': cfg.Data.label_file,
                'batch_size': cfg.Data.train_dataloader.batch_size,
                'num_workers': cfg.Data.train_dataloader.num_workers,
                'n_classes': cfg.Model.n_classes,
                'backbone': cfg.Model.backbone,
                'bag_size': cfg.Data.bag_size,
                }

    dm = MILDataModule(**DataInterface_dict)
    

    #---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path,
                            'backbone': cfg.Model.backbone,
                            }
    model = ModelInterface(**ModelInterface_dict)
    
    #---->Instantiate Trainer
    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs= cfg.General.epochs,
        min_epochs = 200,
        # gpus=cfg.General.gpus,
        gpus = [2,3],
        strategy='ddp',
        amp_backend='native',
        # amp_level=cfg.General.amp_level,  
        precision=cfg.General.precision,  
        accumulate_grad_batches=cfg.General.grad_acc,
        # fast_dev_run = True,
        
        # deterministic=True,
        check_val_every_n_epoch=10,
    )

    #---->train or test
    if cfg.General.server == 'train':
        trainer.fit_loop = KFoldLoop(3, trainer.fit_loop, )
        trainer.fit(model = model, datamodule = dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            print(path)
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)


if __name__ == '__main__':

    args = make_parse()
    cfg = read_yaml(args.config)

    #---->update
    cfg.config = args.config
    cfg.General.gpus = [args.gpus]
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold
    cfg.Loss.base_loss = args.loss
    cfg.Data.bag_size = args.bag_size
    

    #---->main
    main(cfg)
 