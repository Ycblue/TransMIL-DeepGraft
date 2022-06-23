import argparse
from pathlib import Path
import numpy as np
import glob

from sklearn.model_selection import KFold

from datasets.data_interface import DataInterface, MILDataModule, CrossVal_MILDataModule
from models.model_interface import ModelInterface
import models.vision_transformer as vits
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from train_loop import KFoldLoop

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='DeepGraft/TransMIL.yaml',type=str)
    parser.add_argument('--version', default=0,type=int)
    parser.add_argument('--epoch', default='0',type=str)
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
    # cfg.load_loggers = load_loggers(cfg)

    # print(cfg.load_loggers)
    # save_path = Path(cfg.load_loggers[0].log_dir) 

    #---->load callbacks
    # cfg.callbacks = load_callbacks(cfg, save_path)

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
        # logger=cfg.load_loggers,
        # callbacks=cfg.callbacks,
        max_epochs= cfg.General.epochs,
        min_epochs = 200,
        gpus=cfg.General.gpus,
        # gpus = [0,2],
        # strategy='ddp',
        amp_backend='native',
        # amp_level=cfg.General.amp_level,  
        precision=cfg.General.precision,  
        accumulate_grad_batches=cfg.General.grad_acc,
        # fast_dev_run = True,
        
        # deterministic=True,
        check_val_every_n_epoch=10,
    )

    #---->train or test
    log_path = cfg.log_path
    # print(log_path)
    # log_path = Path('lightning_logs/2/checkpoints')
    model_paths = list(log_path.glob('*.ckpt'))

    if cfg.epoch == 'last':
        model_paths = [str(model_path) for model_path in model_paths if f'last' in str(model_path)]
    else:
        model_paths = [str(model_path) for model_path in model_paths if f'epoch={cfg.epoch}' in str(model_path)]

    # model_paths = [str(model_path) for model_path in model_paths if f'epoch={cfg.epoch}' in str(model_path)]
    # model_paths = [f'lightning_logs/0/.ckpt']
    # model_paths = [f'{log_path}/last.ckpt']
    if not model_paths: 
        print('No Checkpoints vailable!')
    for path in model_paths:
        # with open(f'{log_path}/test_metrics.txt', 'w') as f:
        #     f.write(str(path) + '\n')
        print(path)
        new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
        trainer.test(model=new_model, datamodule=dm)
    
    # Top 5 scoring patches for patient
    # GradCam


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
    cfg.version = args.version
    cfg.epoch = args.epoch

    log_path = Path(cfg.General.log_path) / str(Path(cfg.config).parent)
    Path(cfg.General.log_path).mkdir(exist_ok=True, parents=True)
    log_name =  f'_{cfg.Model.backbone}' + f'_{cfg.Loss.base_loss}'
    task = '_'.join(Path(cfg.config).name[:-5].split('_')[2:])
    # task = Path(cfg.config).name[:-5].split('_')[2:][0]
    cfg.log_path = log_path / f'{cfg.Model.name}' / task / log_name / 'lightning_logs' / f'version_{cfg.version}' / 'checkpoints'
    
    

    #---->main
    main(cfg)
 