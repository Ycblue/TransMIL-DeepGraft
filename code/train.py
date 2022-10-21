import argparse
from pathlib import Path
import numpy as np
import glob

from sklearn.model_selection import KFold

from datasets.data_interface import DataInterface, MILDataModule, CrossVal_MILDataModule
from models.model_interface import ModelInterface
from models.model_interface_dtfd import ModelInterface_DTFD
import models.vision_transformer as vits
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
import torch
from train_loop import KFoldLoop
from pytorch_lightning.plugins.training_type import DDPPlugin


try:
    import apex
    from apex.parallel import DistributedDataParallel
    print('Apex available.')
except ModuleNotFoundError:
    # Error handling
    pass

def unwrap_lightning_module(wrapped_model):
    from apex.parallel import DistributedDataParallel
    from pytorch_lightning.overrides.base import (
        _LightningModuleWrapperBase,
        _LightningPrecisionModuleWrapperBase,
    )

    model = wrapped_model
    if isinstance(model, DistributedDataParallel):
        model = unwrap_lightning_module(model.module)
    if isinstance(
        model, (_LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase)
    ):
        model = unwrap_lightning_module(model.module)
    return model


class ApexDDPPlugin(DDPPlugin):
    def _setup_model(self, model):
        from apex.parallel import DistributedDataParallel

        return DistributedDataParallel(model, delay_allreduce=False)

    @property
    def lightning_module(self):
        return unwrap_lightning_module(self._model)



#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='DeepGraft/TransMIL.yaml',type=str)
    parser.add_argument('--version', default=2,type=int)
    parser.add_argument('--gpus', nargs='+', default = [2], type=int)
    parser.add_argument('--loss', default = 'CrossEntropyLoss', type=str)
    parser.add_argument('--fold', default = 0)
    parser.add_argument('--bag_size', default = 1024, type=int)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--label_file', type=str)
    # parser.add_argument('--ckpt_path', default = , type=str)
    

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
    save_path = Path(cfg.load_loggers[0].log_dir) 

    #---->load callbacks
    cfg.callbacks = load_callbacks(cfg, save_path)

    #---->Define Data 
    # DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
    #             'train_num_workers': cfg.Data.train_dataloader.num_workers,
    #             'test_batch_size': cfg.Data.test_dataloader.batch_size,
    #             'test_num_workers': cfg.Data.test_dataloader.num_workers,
    #             'dataset_name': cfg.Data.dataset_name,
    #             'dataset_cfg': cfg.Data,}
    # dm = DataInterface(**DataInterface_dict)
    home = Path.cwd().parts[1]
    if cfg.Model.backbone == 'features':
        use_features = True
    else: use_features = False
    DataInterface_dict = {
                'data_root': cfg.Data.data_dir,
                'label_path': cfg.Data.label_file,
                'batch_size': cfg.Data.train_dataloader.batch_size,
                'num_workers': cfg.Data.train_dataloader.num_workers,
                'n_classes': cfg.Model.n_classes,
                'bag_size': cfg.Data.bag_size,
                'use_features': use_features,
                }

    if cfg.Data.cross_val:
        dm = CrossVal_MILDataModule(**DataInterface_dict)
    else: dm = MILDataModule(**DataInterface_dict)
    

    #---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path,
                            'backbone': cfg.Model.backbone,
                            'task': cfg.task,
                            }
    if cfg.Model.name == 'DTFDMIL':
        model = ModelInterface_DTFD(**ModelInterface_dict)
    else:
        model = ModelInterface(**ModelInterface_dict)
    
    #---->Instantiate Trainer
    # plugins = []
    # if apex: 
    #     plugins.append(ApexDDPPlugin())

    if len(cfg.General.gpus) > 1:
        trainer = Trainer(
            logger=cfg.load_loggers,
            callbacks=cfg.callbacks,
            max_epochs= cfg.General.epochs,
            min_epochs = 100,
            accelerator='gpu',
            # plugins=plugins,
            devices=cfg.General.gpus,
            strategy=DDPStrategy(find_unused_parameters=False),
            replace_sampler_ddp=False,
            amp_backend='native',
            precision=cfg.General.precision,  
            # accumulate_grad_batches=cfg.General.grad_acc,
            gradient_clip_val=0.0,
            # fast_dev_run = True,
            # limit_train_batches=1,
            
            # deterministic=True,
            check_val_every_n_epoch=5,
        )
    else:
        trainer = Trainer(
            # num_sanity_val_steps=0, 
            logger=cfg.load_loggers,
            callbacks=cfg.callbacks,
            max_epochs= cfg.General.epochs,
            min_epochs = 100,

            # gpus=cfg.General.gpus,
            accelerator='gpu'
            devices=cfg.General.gpus,
            amp_backend='native',
            # amp_level=cfg.General.amp_level,  
            precision=cfg.General.precision,  
            accumulate_grad_batches=cfg.General.grad_acc,
            gradient_clip_val=0.0,
            # fast_dev_run = True,
            # limit_train_batches=1,
            
            # deterministic=True,
            check_val_every_n_epoch=5,
        )
    # print(cfg.log_path)
    # print(trainer.loggers[0].log_dir)
    # print(trainer.loggers[1].log_dir)
    #----> Copy Code

    # home = Path.cwd()[0]

    copy_path = Path(trainer.loggers[0].log_dir) / 'code'
    copy_path.mkdir(parents=True, exist_ok=True)
    copy_origin = '/' / Path('/'.join(cfg.log_path.parts[1:5])) / 'code'
    # print(copy_path)
    # print(copy_origin)
    shutil.copytree(copy_origin, copy_path, dirs_exist_ok=True)

    
    # print(trainer.loggers[0].log_dir)

    #---->train or test
    if cfg.resume_training:
        last_ckpt = log_path = Path(cfg.log_path) / 'lightning_logs' / f'version_{cfg.version}' / 'last.ckpt'
        # model = model.load_from_checkpoint(last_ckpt)
        # trainer.fit(model, dm) #, datamodule = dm
        trainer.fit(model = model, ckpt_path=last_ckpt) #, datamodule = dm

    if cfg.General.server == 'train':

        # k-fold cross validation loop
        if cfg.Data.cross_val: 
            internal_fit_loop = trainer.fit_loop
            trainer.fit_loop = KFoldLoop(cfg.Data.nfold, export_path = cfg.log_path, **ModelInterface_dict)
            trainer.fit_loop.connect(internal_fit_loop)
            trainer.fit(model, dm)
        else:
            trainer.fit(model = model, datamodule = dm)
    else:
        log_path = Path(cfg.log_path) / 'lightning_logs' / f'version_{cfg.version}' 

        test_path = Path(log_path) / 'test'
        for n in range(cfg.Model.n_classes):
            n_output_path = test_path / str(n)
            n_output_path.mkdir(parents=True, exist_ok=True)
        # print(cfg.log_path)
        model_paths = list(log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        # model_paths = [f'{log_path}/epoch=279-val_loss=0.4009.ckpt']
        for path in model_paths:
            print(path)
            new_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            trainer.test(model=new_model, datamodule=dm)


def check_home(cfg):
    # replace home directory
    
    home = Path.cwd().parts[1]

    x = cfg.General.log_path
    if Path(x).parts[1] != home:
        new_path = Path(home).joinpath(*Path(x).parts[2:])
        cfg.General.log_path = '/' + str(new_path)

    x = cfg.Data.data_dir
    if Path(x).parts[1] != home:
        new_path = Path(home).joinpath(*Path(x).parts[2:])
        cfg.Data.data_dir = '/' + str(new_path)
        
    x = cfg.Data.label_file
    if Path(x).parts[1] != home:
        new_path = Path(home).joinpath(*Path(x).parts[2:])
        cfg.Data.label_file = '/' + str(new_path)

    return cfg


if __name__ == '__main__':

    args = make_parse()
    cfg = read_yaml(args.config)

    #---->update
    cfg.config = args.config
    cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold
    cfg.Loss.base_loss = args.loss
    cfg.Data.bag_size = args.bag_size
    cfg.version = args.version
    if args.label_file: 
        cfg.Data.label_file = '/home/ylan/DeepGraft/training_tables/' + args.label_file

    cfg = check_home(cfg)

    config_path = '/'.join(Path(cfg.config).parts[1:])
    log_path = Path(cfg.General.log_path) / str(Path(config_path).parent)
    # print(log_path)


    Path(cfg.General.log_path).mkdir(exist_ok=True, parents=True)
    log_name =  f'_{cfg.Model.backbone}' + f'_{cfg.Loss.base_loss}'
    task = '_'.join(Path(cfg.config).name[:-5].split('_')[2:])
    cfg.task = task
    # task = Path(cfg.config).name[:-5].split('_')[2:][0]
    cfg.log_path = log_path / f'{cfg.Model.name}' / task / log_name 
    
    

    # ---->main
    main(cfg)
 