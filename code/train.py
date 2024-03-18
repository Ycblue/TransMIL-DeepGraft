import argparse
from pathlib import Path
import numpy as np
import glob

from sklearn.model_selection import KFold

from datasets.data_interface import MILDataModule, CrossVal_MILDataModule
# from datasets.data_interface import DataInterface, MILDataModule, CrossVal_MILDataModule
from models.model_interface import ModelInterface
from models.model_interface_classic import ModelInterface_Classic
# from models.model_interface_dtfd import ModelInterface_DTFD
import models.vision_transformer as vits
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
import torch
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.tuner import Tuner

from experiment_impact_tracker.data_interface import DataInterface
from experiment_impact_tracker.compute_tracker import ImpactTracker
# from train_loop import KFoldLoop
# from pytorch_lightning.plugins.training_type import DDPPlugin


# try:
#     import apex
#     from apex.parallel import DistributedDataParallel
#     print('Apex available.')
# except ModuleNotFoundError:
#     # Error handling
#     pass

# def unwrap_lightning_module(wrapped_model):
#     from apex.parallel import DistributedDataParallel
#     from pytorch_lightning.overrides.base import (
#         _LightningModuleWrapperBase,
#         _LightningPrecisionModuleWrapperBase,
#     )

#     model = wrapped_model
#     if isinstance(model, DistributedDataParallel):
#         model = unwrap_lightning_module(model.module)
#     if isinstance(
#         model, (_LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase)
#     ):
#         model = unwrap_lightning_module(model.module)
#     return model


# class ApexDDPPlugin(DDPPlugin):
#     def _setup_model(self, model):
#         from apex.parallel import DistributedDataParallel

#         return DistributedDataParallel(model, delay_allreduce=False)

#     @property
#     def lightning_module(self):
#         return unwrap_lightning_module(self._model)



#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='DeepGraft/TransMIL.yaml',type=str)
    parser.add_argument('--version', default=2,type=int)
    parser.add_argument('--epoch', default=None,type=str)

    parser.add_argument('--gpus', nargs='+', default = [0], type=int)
    parser.add_argument('--loss', default = 'CrossEntropyLoss', type=str)
    parser.add_argument('--fold', default = 0)
    parser.add_argument('--bag_size', default = 1024, type=int)
    # parser.add_argument('--batch_size', default = 1, type=int)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--label_file', type=str)
    # parser.add_argument('--from_ft', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--fast_dev_run', action='store_true')
    

    args = parser.parse_args()
    return args

#---->main
def main(cfg):

    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('high')

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

    train_classic = False
    if cfg.Model.name in ['inception', 'resnet18', 'vit', 'efficientnet']:
        train_classic = True
        use_features = False

    if cfg.Model.backbone == 'features':
        use_features = True
    # elif cfg.Model.backbone == 'simple':
    #     use_features = False
    else: use_features = False

    # print(cfg.Data.bag_size)
    
    DataInterface_dict = {
                'data_root': cfg.Data.data_dir,
                'label_path': cfg.Data.label_file,
                'batch_size': cfg.Data.train_dataloader.batch_size,
                'num_workers': cfg.Data.train_dataloader.num_workers,
                'n_classes': cfg.Model.n_classes,
                'bag_size': cfg.Data.bag_size,
                'use_features': use_features,
                'mixup': cfg.Data.mixup,
                'aug': cfg.Data.aug,
                'cache': cfg.Data.cache,
                'train_classic': train_classic,
                'model_name': cfg.Model.name,
                'in_features': cfg.Model.in_features,
                'feature_extractor': cfg.Data.feature_extractor,
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
                            'in_features': cfg.Model.in_features,
                            'out_features': cfg.Model.out_features,
                            'bag_size': cfg.Data.bag_size,
                            # 'batch_size': cfg.Data.train_dataloader.batch_size,
                            }

    if train_classic:
        model = ModelInterface_Classic(**ModelInterface_dict)
    # elif cfg.Model.name == 'DTFDMIL':
    #     model = ModelInterface_DTFD(**ModelInterface_dict)
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
            strategy='ddp_find_unused_parameters_true', # inception with frozen params
            # plugins=plugins,
            devices=cfg.General.gpus,
            # replace_sampler_ddp=False,
            # amp_backend='native',
            # precision='16-mixed',  
            precision=cfg.General.precision,  
            # accumulate_grad_batches=cfg.General.grad_acc,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            gradient_clip_val=0.0,
            fast_dev_run = cfg.fast_dev_run,
            # limit_train_batches=1,
            
            # deterministic=True,
            accumulate_grad_batches=10,
            check_val_every_n_epoch=1,
        )
    else:
        trainer = Trainer(
            # deterministic=True,
            num_sanity_val_steps=-1, 
            logger=cfg.load_loggers,
            callbacks=cfg.callbacks,
            max_epochs= cfg.General.epochs,
            # max_epochs= 2,
            min_epochs = 500,

            # gpus=cfg.General.gpus,
            accelerator='gpu',
            devices=cfg.General.gpus,
            # precision='16-mixed',  
            precision=cfg.General.precision,  
            accumulate_grad_batches=cfg.General.grad_acc,
            gradient_clip_val=0.0,
            # log_every_n_steps=10,
            fast_dev_run = cfg.fast_dev_run,
            # limit_train_batches=1,
            
            # deterministic=True,
            # num_sanity_val_steps=0,
            check_val_every_n_epoch=1,
            log_every_n_steps=20,
            # profiler='simple',

        )
    # print(cfg.log_path)
    # print(trainer.loggers[0].log_dir)
    # print(trainer.loggers[1].log_dir)
    #----> Copy Code

    # home = Path.cwd()[0]
    # comment out for fast_dev_run because no logger is initiated
    if not cfg.fast_dev_run:
        if cfg.General.server == 'train':
            copy_path = Path(trainer.loggers[0].log_dir) / 'code'
            copy_path.mkdir(parents=True, exist_ok=True)
            copy_origin = '/' / Path('/'.join(cfg.log_path.parts[1:5])) / 'code'
            shutil.copytree(copy_origin, copy_path, dirs_exist_ok=True)

    #---->train or test
    if cfg.resume_training:
        last_ckpt = Path(cfg.log_path) / 'lightning_logs' / f'version_{cfg.version}' / 'checkpoints' / 'last.ckpt'
        print('Resume Training from: ', last_ckpt)
        model = model.load_from_checkpoint(checkpoint_path=last_ckpt, cfg=cfg)
        # trainer.fit(model = model, ckpt_path=last_ckpt) #, datamodule = dm
        trainer.fit(model, dm)
    # print(cfg.resume_training)

    if cfg.General.server == 'train' or cfg.General.server == 'fine_tune':

        # k-fold cross validation loop
        if cfg.Data.cross_val: 
            internal_fit_loop = trainer.fit_loop
            trainer.fit_loop = KFoldLoop(cfg.Data.nfold, export_path = cfg.log_path, **ModelInterface_dict)
            trainer.fit_loop.connect(internal_fit_loop)
            trainer.fit(model, dm)
        elif cfg.resume_training:
            last_ckpt = Path(cfg.log_path) / 'lightning_logs' / f'version_{cfg.version}' / 'checkpoints' / 'last.ckpt'
            print('Resume Training from: ', last_ckpt)
            model = model.load_from_checkpoint(checkpoint_path=last_ckpt, cfg=cfg)
            # trainer.fit(model = model, ckpt_path=last_ckpt) #, datamodule = dm
            trainer.fit(model, dm)
        else:                                                   
            # tuner = Tuner(trainer)
            # tuner.scale_batch_size(model, datamodule=dm)
            # tuner.lr_find(model, datamodule=dm)
            trainer.fit(model = model, datamodule = dm)
            # trainer.test(model = model, datamodule = dm)
    else:
        # if cfg.fine_tune:
        #    log_path = Path(cfg.log_path) / 'lightning_logs' / f'version_{cfg.version}'/'checkpoints' 
        # else:
        log_path = Path(cfg.log_path) / 'lightning_logs' / f'version_{cfg.version}'/'checkpoints' 

        model_paths = list(log_path.glob('*.ckpt'))
        # print(model_paths)
        # print(f'epoch={cfg.epoch}')
        # for i in model_paths:
        #     print(f'epoch={cfg.epoch}' in str(i))
        if not cfg.epoch:
            model_paths = [str(model_path) for model_path in model_paths if f'.ckpt' in str(model_path)]
        elif cfg.epoch == 'last':
            model_paths = [str(model_path) for model_path in model_paths if f'last' in str(model_path)]
        elif int(cfg.epoch) < 10:
            cfg.epoch = f'0{cfg.epoch}'
        else:
            model_paths = [str(model_path) for model_path in model_paths if f'epoch={cfg.epoch}' in str(model_path)]
        # model_paths = [f'{log_path}/epoch=279-val_loss=0.4009.ckpt']
        # model_paths = [m for m in model_paths if Path(m).stem[-2]!='v']
        # print(model_paths)
        
        for path in model_paths:
            print('Epoch: ', path)
        # path  = model_paths[0]
            if 'last' in str(path):
                epoch = 'last'
            else:
                name = Path(path).stem
                epoch = name.split('-')[0].split('=')[1]
            # print(int(Path(path).stem.split('-')[0].split('=')[1]))
            cfg.epoch = epoch
            # print(cfg)
            cfg.callbacks = load_callbacks(cfg, save_path)
            # print(cfg)
            # print(trainer.callbacks)
            cfg.load_loggers = load_loggers(cfg)
            trainer = Trainer(
                logger=cfg.load_loggers,
                callbacks=cfg.callbacks,
                max_epochs= cfg.General.epochs,
                min_epochs = 100,
                accelerator='gpu',
                devices=cfg.General.gpus,
                precision=cfg.General.precision,  
                accumulate_grad_batches=cfg.General.grad_acc,
                gradient_clip_val=0.0,
            )
            # # print('Loading from: ', path)
            model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            if cfg.General.server == 'val':
                trainer.validate(model=model, datamodule=dm)
            elif cfg.General.server == 'test':
                trainer.test(model=model, datamodule=dm)


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
    # cfg.Data.bag_size = args.bag_size
    cfg.version = args.version
    cfg.fine_tune = args.fine_tune
    cfg.resume_training = args.resume_training
    cfg.fast_dev_run = args.fast_dev_run
    

    if args.label_file: 
        cfg.Data.label_file = '/home/ylan/DeepGraft/training_tables/' + args.label_file

    cfg = check_home(cfg)

    config_path = '/'.join(Path(cfg.config).parts[1:])
    log_path = Path(cfg.General.log_path) / str(Path(config_path).parent)
    # print(log_path)


    Path(cfg.General.log_path).mkdir(exist_ok=True, parents=True)
    log_name =  f'_{cfg.Model.backbone}' + f'_{cfg.Loss.base_loss}'
    task = '_'.join(Path(cfg.config).name[:-5].split('_')[2:])
    task = task.split('-')[0]
    cfg.task = task
    # task = Path(cfg.config).name[:-5].split('_')[2:][0]
    cfg.log_path = log_path / f'{cfg.Model.name}' / task / log_name 
    cfg.log_name = log_name
    print(cfg.task)

    if cfg.Data.feature_extractor == 'retccl':
        cfg.Model.in_features = 2048
    elif cfg.Data.feature_extractor == 'histoencoder':
        cfg.Model.in_features = 384
    elif cfg.Data.feature_extractor == 'ctranspath':
        cfg.Model.in_features = 784



    cfg.epoch = args.epoch
    

    # ---->main
    
    # tracker = ImpactTracker(f'co2log/')
    # tracker.launch_impact_monitor()

    main(cfg)
 

    # tracker.stop()
    # data_interface = DataInterface(['co2log'])

    # epochs = 50
    # # epochs = cfg.epoch
    # bag_size = 1000
    # data_size = 3489
    # print('average bag_size: ', bag_size)
    # print('====================')
    # print(f'{cfg.Model.name}')
    # print('====================')
    # print('kg_carbon: ', data_interface.kg_carbon)
    # print('kg_carbon/epoch: ', data_interface.kg_carbon / epochs)
    # print('g_carbon: ', data_interface.kg_carbon * 1000)
    # print('g_carbon/epoch: ', data_interface.kg_carbon * 1000 / epochs)
    # print('g_carbon/slide with 1000 patches: ', 1000*(data_interface.kg_carbon * 1000 / (data_size) / bag_size))
    # kg_carbon = data_interface.kg_carbon / epochs
    # print(kg_carbon)

    # '''Netherlands'''

    # txt_path = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/co2_emission/{self.model_name}_vis.txt'
    # # Path(txt_path).mkdir(exist_ok=True)
    # bag_size = sum(bag_array) / len(bag_array)
    # with open(txt_path, 'a') as f:
    #     f.write(f'==================================================================================== \n')
    #     f.write(f'Emissions calculated for {data_size} slides, {bag_size} patches/slide, per epoch \n')
    #     f.write(f'{self.model_name}: {kg_carbon} [kg]\n')
    #     f.write(f'{self.model_name}: {kg_carbon*1000} [g]\n')
    #     f.write(f'Emissions calculated for 1 slides, {bag_size} patches/slide, per epoch \n')
    #     f.write(f'{self.model_name}: {kg_carbon/data_size} [kg]\n')
    #     f.write(f'{self.model_name}: {kg_carbon*1000/data_size} [g]\n')
    #     f.write(f'==================================================================================== \n')
