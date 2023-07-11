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
from train_loop import KFoldLoop
from pytorch_lightning.plugins.training_type import DDPPlugin
from datasets import FeatureBagLoader
import pprint


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
    parser.add_argument('--epoch', default='0',type=str)

    parser.add_argument('--gpus', nargs='+', default = [0], type=int)
    parser.add_argument('--loss', default = 'CrossEntropyLoss', type=str)
    parser.add_argument('--fold', default = 0)
    parser.add_argument('--bag_size', default = 1024, type=int)
    # parser.add_argument('--batch_size', default = 1, type=int)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--label_file', type=str)
    parser.add_argument('--fine_tune', action='store_true')
    

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
    if cfg.Model.name in ['inception', 'resnet18', 'resnet50', 'vit', 'efficientnet']:
        train_classic = True
        use_features = False

    if cfg.Model.backbone == 'features':
        use_features = True
    # elif cfg.Model.backbone == 'simple':
    #     use_features = False
    else: use_features = False
    


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
                'fine_tune': cfg.fine_tune,
                }

    # if cfg.Data.cross_val:
    #     dm = CrossVal_MILDataModule(**DataInterface_dict)
    dm = MILDataModule(**DataInterface_dict)
    
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
            min_epochs = 500,
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
            check_val_every_n_epoch=1,
        )
    else:
        trainer = Trainer(
            # default_root_dir=cfg.log_path / f'test_epoch_{cfg.epoch}' / 'checkpoint',
            num_sanity_val_steps=-1, 
            logger=cfg.load_loggers,
            callbacks=cfg.callbacks,
            max_epochs= 20,
            # min_epochs = 10,

            # gpus=cfg.General.gpus,
            accelerator='gpu',
            devices=cfg.General.gpus,
            # amp_backend='native',
            # amp_level=cfg.General.amp_level,  
            precision=cfg.General.precision,  
            accumulate_grad_batches=cfg.General.grad_acc,
            gradient_clip_val=0.0,
            log_every_n_steps=10,
            # fast_dev_run = True,
            # limit_train_batches=1,
            
            # deterministic=True,
            # num_sanity_val_steps=0,
            check_val_every_n_epoch=1,
        )
    # print(cfg.log_path)
    # print(trainer.loggers[0].log_dir)
    # print(trainer.loggers[1].log_dir)
    #----> Copy Code

    # home = Path.cwd()[0]
    data_root = cfg.Data.data_dir,
    label_path =  cfg.Data.label_file,
    batch_size = cfg.Data.train_dataloader.batch_size,
    n_classes = cfg.Model.n_classes,
    # 'bag_size': cfg.Data.bag_size,
    # 'use_features': use_features,
    mixup = cfg.Data.mixup,
    aug = cfg.Data.aug,
    cache = cfg.Data.cache,
    # 'train_classic': train_classic,
    model_name = cfg.Model.name,
    # 'fine_tune': True,

    if cfg.General.server == 'train':
        # data_root = 
        # train_data = FeatureBagLoader(data_root, label_path=label_path[0], mode='fine_tune', n_classes=n_classes, cache=cache, mixup=mixup, aug=aug, model=model_name)
        # test_data = FeatureBagLoader(data_root, label_path=label_path[0], mode='test', n_classes=n_classes, cache=False, model=model_name, mixup=False, aug=False)
        # ckpt = 'test/fine_tune/DeepGraft/TransMIL/norm_rest/epoch=10-val_loss=0.5929-val_auc=0.8765-val_patient_auc= 0.8953.ckpt'
        # last_ckpt = Path(cfg.log_path) / 'lightning_logs' / f'version_{cfg.version}' / 'checkpoints'/ 'last.ckpt'
        # trainer.fit(model = model, ckpt_path=ckpt) #, datamodule = dm

        
        ckptdir_path = Path(cfg.log_path) / 'lightning_logs' / f'version_{cfg.version}'/'checkpoints' 

        model_paths = list(ckptdir_path.glob('*.ckpt'))

        if cfg.epoch == 'last':
            model_paths = [str(model_path) for model_path in model_paths if f'last' in str(model_path)]
        elif int(cfg.epoch) < 10:
            cfg.epoch = f'0{cfg.epoch}'
        
        else:
            model_paths = [str(model_path) for model_path in model_paths if f'epoch={cfg.epoch}' in str(model_path)]
        # model_paths = [f'{log_path}/epoch=279-val_loss=0.4009.ckpt']
        # print(model_paths)
        
        # for path in model_paths:
        path  = model_paths[0]
        # print(path)
            # print(path)
        # train_dl = DataLoader(train_data, batch_size = batch_size, num_workers=8)
        
        model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
        trainer.fit(model=model, datamodule=dm)

        ft_ckptdir_path = cfg.log_path / 'lightning_logs' / f'version_{cfg.version}'/ f'ft_epoch_{cfg.epoch}'/ 'checkpoints'
        ft_model_paths = list(ft_ckptdir_path.glob('*.ckpt'))
        ft_model_paths = [str(model_path) for model_path in ft_model_paths] # if 'last' in str(model_path)
        # print(ft_model_paths)
        for path in ft_model_paths:
            print(path)
            checkpoint = torch.load(path)
            model_weights = checkpoint['state_dict']
            model.load_state_dict(model_weights)

            trainer.validate(model=model, datamodule=dm)
            trainer.test(model=model, datamodule=dm)
            # print(cfg.Model)
            # ft_model = ModelInterface(**ModelInterface_dict)
            # ft_model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
            # trainer.val(model=ft_model, datamodule=dm)
        # path = '/home/ylan/workspace/TransMIL-DeepGraft/logs/DeepGraft/TransMIL/norm_rest/_features_CrossEntropyLoss/lightning_logs/version_745/ft_epoch_27/checkpoints/last.ckpt'
        # path = '/home/ylan/workspace/TransMIL-DeepGraft/logs/DeepGraft/TransMIL/norm_rest/_features_CrossEntropyLoss/lightning_logs/version_745/checkpoints/last.ckpt'
        # path = '/home/ylan/workspace/TransMIL-DeepGraft/logs/DeepGraft/TransMIL/norm_rest/_features_CrossEntropyLoss/lightning_logs/version_745/ft_epoch_27/checkpoints/last.ckpt'
        
        # pprint.pprint(load_dict)
        # print(model)
        # for k in load_dict.keys():
            # print(k)
            # p
            # print(load_dict[k].keys())
        # model.load_state_dict(torch.load(path))
        # model = model.load_from_checkpoint(checkpoint_path=path, cfg=cfg)
        

        # for key in list(model_weights):
        #     model_weights[key.replace('model.', '')] = model_weights.pop(key)
        
        
        # if cfg.General.server == 'val':
        #     trainer.validate(model=model, datamodule=dm)
        # elif cfg.General.server == 'test':
        #     trainer.test(model=model, datamodule=dm)


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
    cfg.fine_tune = args.fine_tune
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
    cfg.log_name = log_name

    cfg.epoch = args.epoch
    
    print(cfg)
    # ---->main



    
    main(cfg)
 