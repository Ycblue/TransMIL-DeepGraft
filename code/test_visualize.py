import argparse
from pathlib import Path
import numpy as np
import glob
import re

from sklearn.model_selection import KFold
from scipy.interpolate import griddata

# from datasets.data_interface import DataInterface, MILDataModule, CrossVal_MILDataModule
from datasets import JPGMILDataloader, MILDataModule
from models.model_interface import ModelInterface
import models.vision_transformer as vits
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch

from pytorch_grad_cam import GradCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import cv2
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='test', type=str)
    parser.add_argument('--config', default='DeepGraft/TransMIL.yaml',type=str)
    parser.add_argument('--version', default=0,type=int)
    parser.add_argument('--epoch', default='0',type=str)
    parser.add_argument('--gpus', default = 2, type=int)
    parser.add_argument('--loss', default = 'CrossEntropyLoss', type=str)
    parser.add_argument('--fold', default = 0)
    parser.add_argument('--bag_size', default = 1024, type=int)

    args = parser.parse_args()
    return args

class custom_test_module(ModelInterface):

    def test_step(self, batch, batch_idx):

        torch.set_grad_enabled(True)
        input_data, label, (wsi_name, batch_names) = batch
        wsi_name = wsi_name[0]
        label = label.float()
        # logits, Y_prob, Y_hat = self.step(data) 
        # print(data.shape)
        input_data = input_data.squeeze(0).float()
        logits, attn = self(input_data)
        attn = attn.detach()
        logits = logits.detach()

        Y = torch.argmax(label)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        
        #----> Get GradCam maps, map each instance to attention value, assemble, overlay on original WSI 
        if self.model_name == 'TransMIL':
            target_layers = [self.model.layer2.norm] # 32x32
            # target_layers = [self.model_ft[0].features[-1]] # 32x32
            self.cam = GradCAM(model=self.model, target_layers = target_layers, use_cuda=True, reshape_transform=self.reshape_transform) #, reshape_transform=self.reshape_transform
            # self.cam_ft = GradCAM(model=self.model, target_layers = target_layers_ft, use_cuda=True) #, reshape_transform=self.reshape_transform
        else:
            target_layers = [self.model.attention_weights]
            self.cam = GradCAM(model = self.model, target_layers = target_layers, use_cuda=True)

        data_ft = self.model_ft(input_data).unsqueeze(0).float()
        instance_count = input_data.size(0)
        target = [ClassifierOutputTarget(Y)]
        grayscale_cam = self.cam(input_tensor=data_ft, targets=target)
        grayscale_cam = torch.Tensor(grayscale_cam)[:instance_count, :] #.to(self.device)

        #----------------------------------------------------
        # Get Topk Tiles and Topk Patients
        #----------------------------------------------------
        summed = torch.mean(grayscale_cam, dim=2)
        topk_tiles, topk_indices = torch.topk(summed.squeeze(0), 5, dim=0)
        topk_data = input_data[topk_indices].detach()
        
        #----------------------------------------------------
        # Log Correct/Count
        #----------------------------------------------------
        Y = torch.argmax(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        #----------------------------------------------------
        # Tile Level Attention Maps
        #----------------------------------------------------

        self.save_attention_map(wsi_name, input_data, batch_names, grayscale_cam, target=Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : Y, 'name': wsi_name, 'topk_data': topk_data} #
        # return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label, 'name': name} #, 'topk_data': topk_data

    def test_epoch_end(self, output_results):

        logits = torch.cat([x['logits'] for x in output_results], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in output_results])
        max_probs = torch.stack([x['Y_hat'] for x in output_results])
        # target = torch.stack([x['label'] for x in output_results], dim = 0)
        target = torch.stack([x['label'] for x in output_results])
        # target = torch.argmax(target, dim=1)
        patients = [x['name'] for x in output_results]
        topk_tiles = [x['topk_data'] for x in output_results]
        #---->

        auc = self.AUROC(probs, target)
        metrics = self.test_metrics(logits , target)


        # metrics = self.test_metrics(max_probs.squeeze() , torch.argmax(target.squeeze(), dim=1))
        metrics['test_auc'] = auc

        # self.log('auc', auc, prog_bar=True, on_epoch=True, logger=True)

        #---->get highest scoring patients for each class
        # test_path = Path(self.save_path) / 'most_predictive' 
        
        # Path.mkdir(output_path, exist_ok=True)
        topk, topk_indices = torch.topk(probs.squeeze(0), 5, dim=0)
        for n in range(self.n_classes):
            print('class: ', n)
            
            topk_patients = [patients[i[n]] for i in topk_indices]
            topk_patient_tiles = [topk_tiles[i[n]] for i in topk_indices]
            for x, p, t in zip(topk, topk_patients, topk_patient_tiles):
                print(p, x[n])
                patient = p
                # outpath = test_path / str(n) / patient 
                outpath = Path(self.save_path) / str(n) / patient
                outpath.mkdir(parents=True, exist_ok=True)
                for i in range(len(t)):
                    tile = t[i]
                    tile = tile.cpu().numpy().transpose(1,2,0)
                    tile = (tile - tile.min())/ (tile.max() - tile.min()) * 255
                    tile = tile.astype(np.uint8)
                    img = Image.fromarray(tile)
                    
                    img.save(f'{outpath}/{i}.jpg')

        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            metrics[keys] = values.cpu().numpy()
        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]


        # self.log_roc_curve(probs, target, 'test')
        self.log_confusion_matrix(max_probs, target, stage='test')
        #---->
        result = pd.DataFrame([metrics])
        result.to_csv(Path(self.save_path) / f'test_result.csv', mode='a', header=not Path(self.save_path).exists())

    def save_attention_map(self, wsi_name, data, batch_names, grayscale_cam, target):

        def get_coords(batch_names): #ToDO: Change function for precise coords
            coords = []
            
            for tile_name in batch_names: 
                pos = re.findall(r'\((.*?)\)', tile_name[0])
                x, y = pos[0].split('_')
                coords.append((int(x),int(y)))
            return coords
        
        coords = get_coords(batch_names)
        # temp_data = data.cpu()
        # print(data.shape)
        wsi = self.assemble(data, coords).cpu().numpy()
        # wsi = (wsi-wsi.min())/(wsi.max()-wsi.min())
        # wsi = wsi

        #--> Get interpolated mask from GradCam
        W, H = wsi.shape[0], wsi.shape[1]
        
        
        attention_map = grayscale_cam[:, :, 1].squeeze()
        attention_map = F.relu(attention_map)
        # print(attention_map)
        input_h = 256
        
        mask = torch.ones(( int(W/input_h), int(H/input_h))).to(self.device)

        for i, (x,y) in enumerate(coords):
            mask[y][x] = attention_map[i]
        mask = mask.unsqueeze(0).unsqueeze(0)
        # mask = torch.stack([mask, mask, mask]).unsqueeze(0)

        mask = F.interpolate(mask, (W,H), mode='bilinear')
        mask = mask.squeeze(0).permute(1,2,0)

        mask = (mask - mask.min())/(mask.max()-mask.min())
        mask = mask.cpu().numpy()
        
        def show_cam_on_image(img, mask):
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap*0.4 + np.float32(img)
            cam = cam / np.max(cam)
            return cam

        wsi_cam = show_cam_on_image(wsi, mask)
        wsi_cam = ((wsi_cam-wsi_cam.min())/(wsi_cam.max()-wsi_cam.min()) * 255.0).astype(np.uint8)
        
        img = Image.fromarray(wsi_cam)
        img = img.convert('RGB')
        output_path = self.save_path / str(target.item())
        output_path.mkdir(parents=True, exist_ok=True)
        img.save(f'{output_path}/{wsi_name}_gradcam.jpg')

        wsi = ((wsi-wsi.min())/(wsi.max()-wsi.min()) * 255.0).astype(np.uint8)
        img = Image.fromarray(wsi)
        img = img.convert('RGB')
        output_path = self.save_path / str(target.item())
        output_path.mkdir(parents=True, exist_ok=True)
        img.save(f'{output_path}/{wsi_name}.jpg')


    def assemble(self, tiles, coords): # with coordinates (x-y)
        
        def getPosition(img_name):
            pos = re.findall(r'\((.*?)\)', img_name) #get strings in brackets (0-0)
            a = int(pos[0].split('-')[0])
            b = int(pos[0].split('-')[1])
            return a, b

        position_dict = {}
        assembled = []
        # for tile in self.predictions:
        count = 0
        # max_x = max(coords, key = lambda t: t[0])[0]
        d = tiles[0,:,:,:].permute(1,2,0).shape
        print(d)
        white_value = 0
        x_max = max([x[0] for x in coords])
        y_max = max([x[1] for x in coords])

        for i, (x,y) in enumerate(coords):

            # name = n[0]
            # image = tiles[i,:,:,:].permute(1,2,0)
            
            # d = image.shape
            # print(image.min())
            # print(image.max())
            # if image.max() > white_value:
            #     white_value = image.max()
            # # print(image.shape)
            
            # tile_position = '-'.join(name.split('_')[-2:])
            # x,y = getPosition(tile_position)
            
            # y_max = y if y > y_max else y_max
            if x not in position_dict.keys():
                position_dict[x] = [(y, i)]
            else: position_dict[x].append((y, i))
            # count += 1
        print(position_dict.keys())
        x_positions = sorted(position_dict.keys())
        # print(x_max)
        # complete_image = torch.zeros([x_max, y_max, 3])


        for i in range(x_max+1):

            # if i in position_dict.keys():
            #     print(i)
            column = [None]*(int(y_max+1))
            # if len(d) == 3:
            # empty_tile = torch.zeros(d).to(self.device)
            # else:
            # empty_tile = torch.ones(d)
            empty_tile = torch.ones(d).to(self.device)
            # print(i)
            if i in position_dict.keys():
                # print(i)
                for j in position_dict[i]:
                    print(j)
                    sample_idx = j[1]
                    print(sample_idx)
                    # img = tiles[sample_idx, :, :, :].permute(1,2,0)
                    column[int(j[0])] = tiles[sample_idx, :, :, :]
            column = [empty_tile if i is None else i for i in column]
            print(column)
            # for c in column:
            #     print(c.shape)
            # column = torch.vstack(column)
            # print(column)
            column = torch.stack(column)
            assembled.append((i, column))
        
        assembled = sorted(assembled, key=lambda x: x[0])

        stack = [i[1] for i in assembled]
        # print(stack)
        img_compl = torch.hstack(stack)
        print(img_compl)
        return img_compl


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
    cfg.Data.label_file = '/home/ylan/DeepGraft/training_tables/split_PAS_tcmr_viral.json'
    cfg.Data.data_dir = '/home/ylan/data/DeepGraft/224_128um/'
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
    # model = ModelInterface(**ModelInterface_dict)
    model = custom_test_module(**ModelInterface_dict)
    # model.save_path = cfg.log_path
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
    log_path = Path(cfg.log_path) / 'checkpoints'
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
        new_model.save_path = Path(cfg.log_path) / 'visualization'
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

    config_path = '/'.join(Path(cfg.config).parts[1:])
    log_path = Path(cfg.General.log_path) / str(Path(config_path).parent)

    Path(cfg.General.log_path).mkdir(exist_ok=True, parents=True)
    log_name =  f'_{cfg.Model.backbone}' + f'_{cfg.Loss.base_loss}'
    task = '_'.join(Path(cfg.config).name[:-5].split('_')[2:])
    # task = Path(cfg.config).name[:-5].split('_')[2:][0]
    cfg.log_path = log_path / f'{cfg.Model.name}' / task / log_name / 'lightning_logs' / f'version_{cfg.version}' 
    
    

    #---->main
    main(cfg)
 