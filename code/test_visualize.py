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
import torch.nn as nn

from pytorch_grad_cam import GradCAM, EigenGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import cv2
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import json
import pprint


#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='test', type=str)
    parser.add_argument('--config', default='../DeepGraft/TransMIL.yaml',type=str)
    parser.add_argument('--version', default=0,type=int)
    parser.add_argument('--epoch', default='0',type=str)
    parser.add_argument('--gpus', default = 0, type=int)
    parser.add_argument('--loss', default = 'CrossEntropyLoss', type=str)
    parser.add_argument('--fold', default = 0)
    parser.add_argument('--bag_size', default = 10000, type=int)

    args = parser.parse_args()
    return args

class custom_test_module(ModelInterface):

    # self.task = kargs['task']    
    # self.task = 'tcmr_viral'

    def test_step(self, batch, batch_idx):

        torch.set_grad_enabled(True)

        input_data, label, (wsi_name, batch_names, patient) = batch
        patient = patient[0]
        wsi_name = wsi_name[0]
        label = label.float()
        # logits, Y_prob, Y_hat = self.step(data) 
        # print(data.shape)
        input_data = input_data.squeeze(0).float()
        # print(self.model_ft)
        # print(self.model)
        logits, _ = self(input_data)
        # attn = attn.detach()
        # logits = logits.detach()

        Y = torch.argmax(label)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        

        # print('Y_hat:', Y_hat)
        # print('Y_prob:', Y_prob)

        
        #----> Get GradCam maps, map each instance to attention value, assemble, overlay on original WSI 
        if self.model_name == 'TransMIL':
            target_layers = [self.model.layer2.norm] # 32x32
            # target_layers = [self.model_ft[0].features[-1]] # 32x32
            self.cam = GradCAM(model=self.model, target_layers = target_layers, use_cuda=True, reshape_transform=self.reshape_transform) #, reshape_transform=self.reshape_transform
            # self.cam_ft = GradCAM(model=self.model, target_layers = target_layers_ft, use_cuda=True) #, reshape_transform=self.reshape_transform
        elif self.model_name == 'TransformerMIL':
            target_layers = [self.model.layer1.norm]
            self.cam = EigenCAM(model=self.model, target_layers = target_layers, use_cuda=True, reshape_transform=self.reshape_transform)
            # self.cam = GradCAM(model=self.model, target_layers = target_layers, use_cuda=True, reshape_transform=self.reshape_transform)
        else:
            target_layers = [self.model.attention_weights]
            self.cam = GradCAM(model = self.model, target_layers = target_layers, use_cuda=True)

        if self.model_ft:
            data_ft = self.model_ft(input_data).unsqueeze(0).float()
        else:
            data_ft = input_data.unsqueeze(0).float()
        instance_count = input_data.size(0)
        # data_ft.requires_grad=True
        
        target = [ClassifierOutputTarget(Y)]
        # print(target)
        
        grayscale_cam = self.cam(input_tensor=data_ft, targets=target, eigen_smooth=True)
        grayscale_cam = torch.Tensor(grayscale_cam)[:instance_count, :] #.to(self.device)

        #----------------------------------------------------
        # Get Topk Tiles and Topk Patients
        #----------------------------------------------------
        k = 10
        summed = torch.mean(grayscale_cam, dim=2)
        topk_tiles, topk_indices = torch.topk(summed.squeeze(0), k, dim=0)
        topk_data = input_data[topk_indices].detach()
        # print(topk_tiles)
        
        #----------------------------------------------------
        # Log Correct/Count
        #----------------------------------------------------
        Y = torch.argmax(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        #----------------------------------------------------
        # Tile Level Attention Maps
        #----------------------------------------------------

        # print(input_data.shape)
        # print(len(batch_names))
        # if visualize:
        # self.save_attention_map(wsi_name, batch_names, grayscale_cam, target=Y)
        # print('test_step_patient: ', patient)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : Y, 'name': wsi_name, 'patient': patient, 'topk_data': topk_data} #
        # return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label, 'name': name} #, 'topk_data': topk_data

    def test_epoch_end(self, output_results):

        k_patient = 1
        k_slide = 1

        pp = pprint.PrettyPrinter(indent=4)

        logits = torch.cat([x['logits'] for x in output_results], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in output_results])
        max_probs = torch.stack([x['Y_hat'] for x in output_results])
        # target = torch.stack([x['label'] for x in output_results], dim = 0)
        target = torch.stack([x['label'] for x in output_results])
        # target = torch.argmax(target, dim=1)
        slide = [x['name'] for x in output_results]
        patients = [x['patient'] for x in output_results]
        topk_tiles = [x['topk_data'] for x in output_results]
        #---->

        if len(target.unique()) !=1:
            auc = self.AUROC(probs, target)
        else: auc = torch.tensor(0)
        metrics = self.test_metrics(logits , target)


        # metrics = self.test_metrics(max_probs.squeeze() , torch.argmax(target.squeeze(), dim=1))
        metrics['test_auc'] = auc

        # print(metrics)
        np_metrics = {k: metrics[k].item() for k in metrics.keys()}
        # print(np_metrics)

        
        complete_patient_dict = {}
        '''
        Patient
        -> slides:
            -> SlideName:
                ->probs = [0.5, 0.5] 
                ->topk = [10,3,224,224]
        -> score = []
        '''


        for p, s, l, topkt in zip(patients, slide, probs, topk_tiles):
            if p not in complete_patient_dict.keys():
                complete_patient_dict[p] = {'slides':{}}
            complete_patient_dict[p]['slides'][s] = {'probs': l, 'topk':topkt}

        patient_list = []            
        patient_score = []            
        for p in complete_patient_dict.keys():
            score = []
            
            for s in complete_patient_dict[p]['slides'].keys():
                score.append(complete_patient_dict[p]['slides'][s]['probs'])
            score = torch.mean(torch.stack(score), dim=0) #.cpu().detach().numpy()
            complete_patient_dict[p]['score'] = score
            # print(p, score)
            patient_list.append(p)    
            patient_score.append(score)    

        # print(patient_list)
        #topk patients: 


        # task = 'tcmr_viral'
        task = Path(self.save_path).parts[-5]
        label_map_path = 'label_map.json'
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
        
        # topk_patients, topk_p_indices = torch.topk(score, 5, dim=0)

        # print(probs.squeeze(0))
        # topk, topk_indices = torch.topk(probs.squeeze(0), 5, dim=0) # topk = 
        # print(topk)
        
        # topk_indices = topk_indices.transpose(0, 1)

        output_dict = {}
    

        for n in range(self.n_classes):

            class_name = f'{n}_{label_map[task][str(n)]}'

            output_dict[class_name] = {}
            # class_name = str(n)
            print('class: ', class_name)
            # print(score)
            _, topk_indices = torch.topk(score, k_patient, dim=0) # change to 3
            # print(topk_indices)

            topk_patients = [patient_list[i] for i in topk_indices]

            patient_top_slides = {} 
            for p in topk_patients:
                # print(p)
                output_dict[class_name][p] = {}
                output_dict[class_name][p]['Patient_Score'] = complete_patient_dict[p]['score'].cpu().detach().numpy().tolist()

                slides = list(complete_patient_dict[p]['slides'].keys())
                slide_scores = [complete_patient_dict[p]['slides'][s]['probs'] for s in slides]
                slide_scores = torch.stack(slide_scores)
                # print(slide_scores)
                _, topk_slide_indices = torch.topk(slide_scores, k_slide, dim=0)
                # topk_slide_indices = topk_slide_indices.squeeze(0)
                # print(topk_slide_indices[0])
                topk_patient_slides = [slides[i] for i in topk_slide_indices[0]]
                patient_top_slides[p] = topk_patient_slides

                output_dict[class_name][p]['Top_Slides'] = [{slides[i]: {'Slide_Score': slide_scores[i].cpu().detach().numpy().tolist()}} for i in topk_slide_indices[0]]

            for p in topk_patients: 

                score = complete_patient_dict[p]['score']
                # print(p, score)
                print('Topk Slides:')
                for slide in patient_top_slides[p]:
                    print(slide)
                    outpath = Path(self.save_path) / class_name / p / slide
                    outpath.mkdir(parents=True, exist_ok=True)
                
                    topk_tiles = complete_patient_dict[p]['slides'][slide]['topk']
                    # for i in range(topk_tiles.shape[0]):
                    #     tile = topk_tiles[i]
                    #     tile = tile.cpu().numpy().transpose(1,2,0)
                    #     tile = (tile - tile.min())/ (tile.max() - tile.min()) * 255
                    #     tile = tile.astype(np.uint8)
                    #     img = Image.fromarray(tile)
                    
                    #     img.save(f'{outpath}/{i}.jpg')
        output_dict['Test_Metrics'] = np_metrics
        pp.pprint(output_dict)
        json.dump(output_dict, open(f'{self.save_path}/test_metrics.json', 'w'))

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

    def save_attention_map(self, wsi_name, batch_names, grayscale_cam, target):

        # def get_coords(batch_names): #ToDO: Change function for precise coords
        #     coords = []
            
        #     for tile_name in batch_names: 
        #         pos = re.findall(r'\((.*?)\)', tile_name[0])
        #         x, y = pos[-1].split('_')
        #         coords.append((int(x),int(y)))
        #     return coords

        home = Path.cwd().parts[1]
        jpg_dir = f'/{home}/ylan/data/DeepGraft/224_128um_annotated/Aachen_Biopsy_Slides/BLOCKS'

        coords = batch_names.squeeze()
        data = []
        for co in coords:

            tile_path =  Path(jpg_dir) / wsi_name / f'{wsi_name}_({co[0]}_{co[1]}).jpg'
            img = np.asarray(Image.open(tile_path)).astype(np.uint8)
            img = torch.from_numpy(img)
            # print(img.shape)
            data.append(img)
        # coords_set = set(coords)
        # data = data.unsqueeze(0)
        # print(data.shape)
        data = torch.stack(data)
        # print(data.max())
        # print(data.min())
        # print(coords)
        # temp_data = data.cpu()
        # print(data.shape)
        wsi = self.assemble(data, coords).cpu().numpy()
        # wsi = (wsi-wsi.min())/(wsi.max()-wsi.min())
        # wsi = wsi
        # print(coords)
        # print('wsi.shape: ', wsi.shape)
        #--> Get interpolated mask from GradCam
        W, H = wsi.shape[0], wsi.shape[1]
        
        
        attention_map = grayscale_cam[:, :, 1].squeeze()
        attention_map = F.relu(attention_map)
        # print(attention_map)
        input_h = 224
        
        mask = torch.ones(( int(W/input_h), int(H/input_h))).to(self.device)
        # print('mask.shape: ', mask.shape)
        # print('attention_map.shape: ', attention_map.shape)
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
        
        size = (20000, 20000)

        # img = Image.fromarray(wsi_cam)
        # img = img.convert('RGB')
        # img.thumbnail(size, Image.ANTIALIAS)
        # output_path = self.save_path / str(target.item())
        # output_path.mkdir(parents=True, exist_ok=True)
        # img.save(f'{output_path}/{wsi_name}_gradcam.jpg')

        wsi = ((wsi-wsi.min())/(wsi.max()-wsi.min()) * 255.0).astype(np.uint8)
        img = Image.fromarray(wsi)
        img = img.convert('RGB')
        img.thumbnail(size, Image.ANTIALIAS)
        output_path = self.save_path / str(target.item())
        output_path.mkdir(parents=True, exist_ok=True)
        img.save(f'{output_path}/{wsi_name}.jpg')
        del wsi
        del img
        del wsi_cam
        del mask


    def assemble(self, tiles, coords): # with coordinates (x-y)
        
        # def getPosition(img_name):
        #     pos = re.findall(r'\((.*?)\)', img_name) #get strings in brackets (0-0)
        #     a = int(pos[0].split('-')[0])
        #     b = int(pos[0].split('-')[1])
        #     return a, b

        position_dict = {}
        assembled = []
        # for tile in self.predictions:
        count = 0
        # max_x = max(coords, key = lambda t: t[0])[0]
        d = tiles[0,:,:,:].permute(1,2,0).shape
        # print(d)
        white_value = 0
        x_max = max([x[0] for x in coords])
        y_max = max([x[1] for x in coords])

        for i, (x,y) in enumerate(coords):
            if x not in position_dict.keys():
                position_dict[x.item()] = [(y.item(), i)]
            else: position_dict[x.item()].append((y.item(), i))
        # x_positions = sorted(position_dict.keys())

        test_img_compl = torch.ones([(y_max+1)*224, (x_max+1)*224, 3]).to(self.device)

        for i in range(x_max+1):
            if i in position_dict.keys():
                for j in position_dict[i]:
                    sample_idx = j[1]
                    # if tiles[sample_idx, :, :, :].shape != [3,224,224]:
                    #     img = tiles[sample_idx, :, :, :].permute(2,0,1)
                    # else: 
                    img = tiles[sample_idx, :, :, :]
                    # print(img.shape)
                    # print(img.max())
                    # print(img.min())
                    y_coord = int(j[0])
                    x_coord = int(i)
                    test_img_compl[y_coord*224:(y_coord+1)*224, x_coord*224:(x_coord+1)*224, :] = img



        # for i in range(x_max+1):
        #     column = [None]*(int(y_max+1))
        #     empty_tile = torch.ones(d).to(self.device)
        #     if i in position_dict.keys():
        #         for j in position_dict[i]:
        #             sample_idx = j[1]
        #             if tiles[sample_idx, :, :, :].shape != [3,224,224]:
        #                 img = tiles[sample_idx, :, :, :].permute(1,2,0)
        #             else: 
        #                 img = tiles[sample_idx, :, :, :]
        #             column[int(j[0])] = img
        #     column = [empty_tile if i is None else i for i in column]
        #     column = torch.vstack(column)
        #     assembled.append((i, column))
        
        # assembled = sorted(assembled, key=lambda x: x[0])

        # stack = [i[1] for i in assembled]
        # # print(stack)
        # img_compl = torch.hstack(stack)
        # print(img_compl.shape)
        # print(test_img_compl)
        # print(torch.nonzero(img_compl - test_img_compl))
        # print(img_compl)
        return test_img_compl.cpu().detach()


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
    # cfg.Data.label_file = '/home/ylan/DeepGraft/training_tables/split_PAS_tcmr_viral_Utrecht.json'
    # cfg.Data.label_file = '/homeStor1/ylan/DeepGraft/training_tables/split_debug.json'
    # cfg.Data.label_file = '/home/ylan/DeepGraft/training_tables/dg_limit_20_split_PAS_HE_Jones_norm_rest.json'
    # cfg.Data.patient_slide = '/homeStor1/ylan/DeepGraft/training_tables/cohort_stain_dict.json'
    # cfg.Data.data_dir = '/homeStor1/ylan/data/DeepGraft/224_128um_v2/'
    if cfg.Model.backbone == 'features':
        use_features = True
    else: use_features = False
    DataInterface_dict = {
                'data_root': cfg.Data.data_dir,
                'label_path': cfg.Data.label_file,
                'batch_size': cfg.Data.train_dataloader.batch_size,
                'num_workers': cfg.Data.train_dataloader.num_workers,
                'n_classes': cfg.Model.n_classes,
                'backbone': cfg.Model.backbone,
                'bag_size': cfg.Data.bag_size,
                'use_features': use_features,
                }

    dm = MILDataModule(**DataInterface_dict)
    

    #---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path,
                            'backbone': cfg.Model.backbone,
                            'task': cfg.task,
                            }
    # model = ModelInterface(**ModelInterface_dict)
    model = custom_test_module(**ModelInterface_dict)
    # model._fc1 = nn.Sequential(nn.Linear(512, 512), nn.GELU())
    # model.save_path = cfg.log_path
    #---->Instantiate Trainer
    
    tb_logger = pl_loggers.TensorBoardLogger(cfg.log_path)

    trainer = Trainer(
        num_sanity_val_steps=0, 
        logger=tb_logger,
        # callbacks=cfg.callbacks,
        max_epochs= cfg.General.epochs,
        min_epochs = 200,
        accelerator='gpu',
        devices=cfg.General.gpus,
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

    # print(model_paths)
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
    cfg.General.gpus = [args.gpus]
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold
    cfg.Loss.base_loss = args.loss
    cfg.Data.bag_size = args.bag_size
    cfg.version = args.version
    cfg.epoch = args.epoch

    cfg = check_home(cfg)

    config_path = '/'.join(Path(cfg.config).parts[1:])
    log_path = Path(cfg.General.log_path) / str(Path(config_path).parent)

    Path(cfg.General.log_path).mkdir(exist_ok=True, parents=True)
    log_name =  f'_{cfg.Model.backbone}' + f'_{cfg.Loss.base_loss}'
    task = '_'.join(Path(cfg.config).name[:-5].split('_')[2:])
    # task = Path(cfg.config).name[:-5].split('_')[2:][0]
    cfg.task = task
    cfg.log_path = log_path / f'{cfg.Model.name}' / task / log_name / 'lightning_logs' / f'version_{cfg.version}' 
    
    # cfg.model_path = cfg.log_patth / 'code' / 'models' / 
    
    

    #---->main
    # main(cfg)
    from models import TransMIL
    from datasets.zarr_feature_dataloader_simple import ZarrFeatureBagLoader
    from datasets.feature_dataloader import FeatureBagLoader
    from torch.utils.data import random_split, DataLoader
    import time
    from tqdm import tqdm
    import torchmetrics

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    scaler = torch.cuda.amp.GradScaler()
    
    log_path = Path(cfg.log_path) / 'checkpoints'
    model_paths = list(log_path.glob('*.ckpt'))

    # print(model_paths)
    if cfg.epoch == 'last':
        model_paths = [str(model_path) for model_path in model_paths if f'last' in str(model_path)]
    else:
        model_paths = [str(model_path) for model_path in model_paths if f'epoch={cfg.epoch}' in str(model_path)]

    # checkpoint = torch.load(f'{cfg.log_path}/checkpoints/epoch=04-val_loss=0.4243-val_auc=0.8243-val_patient_auc=0.8282244801521301.ckpt')
    # checkpoint = torch.load(f'{cfg.log_path}/checkpoints/epoch=73-val_loss=0.8574-val_auc=0.9682-val_patient_auc=0.9724310636520386.ckpt')
    checkpoint = torch.load(model_paths[0])

    hyper_parameters = checkpoint['hyper_parameters']
    n_classes = hyper_parameters['model']['n_classes']

    # model = TransMIL()
    model = TransMIL(n_classes).to(device)
    model_weights = checkpoint['state_dict']

    for key in list(model_weights):
        model_weights[key.replace('model.', '')] = model_weights.pop(key)
    
    model.load_state_dict(model_weights)

    count = 0
    # for m in model.modules():
    #     if isinstance(m, torch.nn.BatchNorm2d):
    #         # # m.track_running_stats = False
    #         # count += 1 #skip the first BatchNorm layer in my ResNet50 based encoder
    #         # if count >= 2:
    #             # m.eval()
    #         print(m)
    #         m.track_running_stats = False
    #         m.running_mean = None
    #         m.running_var = None
    
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    home = Path.cwd().parts[1]
    data_root = f'/{home}/ylan/data/DeepGraft/224_128uM_annotated'
    label_path = f'/{home}/ylan/DeepGraft/training_tables/dg_split_PAS_HE_Jones_norm_rest.json'
    dataset = FeatureBagLoader(data_root, label_path=label_path, mode='test', cache=False, n_classes=n_classes)

    dl = DataLoader(dataset, batch_size=1, num_workers=8)

    

    AUROC = torchmetrics.AUROC(num_classes = n_classes)

    start = time.time()
    test_logits = []
    test_probs = []
    test_labels = []
    data = [{"count": 0, "correct": 0} for i in range(n_classes)]

    for item in tqdm(dl): 

        bag, label, (name, batch_coords, patient) = item
        # label = label.float()
        Y = int(label)

        bag = bag.float().to(device)
        # print(bag.shape)
        bag = bag.unsqueeze(0)
        with torch.cuda.amp.autocast():
            logits = model(bag)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)

        # print(Y_prob)

        test_logits.append(logits)
        test_probs.append(Y_prob)

        test_labels.append(label)
        data[Y]['count'] += 1
        data[Y]['correct'] += (int(Y_hat) == Y)
    probs = torch.cat(test_probs).detach().cpu()
    targets = torch.stack(test_labels).squeeze().detach().cpu()
    print(probs.shape)
    print(targets.shape)

    
    for c in range(n_classes):
        count = data[c]['count']
        correct = data[c]['correct']
        if count == 0:
            acc = None
        else: 
            acc = float(correct) / count
        print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))



    auroc = AUROC(probs, targets)
    print(auroc)
    end = time.time()
    print('Bag Time: ', end-start)



 