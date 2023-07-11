import argparse
from pathlib import Path
import numpy as np
import glob
import re

from sklearn.model_selection import KFold
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from models.model_interface import ModelInterface
import models.vision_transformer as vits
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from pytorch_grad_cam import GradCAM, EigenGradCAM, EigenCAM, XGradCAM, ScoreCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import cv2
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
import pandas as pd
import json
import pprint

from models import TransMIL
# from datasets.zarr_feature_dataloader_simple import ZarrFeatureBagLoader
from datasets.feature_dataloader import FeatureBagLoader
from datasets.jpg_dataloader import JPGMILDataloader
from torch.utils.data import random_split, DataLoader
import time
from tqdm import tqdm
import torchmetrics
import models.ResNet as ResNet
import torchvision.models as models
import timm

from torchmetrics.functional.classification import binary_auroc, multiclass_auroc, binary_precision_recall_curve, multiclass_precision_recall_curve

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
    parser.add_argument('--total_classes', default = 2, type=int)


    args = parser.parse_args()
    return args

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



class Visualize():

    def __init__(self, checkpoint_path, task, classic=False):
        super().__init__()

        home = Path.cwd().parts[1]

        self.jpg_dir = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated/Aachen_Biopsy_Slides/TEST'
        self.roi_dir = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated/Aachen_Biopsy_Slides/ROI'
        self.save_path = Path(f'/{home}/ylan/workspace/TransMIL-DeepGraft/test/classic_model/')
        self.output_path = self.save_path
        # output_path = save_path / str(target.item())
        

        checkpoint = torch.load(checkpoint_path)
        self.hparams = checkpoint['hyper_parameters']
        self.n_classes = self.hparams['model']['n_classes']
        self.model_name = self.hparams['model']['name']
        self.in_features = self.hparams['model']['in_features']

        self.output_path = self.save_path / self.model_name / task
        self.label_path = self.hparams['data']['label_file']
        self.data_root = self.hparams['data']['data_dir']

        self.model = None
        self.cam = None
        self.topk_dict = {}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self._load_model_from_checkpoint(checkpoint)
        self._get_cam_object(self.model_name)


    def _load_model_from_checkpoint(self, checkpoint):

        if self.model_name == 'resnet18':
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, self.n_classes))
        elif self.model_name == 'resnet50':
            self.model = models.resnet50(weights='IMAGENET1K_V1')
            self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, self.n_classes),)
        elif self.model_name == 'vit':
            home = Path.cwd().parts[1]
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=self.n_classes)
            # model = timm.create_model(“vit_base_patch16_224”, pretrained=True)
            outputs_attrs = self.n_classes
            num_inputs = self.model.head.in_features
            last_layer = nn.Linear(num_inputs, outputs_attrs)
            self.model.head = last_layer

        model_weights = checkpoint['state_dict']

        for key in list(model_weights):
            model_weights[key.replace('model.', '')] = model_weights.pop(key)
        
        self.model.load_state_dict(model_weights)
        # self.model.eval()
        
    

    # def _reshape_transform(self, tensor):
    #     H = tensor.shape[1]
    #     _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
    #     add_length = _H * _W - H
    #     tensor = torch.cat([tensor, tensor[:,:add_length,:]],dim = 1)
    #     result = tensor[:, :, :].reshape(tensor.size(0), _H, _W, tensor.size(2))
    #     result = result.transpose(2,3).transpose(1,2)
    #     return result
    def _reshape_transform(self, tensor, height=14, width=14):
        result = tensor[:, 1 :  , :].reshape(tensor.size(0),
            height, width, tensor.size(2))
    
        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    
    def _get_position_dict(self, batch_coords):
        coords = []
        position_dict = {}
        batch_coords = batch_coords.squeeze(0)
        for i in range(batch_coords.shape[0]): 
            c = batch_coords[i, :]
            x = c[0]
            y = c[1]
            coords.append((int(x),int(y)))

        for i, (x,y) in enumerate(coords):
            if x not in position_dict.keys():
                position_dict[x] = [(y, i)]
            else: position_dict[x].append((y, i))
        return position_dict, coords
        # return coords

    def _get_cam_object(self, model_name):
        if model_name == 'resnet18' or model_name == 'resnet50':
            target_layers = [self.model.layer4[-1]]
            self.cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=True)
        elif model_name == 'vit':
            target_layers = [self.model.blocks[-1].norm1]
            self.cam = GradCAM(model=self.model, target_layers=target_layers, use_cuda=True, reshape_transform=self._reshape_transform)

    def _get_topk(self):

        for n in range(self.n_classes):
            tpk_csv_path = Path(cfg.log_path) / f'test_epoch_{cfg.epoch}' / f'test_c{n}_top_patients.csv'
            tpk_df = pd.read_csv(tpk_csv_path)
            self.topk_dict[str(n)] = {'patients': list(tpk_df.head(5)['Patient']), 'labels': [n] * len(list(tpk_df.head(5)['Patient']))}

    def _save_attention_map(self, wsi_name, batch_coords, grayscale_cam, input_h=224):

        position_dict, coords = self._get_position_dict(batch_coords)
        x_max = max([x[0] for x in coords])
        y_max = max([x[1] for x in coords])
        wsi = torch.ones([(y_max+1)*input_h, (x_max+1)*input_h, 3])

        roi = torch.zeros([(y_max+1)*input_h, (x_max+1)*input_h])
        mask = torch.zeros(roi.shape)
        for i in range(x_max+1):
            if i in position_dict.keys():
                for j in position_dict[i]:
                    y_coord = int(j[0])
                    x_coord = int(i)
                    co = coords[j[1]]
                    tile_path =  Path(self.jpg_dir) / wsi_name / f'{wsi_name}_({co[0]}-{co[1]}).png'
                    img = np.asarray(Image.open(tile_path)).astype(np.uint8)
                    img = img / 255.0
                    img = torch.from_numpy(img)
                    wsi[y_coord*input_h:(y_coord+1)*input_h, x_coord*input_h:(x_coord+1)*input_h, :] = img

                    roi_path =  Path(self.roi_dir) / wsi_name / f'{wsi_name}_({co[0]}-{co[1]}).png'
                    img = np.asarray(Image.open(roi_path)).astype(np.uint8)
                    img = img / 255.0
                    img = torch.from_numpy(img)

                    roi[y_coord*input_h:(y_coord+1)*input_h, x_coord*input_h:(x_coord+1)*input_h] = img
        W, H = wsi.shape[0], wsi.shape[1]
        #----------------------------------------------
        # Get mask from gradcam
        #----------------------------------------------
        
        # attention_map = (attention_map-attention_map.min()) / (attention_map.max() - attention_map.min())
        print(W, H)
        if self.model_name == 'vit':
            # attention_map = torch.from_numpy(grayscale_cam)
            print(grayscale_cam.shape)
            attention_map = torch.from_numpy(grayscale_cam[0, :, :].squeeze())
            input_h = 44
            mask = torch.zeros(( int(W/input_h), int(H/input_h)))
            print(mask.shape)
            print(attention_map.shape)
            for i, (x,y) in enumerate(coords):
                mask[y][x] = attention_map[i]
            mask = mask.unsqueeze(0).unsqueeze(0)

            mask = F.interpolate(mask, (W,H), mode='bilinear')
            mask = mask.squeeze(0).permute(1,2,0)

            mask = (mask - mask.min())/(mask.max()-mask.min())
            mask = mask.numpy()
            #     wsi_cam = show_cam_on_image(wsi.numpy(), mask)
        #     wsi_cam = ((wsi_cam-wsi_cam.min())/(wsi_cam.max()-wsi_cam.min()) * 255.0).astype(np.uint8)
            
        #     size = (20000, 20000)

        #     img = Image.fromarray(wsi_cam)
        #     img = img.convert('RGB')
        #     img.thumbnail(size, Image.Resampling.LANCZOS)
        else:
            attention_map = torch.from_numpy(grayscale_cam)
            mask = torch.zeros(roi.shape)
            for i, (x,y) in enumerate(coords):
                # print(attention_map[i, :, :].shape)
                mask[y*input_h:(y+1)*input_h, x*input_h:(x+1)*input_h] = attention_map[i]
            # if attention_map[i].mean() != 0:
            #     print(i)
            mask = mask.unsqueeze(0).unsqueeze(0)

            # mask = F.interpolate(mask, (W,H), mode='bilinear')
            mask = mask.squeeze(0).permute(1,2,0)

            mask = (mask - mask.min())/(mask.max()-mask.min())
            mask = mask.numpy()
        # mask = gaussian_filter(mask, sigma=15)

        wsi_cam = show_cam_on_image(wsi.numpy(), mask, use_rgb=True, image_weight=0.6)

        #----------------------------------------------
        # Use ROI to filter image
        #----------------------------------------------
        # roi_idx = (roi==0)
        # wsi_cam[roi_idx] = 255
        
        size = (30000, 30000)

        print('Save GradCAM overlay.')
        img = Image.fromarray(wsi_cam)
        img = img.convert('RGB')
        img.save(f'{self.output_path}/{wsi_name}_gradcam.jpg')


    def run(self, target_label):

        patient_slide_dict_path = f'/{home}/ylan/data/DeepGraft/training_tables/patient_slide_dict.json'
        
        self._get_topk()
        

        with open(patient_slide_dict_path, 'r') as f:
            patient_slide_dict = json.load(f)
        slides = []
        for p in self.topk_dict[str(target_label)]['patients']: 
            slides += patient_slide_dict[p]
        self.output_path = self.output_path / str(target_label)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # print(slides)

        test_dataset = JPGMILDataloader(file_path=self.data_root, label_path=self.label_path, mode='test', cache=False, n_classes=self.n_classes, model=self.model_name, slides=slides)
        dl = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True)

        for item in tqdm(dl):
            
            bag, label, (name, batch_coords, patient) = item
            label = torch.Tensor(label)
            # idx = top_patients.index(patient[0])

            slide_name = name[0]
            # print(slide_name)

            bag = bag.to(self.device).squeeze(0)
            # for i in bag.shape[1]:

            # with torch.cuda.amp.autocast():
            #     features = self.feature_model(bag.squeeze())
            # instance_count = bag.size(0)
            # bag = bag.detach()
            cam_target = [ClassifierOutputTarget(target_label)]
            self.cam.batch_size = bag.size(0)
            grayscale_cam = self.cam(input_tensor=bag.squeeze(0), targets=cam_target) #
            # print(grayscale_cam.max())
            # print(target_label)
            self._save_attention_map(slide_name, batch_coords, grayscale_cam)

    # for t in test_dataset:


if __name__ == '__main__':

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
    
    home = Path.cwd().parts[1]

    ckpt_pth = Path(cfg.log_path) / 'checkpoints'
    model_paths = list(ckpt_pth.glob('*.ckpt'))
    
    if cfg.epoch == 'last':
        epoch = 'last'
        # model_paths = [str(model_path) for model_path in model_paths if f'last' in str(model_path)]

    elif int(cfg.epoch) < 10:
        epoch = f'epoch=0{cfg.epoch}'
    else:
        epoch = f'epoch={cfg.epoch}' 
    model_paths = [str(model_path) for model_path in model_paths if epoch in str(model_path)]
    # print(model_paths)
    # for i in range(args.total_classes):

    target_label = 1

    visualizer = Visualize(checkpoint_path=model_paths[0], task=cfg.task)
    visualizer.run(target_label)
