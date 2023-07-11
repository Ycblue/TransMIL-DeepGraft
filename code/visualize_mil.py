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

from torchmetrics.functional.classification import binary_auroc, multiclass_auroc, binary_precision_recall_curve, multiclass_precision_recall_curve



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

    def __init__(self, checkpoint_path, task):
        super().__init__()

        home = Path.cwd().parts[1]

        self.jpg_dir = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated/Aachen_Biopsy_Slides_extended/TEST'
        self.roi_dir = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated/Aachen_Biopsy_Slides_extended/ROI'
        self.save_path = Path(f'/{home}/ylan/workspace/TransMIL-DeepGraft/test/mil_model_test_2/')
        

        self.checkpoint = torch.load(checkpoint_path)
        self.hparams = self.checkpoint['hyper_parameters']
        self.n_classes = self.hparams['model']['n_classes']
        self.model_name = self.hparams['model']['name']
        self.in_features = self.hparams['model']['in_features']

        self.output_path = self.save_path / self.model_name / task
        self.label_path = self.hparams['data']['label_file']
        # print(Path(self.label_path).parts[3])
        if Path(self.label_path).parts[3] != 'data':
            new_path = Path('/' + '/'.join(Path(self.label_path).parts[1:3])) / 'data' / Path('/'.join(Path(self.label_path).parts[3:]))
            a, b = str(new_path).split('.')
            self.label_path = new_path
        # else: self.label_path = 
            # print(Path(new_path))
            # Path(home).joinpath(*Path(x).parts[2:])
            # cfg.General.log_path = '/' + str(new_path)
        

        # Add '_extended' to label file
        a, b = str(self.label_path).split('.')
        # self.label_path = a + '_ext.' + b
        if a.rsplit('_', 1)[1] != 'ext':
            self.label_path = a + '_ext.' + b
        if task == 'norm_rej_rest':
            self.label_path = '/home/ylan/data/DeepGraft/training_tables/dg_split_PAS_HE_Jones_Grocott_norm_rej_rest_ext.json'
        print(self.label_path)
        # self.label_path = '/home/ylan/data/DeepGraft/training_tables/dg_split_PAS_HE_Jones_Grocott_norm_rej_rest_ext.json'

        # self.label_path = new_path
        self.data_root = self.hparams['data']['data_dir']
        print(self.data_root)

        self.mil_model = None
        self.feat_model = None
        self.cam = None
        self.topk_dict = {}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # self._load_model_from_checkpoint(checkpoint)
        # self._load_feature_model()
        


    def _load_model_from_checkpoint(self, checkpoint):

        if self.model_name == 'TransMIL':
            mil_model = TransMIL(n_classes=self.n_classes, in_features=self.in_features)

        model_weights = checkpoint['state_dict']

        for key in list(model_weights):
            model_weights[key.replace('model.', '')] = model_weights.pop(key)
        
        mil_model.load_state_dict(model_weights)
        return mil_model
        # self.mil_model =     mil_model
    
    def _load_feature_model(self, model_name='RetCCL'):

        if model_name == 'RetCCL':
            feature_model = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
            feature_model.load_state_dict(torch.load(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
            feature_model.fc = torch.nn.Identity()

        feature_model.to(self.device)
        feature_model.eval()

        return feature_model
        # self.feature_model = feature_model

    def _reshape_transform(self, tensor):
        H = tensor.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        tensor = torch.cat([tensor, tensor[:,:add_length,:]],dim = 1)
        result = tensor[:, :, :].reshape(tensor.size(0), _H, _W, tensor.size(2))
        result = result.transpose(2,3).transpose(1,2)
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

    def _get_cam_object(self, model_name, model):
        if model_name == 'TransMIL':
            target_layers = [model.norm]
            self.cam = GradCAM(model=model, target_layers = target_layers, use_cuda=True, reshape_transform=self._reshape_transform)
            # target_layers = [model.layer4[-1]]
            # self.cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    def _get_topk(self):

        for n in range(self.n_classes):
            tpk_csv_path = Path(cfg.log_path) / f'test_epoch_{cfg.epoch}' / f'test_c{n}_top_patients.csv'
            tpk_df = pd.read_csv(tpk_csv_path)
            self.topk_dict[str(n)] = {'patients': list(tpk_df.head(5)['Patient']), 'labels': [n] * len(list(tpk_df.head(5)['Patient']))}

    def _save_attention_map(self, wsi_name, batch_coords, mil_grayscale_cam, input_h=224):

        position_dict, coords = self._get_position_dict(batch_coords)
        x_max = max([x[0] for x in coords])
        y_max = max([x[1] for x in coords])
        wsi = torch.ones([(y_max+1)*224, (x_max+1)*224, 3])
        roi = np.zeros([(y_max+1)*224, (x_max+1)*224])
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
                    wsi[y_coord*224:(y_coord+1)*224, x_coord*224:(x_coord+1)*224, :] = img

                    roi_path =  Path(self.roi_dir) / wsi_name / f'{wsi_name}_({co[0]}-{co[1]}).png'
                    img = np.asarray(Image.open(roi_path)).astype(np.uint8)
                    img = img / 255.0
                    # img = torch.from_numpy(img)
                    roi[y_coord*224:(y_coord+1)*224, x_coord*224:(x_coord+1)*224] = img
        W, H = wsi.shape[0], wsi.shape[1]
        #----------------------------------------------
        # Get mask from gradcam
        #----------------------------------------------
        # mil_attention_map = mil_grayscale_cam[:, :, 1].squeeze()
        mil_attention_map = mil_grayscale_cam.squeeze()
        mil_attention_map = (mil_attention_map-mil_attention_map.min()) / (mil_attention_map.max() - mil_attention_map.min()) #normalize to 0:1
        mask = torch.zeros(( int(W/input_h), int(H/input_h)))
        
        for i, (x,y) in enumerate(coords):
            mask[y][x] = mil_attention_map[i]
        mask = mask.unsqueeze(0).unsqueeze(0)

        mask = F.interpolate(mask, (W,H), mode='bilinear')
        mask = mask.squeeze(0).permute(1,2,0)

        # mask = (mask - mask.min())/(mask.max()-mask.min()) # normalize again..?
        mask = mask.numpy(force=True)
        # mask = gaussian_filter(mask, sigma=15)

        wsi_cam = show_cam_on_image(wsi.numpy(), mask, use_rgb=True, image_weight=0.6, colormap=cv2.COLORMAP_JET)

        #----------------------------------------------
        # Use ROI to filter image
        #----------------------------------------------
        roi_idx = (roi==0)
        wsi_cam[roi_idx] = 255
        
        size = (30000, 30000)

        print('Save GradCAM overlay.')
        img = Image.fromarray(wsi_cam)
        img = img.convert('RGB')
        img.save(f'{self.output_path}/{wsi_name}_gradcam.jpg')


    def run(self, target_label):

        patient_slide_dict_path = f'/{home}/ylan/data/DeepGraft/training_tables/patient_slide_dict_ext.json'
        
        self._get_topk()
        
        feature_model = self._load_feature_model()
        mil_model = self._load_model_from_checkpoint(self.checkpoint)
        self._get_cam_object(self.model_name, mil_model)

        with open(patient_slide_dict_path, 'r') as f:
            patient_slide_dict = json.load(f)
        slides = []
        for p in self.topk_dict[str(target_label)]['patients']: 
            slides += patient_slide_dict[p]
        self.output_path = self.output_path / str(target_label)
        self.output_path.mkdir(parents=True, exist_ok=True)
        skip_slides = [x.stem.rsplit('_', 1)[0] for x in list(self.output_path.iterdir()) if Path(x).suffix == '.jpg']
        # skip_slides += ['Aachen_KiBiDatabase_KiBiAcDKIK860_01_006_HE']
        slides = [s for s in slides if s not in skip_slides]
        print(slides)
        try:
            len(slides) != 0
        except:
            print('No Slides available. Please check directory.')

        # print(slides)

        test_dataset = JPGMILDataloader(file_path=self.data_root, label_path=self.label_path, mode='test', cache=False, n_classes=self.n_classes, model=self.model_name, slides=slides)
        
        print(len(test_dataset))
        dl = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True)
        print(len(dl))

        for item in tqdm(dl):
            
            bag, label, (name, batch_coords, patient) = item
            label = torch.Tensor(label)
            # idx = top_patients.index(patient[0])

            slide_name = name[0]
            print(slide_name)
            # if slide_name != 'Aachen_KiBiDatabase_KiBiAcRLKM530_01_006_HE':
            #     continue
            # else:
            

            bag = bag.float().to(self.device)
            # with torch.cuda.amp.autocast():
            with torch.no_grad():
                features = feature_model(bag.squeeze())



            size = features.shape[0]
            # print(features.shape)

            x, attn = mil_model(features.unsqueeze(0), return_attn=True)

            # print(attn)

            # print(attn.shape)
            cls_attention = attn[:,:, 0, :size]
            # print(cls_attention)
            values, indices = torch.max(cls_attention, 1)
            mean = values.mean()
            zeros = torch.zeros(values.shape).cuda()
            filtered = torch.where(values > mean, values, zeros)

            # print(filtered.shape)



            instance_count = bag.size(0)
            bag = bag.detach()
            cam_target = [ClassifierOutputTarget(target_label)]
            
            mil_grayscale_cam = self.cam(input_tensor=features.unsqueeze(0), targets=cam_target)
            mil_grayscale_cam = torch.Tensor(mil_grayscale_cam)[:instance_count, :]
            mil_grayscale_cam = mil_grayscale_cam[:, :, 1].squeeze()

            # print(mil_grayscale_cam.shape)
            features = features.detach()
            bag = bag.detach()
            # print(target_label)
            self._save_attention_map(slide_name, batch_coords, mil_grayscale_cam)
            # self._save_attention_map(slide_name, batch_coords, filtered)

    # for t in test_dataset:

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='test', type=str)
    parser.add_argument('--config', default='../DeepGraft/TransMIL_feat_norm_rest.yaml',type=str)
    parser.add_argument('--version', default=0,type=int)
    parser.add_argument('--epoch', default='0',type=str)
    parser.add_argument('--gpus', default = 0, type=int)
    parser.add_argument('--loss', default = 'CrossEntropyLoss', type=str)
    parser.add_argument('--fold', default = 0)
    parser.add_argument('--bag_size', default = 10000, type=int)
    parser.add_argument('--total_classes', default = 2, type=int)
    parser.add_argument('--target_label', default = 1, type=int)


    args = parser.parse_args()
    return args

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
        model_paths = [str(model_path) for model_path in model_paths if f'last' in str(model_path)]
    else:
        model_paths = [str(model_path) for model_path in model_paths if f'epoch={cfg.epoch}' in str(model_path)]
        
    # for i in range(args.total_classes):

    # target_label = 1
    target_label = args.target_label
    
    # for target_label in range(args.total_classes):
    visualizer = Visualize(checkpoint_path=model_paths[0], task=cfg.task)
    visualizer.run(target_label)
