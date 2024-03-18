import argparse
from pathlib import Path
import numpy as np
import glob
import re

# from sklearn.model_selection import KFold
# from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
# from models.model_interface import ModelInterface
# import models.vision_transformer as vits
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from pytorch_grad_cam import GradCAM, EigenGradCAM, EigenCAM, XGradCAM, ScoreCAM, GradCAMPlusPlus, HiResCAM
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

import shap



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

        # self.jpg_dir = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated/DEEPGRAFT_RU/BLOCKS'
        # self.roi_dir = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated/DEEPGRAFT_RU/ROI'
        # self.save_path = Path(f'/{home}/ylan/workspace/TransMIL-DeepGraft/test/mil_model_features/')
        self.jpg_dir = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated/Aachen_Biopsy_Slides_extended/BLOCKS'
        self.roi_dir = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated/Aachen_Biopsy_Slides_extended/ROI'
        self.save_path = Path(f'/{home}/ylan/workspace/TransMIL-DeepGraft/test/debug/')
        # self.save_path = Path(f'/{home}/ylan/workspace/TransMIL-DeepGraft/test/results_test/mil_model_features/')
        

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
            # new_path = a + '_ext.' + b
            self.label_path = new_path
        # else: self.label_path = 
            # print(Path(new_path))
            # Path(home).joinpath(*Path(x).parts[2:])
            # cfg.General.log_path = '/' + str(new_path)
        

        # Add '_extended' to label file
        a, b = str(self.label_path).split('.')
        if a.rsplit('_', 1)[1] != 'ext':

            self.label_path = a + '_ext.' + b
        # print(self.label_path)
        if task == 'norm_rej_rest':
            self.label_path = '/home/ylan/data/DeepGraft/training_tables/dg_split_PAS_HE_Jones_Grocott_norm_rej_rest_ext.json'

        # self.label_path = new_path
        self.data_root = self.hparams['data']['data_dir']

        self.mil_model = None
        self.feat_model = None
        self.cam = None
        self.feature_cam = None
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

        # feature_model.to(self.device)
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
            cam = GradCAM(model=model, target_layers = target_layers, use_cuda=True, reshape_transform=self._reshape_transform)
        else:
            target_layers = [model[0].layer4[-1]]
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

        return cam

    def _get_topk(self):

        for n in range(self.n_classes):
            tpk_csv_path = Path(cfg.log_path) / f'test_epoch_{cfg.epoch}' / f'test_c{n}_top_patients.csv'
            tpk_df = pd.read_csv(tpk_csv_path)
            self.topk_dict[str(n)] = {'patients': list(tpk_df.head(5)['Patient']), 'labels': [n] * len(list(tpk_df.head(5)['Patient']))}

    # def _rescale(self, ):


    def assemble(self, wsi_name, batch_coords, grayscale_cam, mil_grayscale_cam, input_h=224):

        Path(f'{self.output_path}/tiles/{wsi_name}').mkdir(parents=True, exist_ok=True)
        position_dict, coords = self._get_position_dict(batch_coords)
        x_max = max([x[0] for x in coords])
        y_max = max([x[1] for x in coords])


        mean_cam = mil_grayscale_cam[:, :, 1].squeeze()
        # mean_cam = mil_grayscale_cam
        # mean_cam = torch.mean(mil_grayscale_cam, dim=2)
        mean_cam -= torch.min(mean_cam)
        mean_cam /= torch.max(mean_cam) #normalize


        # print(mil_grayscale_cam)
        # print(mean_cam.shape)
        # print(mean_cam.shape)
        # print(mean_cam.shape)
        percentage_shown = 0.4 #0.4 for all results
        # topk = int(mean_cam.shape[0]) #
        topk = int(mean_cam.shape[0] * percentage_shown) #
        # print(topk)

        _, topk_indices = torch.topk(mean_cam, topk, dim=0)
        # print(topk_indices)
        # print(len(topk_indices))
        batch_coords = torch.index_select(batch_coords.squeeze(), 0, topk_indices)
        grayscale_cam = torch.index_select(grayscale_cam.squeeze(), 0, topk_indices)
        # print(mean_cam.shape)
        # print(grayscale_cam.shape)
        # print(mean_cam[0])
        # print(grayscale_cam[0, :, :])
        # grayscale_cam = mean_cam@grayscale_cam

        feature_cam = torch.zeros([(y_max+1)*224, (x_max+1)*224])
        print('batch_coords:', batch_coords.shape)
        # print(coords.shape)
        # for i,( c, img, w) in enumerate(zip(batch_coords.squeeze(0), grayscale_cam, mean_cam)):
        for i,( c, img) in enumerate(zip(batch_coords.squeeze(0), grayscale_cam)):
            c = c.squeeze()
            
            x = c[0].item()
            y = c[1].item()
            # print(x, y)
            
            # print(img.shape)
            # if i in topk_indices:
            # feature_cam[y*224:y*224+224, x*224:x*224+224] = img * w
            feature_cam[y*224:y*224+224, x*224:x*224+224] = img
        # feature_cam = (feature_cam - feature_cam.min())/(feature_cam.max()-feature_cam.min())
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

                    # save tile level gradcam
                    tile_features = feature_cam[y_coord*224:(y_coord+1)*224, x_coord*224:(x_coord+1)*224]
                    mask = gaussian_filter(tile_features, sigma=15)
                    tile_cam = show_cam_on_image(img.numpy(), tile_features, use_rgb=True, image_weight=0.6, colormap=cv2.COLORMAP_JET)
                    img_cam = Image.fromarray(tile_cam)
                    img_cam = img_cam.convert('RGB')
                    img_cam.save(f'{self.output_path}/tiles/{wsi_name}/{wsi_name}_({co[0]}-{co[1]})_gradcam.jpg')

                    roi_path =  Path(self.roi_dir) / wsi_name / f'{wsi_name}_({co[0]}-{co[1]}).png'
                    img = np.asarray(Image.open(roi_path)).astype(np.uint8)
                    img = img / 255.0
                    # img = torch.from_numpy(img)
                    roi[y_coord*224:(y_coord+1)*224, x_coord*224:(x_coord+1)*224] = img
        W, H = wsi.shape[0], wsi.shape[1]

        

        
        
        # mil_attention_map = mil_grayscale_cam[:, :, 1].squeeze()
        # mil_attention_map = (mil_attention_map-mil_attention_map.min()) / (mil_attention_map.max() - mil_attention_map.min())
        # mask = torch.zeros(( int(W/input_h), int(H/input_h)))
        # for i, (x,y) in enumerate(coords):
        #     mask[y][x] = mil_attention_map[i]
        # mask = mask.unsqueeze(0).unsqueeze(0)

        # mask = F.interpolate(mask, (W,H), mode='bilinear')
        # mask = mask.squeeze()

        # mask = (mask - mask.min())/(mask.max()-mask.min())
        # # mask[mask<0.1] = 0

        # feature_cam[mask==0] = 0

        # mask = mask + feature_cam

        # mask = (mask - mask.min())/(mask.max()-mask.min())

        mask = gaussian_filter(feature_cam, sigma=15)
        mask = (mask - mask.min())/(mask.max()-mask.min())

        wsi_cam = show_cam_on_image(wsi.numpy(), mask, use_rgb=True, image_weight=0.6, colormap=cv2.COLORMAP_JET)

        roi_idx = (roi==0)
        wsi_cam[roi_idx] = 255
        
        size = (30000, 30000)

        print('Save GradCAM overlay.')
        img = Image.fromarray(wsi_cam)
        img = img.convert('RGB')
        img.save(f'{self.output_path}/{wsi_name}_gradcam.jpg')

        del wsi_cam
        del mask
        del wsi
        del roi


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
        mil_attention_map = mil_grayscale_cam[:, :, 1].squeeze()
        mil_attention_map = (mil_attention_map-mil_attention_map.min()) / (mil_attention_map.max() - mil_attention_map.min())
        mask = torch.zeros(( int(W/input_h), int(H/input_h)))
        for i, (x,y) in enumerate(coords):
            mask[y][x] = mil_attention_map[i]
        mask = mask.unsqueeze(0).unsqueeze(0)

        mask = F.interpolate(mask, (W,H), mode='bilinear')
        mask = mask.squeeze(0).permute(1,2,0)

        mask = (mask - mask.min())/(mask.max()-mask.min())
        mask = mask.numpy()
        mask = gaussian_filter(mask, sigma=15)

        wsi_cam = show_cam_on_image(wsi.numpy(), mask, use_rgb=True, image_weight=0.6)

        #----------------------------------------------
        # Use ROI to filter image
        #----------------------------------------------
        roi_idx = (roi==0)
        wsi_cam[roi_idx] = 255
        
        size = (30000, 30000)

        print('Save GradCAM overlay.')
        img = Image.fromarray(wsi_cam)
        img = img.convert('RGB')
        img.save(f'{self.output_path}/{wsi_name}_mil_gradcam.jpg')


    def run(self, target_label):

        patient_slide_dict_path = f'/{home}/ylan/data/DeepGraft/training_tables/patient_slide_dict_ext.json'
        
        self._get_topk()
        
        feature_model = self._load_feature_model().to(self.device)
        mil_model = self._load_model_from_checkpoint(self.checkpoint).to(self.device)

        model = torch.nn.Sequential(feature_model, mil_model).to(self.device) 
        # print(model)    
        # print(self.model_name)
        mil_cam = self._get_cam_object(self.model_name, mil_model)
        feature_cam = self._get_cam_object('Resnet50', model)

        with open(patient_slide_dict_path, 'r') as f:
            patient_slide_dict = json.load(f)
        slides = []
        for p in self.topk_dict[str(target_label)]['patients']: 
            slides += list(set(patient_slide_dict[p]))
        # print(slides)
        self.output_path = self.output_path / str(target_label)
        self.output_path.mkdir(parents=True, exist_ok=True)
        # print(self.output_path)
        slides_done = [x.stem.rsplit('_', 1)[0] for x in list(self.output_path.iterdir()) if Path(x).suffix == '.jpg']
        # print(slides_done)
        # slides_done += ['Aachen_KiBiDatabase_KiBiAcOSNX750_01_006_HE', 'Aachen_KiBiDatabase_KiBiAcOSNX750_01_008_Jones', 'Aachen_KiBiDatabase_KiBiAcOSNX750_01_018_PAS', 'Aachen_KiBiDatabase_KiBiAcUAYM660_01_006_HE', 'Aachen_KiBiDatabase_KiBiAcUAYM660_01_008_Jones', 'Aachen_KiBiDatabase_KiBiAcUAYM660_01_014_PAS']
        # slides_done += ['Aachen_KiBiDatabase_KiBiAcZXRC970_01_018_PAS', 'Aachen_KiBiDatabase_KiBiAcZXRC970_01_006_HE', 'Aachen_KiBiDatabase_KiBiAcSVXX412_01_006_HE', 'Aachen_KiBiDatabase_KiBiAcUAYM660_01_008_Jones']
        # slides_done += ['Aachen_KiBiDatabase_KiBiAcDKIK860_01_018_PAS', 'Aachen_KiBiDatabase_KiBiAcLAXK110_01_007_PAS', 'Aachen_KiBiDatabase_KiBiAcLAXK110_01_008_Jones']
        # slides_done += ['Aachen_KiBiDatabase_KiBiAcFLGQ191_01_018_PAS', 'Aachen_KiBiDatabase_KiBiAcFLGQ191_01_004_PAS', 'Aachen_KiBiDatabase_KiBiAcFLGQ191_01_008_Jones', ]
        # slides_done += ['Aachen_KiBiDatabase_KiBiAcDKIK860_01_018_PAS'] #, 'Aachen_KiBiDatabase_KiBiAcLAXK110_01_008_Jones', 'Aachen_KiBiDatabase_KiBiAcZXRC970_01_018_PAS', 'Aachen_KiBiDatabase_KiBiAcDKIK860_01_018_PAS']
        slides = [s for s in slides if s not in slides_done]

        try:
            len(slides) != 0
        except:
            print('No Slides available. Please check directory.')
        # print(slides)
        # self.slides_done += ['Aachen_KiBiDatabase_KiBiAcRLKM530_01_006_HE', 'Aachen_KiBiDatabase_KiBiAcRLKM530_01_008_Jones', 'Aachen_KiBiDatabase_KiBiAcRLKM530_01_018_PAS', 'Aachen_KiBiDatabase_KiBiAcUAYM660_01_008_Jones',
        # 'Aachen_KiBiDatabase_KiBiAcUAYM660_01_014_PAS',]
        
        # print('Slides already processed: ', slides_done)
        # self.slides_done.append(['Aachen_KiBiDatabase_KiBiAcDKIK860_01_018_PAS','Aachen_KiBiDatabase_KiBiAcDOST921_01_008_Jones'])

        #skip for norm_rej_rest:

        # print(slides)

        test_dataset = JPGMILDataloader(file_path=self.data_root, label_path=self.label_path, mode='test', cache=False, n_classes=self.n_classes, model=self.model_name, slides=slides)
        dl = DataLoader(test_dataset, batch_size=1, num_workers=4)
        cam_target = [ClassifierOutputTarget(target_label)]


        c = 0
        print(len(dl))
        for item in tqdm(dl):
            # if c >10:
            #     break

            bag, label, (name, batch_coords, patient) = item
            
            # label = torch.Tensor(label)
            # # idx = top_patients.index(patient[0])

            slide_name = name[0]
            print(slide_name)
            print(bag.shape)
            # if slide_name != 'Aachen_KiBiDatabase_KiBiAcRLKM530_01_006_HE':
            # #     continue
            # # else:
            # if slide_name in self.slides_done:
            #     continue
            # if bag.shape[1] > 200:
                
            #     temp = []
            #     size_remaining = bag.shape[1]
            #     i = 1
            #     while size_remaining // 200 != 0:
            #     # for i in range(bag.shape[1]//200 + 1):[[]]
            #         sub_bag = bag[:, (i-1)*200:i*200, : , :, :].flloat().squeeze(0)
            #         i += 1
            #         size_remaining -= 200
            #         temp.append(sub_bag)
            #     else: 
            #         sub_bag = bag[:, size_remaining%200: , : , :, :].flloat().squeeze(0)
            #         temp.append(sub_bag)
                



            # half_size = int(bag.shape[1]/2)
            # half_bag_1 = bag[:,:half_size, :, :, :].float().squeeze(0)
            # half_bag_2 = bag[:,half_size:, :, :, :].float().squeeze(0)



            # # bag = bag.float().squeeze(0) #.to(self.device)
            # instance_count = bag.size(0)

            # grayscale_cam_1 = feature_cam(input_tensor=half_bag_1.detach(), targets=cam_target)
            # grayscale_cam_2 = feature_cam(input_tensor=half_bag_2.detach(), targets=cam_target)

            # # print(grayscale_cam_1.shape)
            # # print(grayscale_cam_2.shape)
            # grayscale_cam = torch.cat((torch.Tensor(grayscale_cam_1), torch.Tensor(grayscale_cam_2)))
            # # grayscale_cam = torch.Tensor(grayscale_cam)
            # # print(grayscale_cam.shape)
            
            # with torch.no_grad():
            #     features = feature_model(bag.squeeze().to(self.device))

            
            bag = bag.squeeze(0)
            print(bag.shape)
            background = torch.full((100,3,224,224), 255.0).to(self.device)
            e = shap.DeepExplainer(model, background)
            shap_values = e.shap_values(bag)
            shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
            test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

            shap.image_plot(shap_numpy, -test_numpy)



            # mil_grayscale_cam = mil_cam(input_tensor=features.unsqueeze(0), targets=cam_target)
            # mil_grayscale_cam = torch.Tensor(mil_grayscale_cam)[:instance_count, :]
            # # mil_grayscale_cam = mil_grayscale_cam[:, :, 1].squeeze()

            
            # self.assemble(slide_name, batch_coords, grayscale_cam, mil_grayscale_cam)
            # self._save_attention_map(slide_name, batch_coords, mil_grayscale_cam)


            # c+= 1

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

    target_label = args.target_label
    print(task)
    print(model_paths)
    print(cfg.log_path)
    
    # for target_label in range(args.total_classes):
    visualizer = Visualize(checkpoint_path=model_paths[0], task=cfg.task)
    visualizer.run(target_label)
