import argparse
from pathlib import Path
import numpy as np
import glob
import re

from sklearn.model_selection import KFold
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# from datasets.data_interface import DataInterface, MILDataModule, CrossVal_MILDataModule
# from datasets import JPGMILDataloader, MILDataModule, FeatureBagLoader
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

class InferenceModel(nn.Module):
    def __init__(self, feature_model, mil_model):
        super(InferenceModel, self).__init__()

        self.feature_model = feature_model
        self.mil_model = mil_model

    def forward(self, x):

        
        # batch_size = x.shape[0]
        bag_size = x.shape[0]
        # bag = x.view(batch_size*bag_size, x.shape[2], x.shape[3], x.shape[4])

        x = x.squeeze()
        feats = self.feature_model(x)
        # feats = feats.view(batch_size, bag_size, -1)
        logits = self.mil_model(feats.unsqueeze(0))

        return logits


class RETCCL_FE(pl.LightningModule):
    def __init__(self):
        self.model_ft = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
        home = Path.cwd().parts[1]
        # pre_model = 
        # self.model_ft.fc = nn.Identity()
        # self.model_ft.load_from_checkpoint(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth', strict=False)
        self.model_ft.load_state_dict(torch.load(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
        for param in self.model_ft.parameters():
            param.requires_grad = False
        self.model_ft.fc = torch.nn.Identity()
        # self.model_ft.to(self.device)
    
    def forward(self, x):
        return self.model_ft(x)
        
# def reshape_transform(tensor, height=14, width=14):
#     result = tensor[:, 1 :  , :].reshape(tensor.size(0),
#         height, width, tensor.size(2))

#     # Bring the channels to the first dimension,
#     # like in CNNs.
#     result = result.transpose(2, 3).transpose(1, 2)
#     return result

def reshape_transform(tensor):
    # print('reshape_transform')
    # print(tensor.shape)
    H = tensor.shape[1]
    _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
    # print(_H, _W)
    add_length = _H * _W - H
    tensor = torch.cat([tensor, tensor[:,:add_length,:]],dim = 1)
    # print(tensor.shape)
    result = tensor[:, :, :].reshape(tensor.size(0), _H, _W, tensor.size(2))
    # print(result.shape)
    result = result.transpose(2,3).transpose(1,2)
    # print(result.shape)
    # print('----------')
    return result


def save_attention_map(wsi_name, batch_names, mil_grayscale_cam, target, task='norm_rest', device='cpu', input_h=224):

    # def get_coords(batch_names): #ToDO: Change function for precise coords
    #     coords = []
        
    #     for tile_name in batch_names: 
    #         # print(tile_name)
    #         pos = re.findall(r'\((.*?)\)', tile_name)
    #         # print(pos)
    #         x, y = pos[-1].split('-')
    #         # print(x, y)
    #         coords.append((int(x),int(y)))

    #     return coords
    

    def get_coords(batch_names):
        coords = []
        batch_names = batch_names.squeeze(0)
        for i in range(batch_names.shape[0]): 
            c = batch_names[i, :]
            # print(c)
            x = c[0]
            y = c[1]
            coords.append((int(x),int(y)))
        return coords

    home = Path.cwd().parts[1]
    jpg_dir = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated/Aachen_Biopsy_Slides/TEST'
    roi_dir = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated/Aachen_Biopsy_Slides/ROI'
    save_path = Path(f'/{home}/ylan/workspace/TransMIL-DeepGraft/test/mil_model/{task}')

    output_path = save_path / str(target)
    # output_path = save_path / str(target.item())
    output_path.mkdir(parents=True, exist_ok=True)

    coords = get_coords(batch_names)
    data = []

    # print('Assembled: ')

    position_dict = {}
    assembled = []
    # for tile in self.predictions:
    count = 0
    white_value = 0
    x_max = max([x[0] for x in coords])
    y_max = max([x[1] for x in coords])

    for i, (x,y) in enumerate(coords):
        if x not in position_dict.keys():

            position_dict[x] = [(y, i)]
        else: position_dict[x].append((y, i))

    wsi = torch.ones([(y_max+1)*224, (x_max+1)*224, 3])
    roi = np.zeros([(y_max+1)*224, (x_max+1)*224])
    for i in range(x_max+1):
        if i in position_dict.keys():
            for j in position_dict[i]:
                y_coord = int(j[0])
                x_coord = int(i)
                # sample_idx = j[1]
                # print(coords[j[1]])
                co = coords[j[1]]
                tile_path =  Path(jpg_dir) / wsi_name / f'{wsi_name}_({co[0]}-{co[1]}).png'
                img = np.asarray(Image.open(tile_path)).astype(np.uint8)
                img = img / 255.0
                img = torch.from_numpy(img)
                wsi[y_coord*224:(y_coord+1)*224, x_coord*224:(x_coord+1)*224, :] = img

                roi_path =  Path(roi_dir) / wsi_name / f'{wsi_name}_({co[0]}-{co[1]}).png'
                img = np.asarray(Image.open(roi_path)).astype(np.uint8)
                img = img / 255.0
                # img = torch.from_numpy(img)
                roi[y_coord*224:(y_coord+1)*224, x_coord*224:(x_coord+1)*224] = img
    
    
    # roi = torch.stack((roi, roi, roi), dim=2)
    W, H = wsi.shape[0], wsi.shape[1]
    # test mil_model attention map
    mil_attention_map = mil_grayscale_cam[:, :, 1].squeeze()
    mil_attention_map = (mil_attention_map-mil_attention_map.min()) / (mil_attention_map.max() - mil_attention_map.min())

    input_h = 224
    # print(input_h)
        
    mil_mask = torch.zeros(( int(W/input_h), int(H/input_h)))
    # print('mil_mask: ', mil_mask.shape)
    # print(len(coords))
    for i, (x,y) in enumerate(coords):
        mil_mask[y][x] = mil_attention_map[i]
    mil_mask = mil_mask.unsqueeze(0).unsqueeze(0)

    mil_mask = F.interpolate(mil_mask, (W,H), mode='bilinear')
    mil_mask = mil_mask.squeeze(0).permute(1,2,0)

    mil_mask = (mil_mask - mil_mask.min())/(mil_mask.max()-mil_mask.min())
    mil_mask = mil_mask.numpy()
    mil_mask = gaussian_filter(mil_mask, sigma=15)
    # mil_mask = mil_mask.unsqueeze(0).unsqueeze(0)
    # print(mil_mask)
    # print(mil_mask.max(), mil_mask.min())

    # Save original
    # print('Save Original.')
    # wsi_out = ((wsi-wsi.min())/(wsi.max()-wsi.min()) * 255.0).numpy().astype(np.uint8)
    # img = Image.fromarray(wsi_out)
    # img = img.convert('RGB')
    # size = (30000, 30000)
    # img.thumbnail(size, Image.Resampling.LANCZOS)
    # output_path = save_path / str(target)
    # # output_path = save_path / str(target.item())
    # output_path.mkdir(parents=True, exist_ok=True)
    # img.save(f'{output_path}/{wsi_name}.png')
    # del wsi_out
    #--> Get interpolated mask from GradCam
    # W, H = wsi.shape[0], wsi.shape[1]
    # mask = torch.zeros((W,H))
    # for i, (x,y) in enumerate(coords):
    #     mask[y*224:(y+1)*224, x*224:(x+1)*224] = grayscale_cam[i] #* mil_attention_map[i]
    
    # mask = mask.unsqueeze(0).unsqueeze(0)
    # mask = mask.squeeze(0).permute(1,2,0)
    # # print(mask)
    # # mask = 255 * (mask / mask.max())
    # mask = mask.numpy()
    # mask = gaussian_filter(mask, sigma=15)
    #------------------------------------------
    # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_PLASMA)
    # img = Image.fromarray(255 - np.uint8(heatmap))
    # img.save(f'{output_path}/{wsi_name}_attention.png')

    # mask_img = Image.fromarray(mask.unsqueeze(0))
    # mask_img = mask_img.filter(ImageFilter.MORE_SMOOTH)

    # print('mask: ', mask.shape)
    
    # def show_cam_on_image(img, mask):
    #     heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    #     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    #     heatmap = np.float32(heatmap) / 255
    #     cam = heatmap*0.4 + np.float32(img)
    #     cam = cam / np.max(cam)
        # return cam

    wsi_cam = show_cam_on_image(wsi.numpy(), mil_mask, use_rgb=True, image_weight=0.6)

    print('Filter with ROI Mask.')
    roi_idx = (roi==0)
    wsi_cam[roi_idx] = 255
    # print(wsi_cam)
    # wsi_cam = ((wsi_cam-wsi_cam.min())/(wsi_cam.max()-wsi_cam.min()) * 255.0).astype(np.uint8)
    
    size = (30000, 30000)

    print('Save GradCAM overlay.')
    img = Image.fromarray(wsi_cam)
    img = img.convert('RGB')
    # img.thumbnail(size, Image.Resampling.LANCZOS)
    # img.resize(size, resample=Image.Resampling.LANCZOS)
    # img = img.filter(ImageFilter.MORE_SMOOTH)
    # output_path = save_path / str(target.item())
    # output_path.mkdir(parents=True, exist_ok=True)
    img.save(f'{output_path}/{wsi_name}_gradcam.jpg')


#---->main
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

    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    scaler = torch.cuda.amp.GradScaler()
    
    home = Path.cwd().parts[1]
    ckpt_pth = Path(cfg.log_path) / 'checkpoints'
    model_paths = list(ckpt_pth.glob('*.ckpt'))

    # print(model_paths)
    if cfg.epoch == 'last':
        model_paths = [str(model_path) for model_path in model_paths if f'last' in str(model_path)]
    else:
        model_paths = [str(model_path) for model_path in model_paths if f'epoch={cfg.epoch}' in str(model_path)]

    # checkpoint = torch.load(f'{cfg.log_path}/checkpoints/epoch=04-val_loss=0.4243-val_auc=0.8243-val_patient_auc=0.8282244801521301.ckpt')
    # checkpoint = torch.load(f'{cfg.log_path}/checkpoints/epoch=73-val_loss=0.8574-val_auc=0.9682-val_patient_auc=0.9724310636520386.ckpt')
    # print(model_paths)
    checkpoint = torch.load(model_paths[0])

    hyper_parameters = checkpoint['hyper_parameters']
    n_classes = hyper_parameters['model']['n_classes']
    # batch_size = 5


    feature_model = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
    feature_model.load_state_dict(torch.load(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
    feature_model.fc = torch.nn.Identity()
    feature_model.to(device)
    feature_model.eval()

    mil_model = TransMIL(n_classes=n_classes, in_features=2048).to(device)
    model_weights = checkpoint['state_dict']

    for key in list(model_weights):
        model_weights[key.replace('model.', '')] = model_weights.pop(key)
    
    mil_model.load_state_dict(model_weights)
    # mil_model.eval()


    big_model = InferenceModel(feature_model, mil_model).to(device)
    big_model.eval()
    count = 0
    
    # for param in big_model.parameters():
    #     param.requires_grad = False
    


    home = Path.cwd().parts[1]
    data_root = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated'
    label_path = hyper_parameters['data']['label_path']
    # label_path = f'/{home}/ylan/data/DeepGraft/training_tables/dg_split_PAS_HE_Jones_norm_rej_rest_val_1.json'
    
    # if self.model_name == 'TransMIL':
        # print(self.model.layer2.norm)
    # print(big_model)
    target_layers = [big_model.feature_model.layer4[-1]] # 32x32
    # target_layers = [big_model.mil_model.layer2.norm] # 32x32
    # target_layers = [mil_model.layer2.norm] # 32x32
    # target_layers = [self.model_ft[0].features[-1]] # 32x32
    cam = GradCAM(model=big_model, target_layers = target_layers, use_cuda=True) #, reshape_transform=self.reshape_transform
    mil_cam = GradCAM(model=mil_model, target_layers = [mil_model.norm], use_cuda=True, reshape_transform=reshape_transform) #, reshape_transform=self.reshape_transform
        # self.cam_ft = GradCAM(model=self.model, target_layers = target_layers_ft, use_cuda=True) #, reshape_transform=self.reshape_transform
    # elif self.model_name == 'TransformerMIL':
    #     target_layers = [self.model.layer1.norm]
    #     self.cam = EigenCAM(model=self.model, target_layers = target_layers, use_cuda=True, reshape_transform=self.reshape_transform)
    #     # self.cam = GradCAM(model=self.model, target_layers = target_layers, use_cuda=True, reshape_transform=self.reshape_transform)
    # else:
    #     target_layers = [self.model.attention_weights]
    #     self.cam = GradCAM(model = self.model, target_layers = target_layers, use_cuda=True)

    start = time.time()
    test_logits = []
    test_probs = []
    test_labels = []
    test_topk_data = []
    data = [{"count": 0, "correct": 0} for i in range(n_classes)]
    count = 0

    # print(len(dl))

    test_patient_dict = {}

    top_patients = []
    target_labels = []
    # for n in range(n_classes):

        #------------------------------------------------------------------------------------------------
        # Get Scores for class 1 
    n = 0
    tpk_csv_path = Path(cfg.log_path) / f'test_epoch_{cfg.epoch}' / f'test_c{n}_top_patients.csv'
    tpk_df = pd.read_csv(tpk_csv_path)
    top_patients += list(tpk_df.head(5)['Patient'])
    target_labels += [n] * len(list(tpk_df.head(5)['Patient']))
            #------------------------------------------------------------------------------------------------
    patient_slide_dict_path = f'/{home}/ylan/data/DeepGraft/training_tables/patient_slide_dict.json'
    with open(patient_slide_dict_path, 'r') as f:
        patient_slide_dict = json.load(f)
    slides = []
    for p in top_patients: 
        slides += patient_slide_dict[p]
    # slides = ['Aachen_KiBiDatabase_KiBiAcSVXX411_01_001_PAS']
    # slides = ['Aachen_KiBiDatabase_KiBiAcWBSS520_01_001_PAS']
    # slides = ['Aachen_KiBiDatabase_KiBiAcTTVB560_01_002_HE']
    # print(slides)
    test_dataset = JPGMILDataloader(data_root, label_path=label_path, mode='test', cache=False, n_classes=n_classes, model='TransMIL', slides=slides)
    # for t in test_dataset:
    #     print(t)
    # test_dataset = FeatureBagLoader(data_root, label_path=label_path, mode='test', n_classes=n_classes, cache=False, model='TransMIL')

    dl = DataLoader(test_dataset, batch_size=1, num_workers=1, pin_memory=True)


    print('top_patients: ')
    print(top_patients)
    print('target_labels: ')
    print(target_labels)

    c = 0
    # for slide in slides:
    #     test_dataset = JPGMILDataloader(data_root, label_path=label_path, mode='test', cache=False, n_classes=n_classes, model='TransMIL', slides=[slide])
    for item in tqdm(dl): 
        
        bag, label, (name, batch_coords, patient) = item

        label = torch.Tensor(label)
        if patient[0] not in top_patients:
            continue
        idx = top_patients.index(patient[0])

        name = name[0]

        bag = bag.float().to(device)
        # bag = bag.unsqueeze(0) 
        print(bag.shape)
        with torch.cuda.amp.autocast():
            features = feature_model(bag.squeeze())
        # features = feature_model(bag.squeeze())
        # logits = self.mil_model(feats.unsqueeze(0))

        # logits = mil_model(bag)
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        #     logits = big_model(bag.squeeze())
        
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        instance_count = bag.size(0)
        # Y = torch.argmax(label)

        target = [ClassifierOutputTarget(target_labels[idx])]
        # target = [ClassifierOutputTarget(1)]
        # print(features.shape)
        mil_grayscale_cam = mil_cam(input_tensor=features.unsqueeze(0), targets=target)
        # mil_grayscale_cam = mil_cam(input_tensor=features, targets=target)
        print('mil_grayscale_cam: ', mil_grayscale_cam.shape)
        mil_grayscale_cam = torch.Tensor(mil_grayscale_cam)[:instance_count, :]
        
        

        # print('GradCAM')
        # grayscale_cam = cam(input_tensor=bag, targets=target)
        
        # # print(grayscale_cam.shape)
        # grayscale_cam = torch.from_numpy(grayscale_cam)
        # grayscale_cam = torch.Tensor(grayscale_cam)[:instance_count, :]
        
        bag = bag.detach()
        print(name)
        save_attention_map(name, batch_coords, mil_grayscale_cam, target=target_labels[idx], device=device)

        torch.cuda.empty_cache()
            # for y, y_hat in zip(label, Y_hat):
            #     y = int(y)
            #     # print(Y_hat)
            #     data[y]["count"] += 1
            #     data[y]["correct"] += (int(y_hat) == y)

    # test_logits = torch.cat(test_logits, dim=0)
    # probs = torch.cat(test_probs).detach().cpu()
    # if n_classes <=2:
    #     out_probs = probs[:,1] 
    # else: out_probs = probs
    # targets = torch.cat(test_labels).squeeze().detach().cpu()
    
    # for c in range(n_classes):
    #     count = data[c]['count']
    #     correct = data[c]['correct']
    #     if count == 0:
    #         acc = None
    #     else: 
    #         acc = float(correct) / count
    #     print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))



    # auroc = AUROC(probs, targets)
    # auroc = binary_auroc(out_probs, targets)
    # print(auroc)
    end = time.time()
    print('Bag Time: ', end-start)



