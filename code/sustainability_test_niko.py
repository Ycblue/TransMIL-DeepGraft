import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import os
from tqdm import tqdm
# from torchvision.datasets import Dataset
from torch.utils.data import random_split, Dataset, DataLoader

from models.ResNet import resnet50
from models import TransMIL, CLAM_MB
import timm
from pathlib import Path
from torchvision import transforms as T
from experiment_impact_tracker.compute_tracker import ImpactTracker
from experiment_impact_tracker.data_interface import DataInterface
import multiprocessing as mp
import torch.nn.functional as F

outdir = r'/home/nschmitz/projects/05_Sustainability/yl_sustain'

class CustomImageDataset(Dataset):
    def __init__(self, data_size=1000, bag_size=1000, feature_size=2048, device='cpu', mode='features'):
        self.data_size = data_size
        self.bag_size = bag_size
        self.device = device
        self.mode = mode
        self.feature_size = feature_size
        # self.data = torch.rand([self.bag_size, 3, 224, 224])

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):

        label = torch.randint(1, (1,1)).to(device)
        if self.mode == 'features':
            bag = torch.rand([self.bag_size, self.feature_size]).to(device)
        else:
            bag = torch.rand([self.bag_size, 3, self.feature_size, self.feature_size]).to(device)
            # bag = T.Resize(224)(bag)
        return bag, label

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='transmil', type=str)
    parser.add_argument('--feature_size', default=224, type=int)
    parser.add_argument('--bag_size', default=250, type=int)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    from utils.custom_resnet50 import resnet50_baseline
    from torchvision import models

    # mp.set_start_method("fork")

    

    args = make_parse()

    model_name = args.model
    feature_size = args.feature_size
    n_classes = 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_list = ['resnet50', 'retccl', 'transmil', 'resnet50', 'vit', 'inception']
    # model_list = ['transmil']
    for model_name in [args.model]:
    # for model_name in model_list:
    

        if model_name == 'transmil':
            # in_features = 2048
            f_model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
            f_model.fc = torch.nn.Identity()
            f_model.load_state_dict(torch.load('/homeStor1/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
            f_model.eval()

            model_ckpt_path = '/homeStor1/ylan/workspace/TransMIL-DeepGraft/logs/DeepGraft/TransMIL/norm_rej_rest/_features_CrossEntropyLoss/lightning_logs/version_53/checkpoints/epoch=17-val_loss=0.9646-val_auc= 0.7541-val_patient_auc=0.0000.ckpt'
            model_ckpt = torch.load(model_ckpt_path)
            model_weights = model_ckpt['state_dict']
            
            model = TransMIL(n_classes=n_classes, in_features=2048)

            for key in list(model_weights):
                model_weights[key.replace('model.', '')] = model_weights.pop(key)
            
            model.load_state_dict(model_weights)
            
            mode = 'images'
            feature_size = 224
            f_model.to(device)
        elif model_name == 'retccl':
            model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
            model.fc = torch.nn.Identity()
            model.load_state_dict(torch.load('/homeStor1/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
            mode = 'images'
            feature_size = 224
        elif model_name == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1')
            model.fc = nn.Linear(model.fc.in_features, 1024)
            for param in model.parameters():
                param.requires_grad = False
            # model.load_state_dict(torch.load('models/ckpt/retccl_best_ckpt.pth'), strict=False)

            mode = 'images'
            # feature_size = 224
        elif model_name == 'vit':
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=n_classes)
            for param in model.parameters():
                param.requires_grad = False
            outputs_attrs = n_classes
            num_inputs = model.head.in_features
            last_layer = nn.Linear(num_inputs, outputs_attrs)
            model.head = last_layer
            mode = 'images'
            # feature_size = 224
        elif model_name == 'inception':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights='Inception_V3_Weights.DEFAULT')
            model.aux_logits = False
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(model.fc.in_features, n_classes)
            mode = 'images'
            feature_size = 299
        elif model_name == 'clam':

            f_model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
            # f_model.fc = torch.nn.Identity()
            f_model.load_state_dict(torch.load('/homeStor1/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
            f_model.fc = torch.nn.Linear(f_model.fc.in_features, 1024)

            # f_model = 


            model = CLAM_MB(n_classes = n_classes)
            mode = 'images'
            feature_size = 224
            # epochs = 100
            f_model.eval()

            f_model.to(device)
        # if f_model: 
        #     f_model.to(device)
        model = model.to(device)
        model.eval()
        # test_data = torch.rand([1000,3,224,224]).to(device)
        epochs=100

        data_size = 1#how many slides are processed

        bag_size = args.bag_size
        
        # epochs = 1

        for bs in [250]:

            dataset = CustomImageDataset(data_size=data_size, bag_size=bs, device=device, mode=mode, feature_size=feature_size)
            dataloader = DataLoader(dataset, batch_size=1)

            # optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
            # tracker_path = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/co2log/{model_name}'
            if not os.path.exists(os.path.join(outdir, f'co2log/{model_name}')):
                os.makedirs(os.path.join(outdir, f'co2log/{model_name}'))

            tracker_path = os.path.join(outdir, f'co2log/{model_name}')
            Path(tracker_path).mkdir(exist_ok=True)
            tracker = ImpactTracker(tracker_path)
            tracker.launch_impact_monitor()
            with torch.no_grad():
                for e in tqdm(range(epochs)):
                    loss = 0
                    for item in dataloader:
                        bag, label = item
                        bag.to(device)
                        
                        if len(bag.shape) > 4:
                            bag = bag.squeeze(0)
                        label.to(device)
                        # if model_name == 'clam':
                        #     bag = bag.squeeze(0)


                        # print(bag.shape)
                        with torch.cuda.amp.autocast():
                            if model_name == 'transmil' or model_name == 'clam':
                                # print(bag.shape)
                                features = f_model(bag)
                                # print(features.shape)
                                if model_name == 'clam':
                                    # print(features.shape)
                                    out = model(features)
                                else:  
                                    out = model(features.unsqueeze(0))
                            else: 
                                features = model(bag)
                        # logits = torch.argmax(features, dim=1)
                        # prob = F.max(logits, dim=1)
                        
                        # # loss += (prob - label)
                        # loss = prob - label
                        # loss.backward()                
                        bag = bag.detach()
            # end = time.time()
                    # print('Epoch Time: ', end-start)

            tracker.stop()
            del tracker

            data_interface = DataInterface([tracker_path])

            print('====================')
            print(f'{model_name}, Epochs: {epochs}, Data Size: {data_size}, Bag Size: {bs}')
            print('====================')
            print('kg_carbon: ', data_interface.kg_carbon)
            print('kg_carbon/epoch: ', data_interface.kg_carbon / epochs)
            print(f'kg_carbon/single slide: ', data_interface.kg_carbon / epochs / data_size)
            print('g_carbon: ', data_interface.kg_carbon * 1000)
            print('g_carbon/epoch: ', data_interface.kg_carbon * 1000 / epochs)
            print(f'g_carbon/single slide: ', data_interface.kg_carbon * 1000 / epochs / data_size)
            kg_carbon = data_interface.kg_carbon / epochs
            # print(kg_carbon)

            '''Netherlands'''

            if not os.path.exists(os.path.join(outdir, 'co2_emission')):
                os.makedirs(os.path.join(outdir, 'co2_emission'))

            txt_path = os.path.join(outdir, f'co2_emission/{model_name}_grad.txt')
            # Path(txt_path).mkdir(exist_ok=True)
            with open(txt_path, 'a') as f:
                f.write(f'==================================================================================== \n')
                f.write(f'Emissions calculated for {data_size} slides, {bs} patches/slide, {feature_size} features, per epoch \n')
                f.write(f'{model_name}: {kg_carbon} [kg]\n')
                f.write(f'{model_name}: {kg_carbon*1000} [g]\n')
                f.write(f'Emissions calculated for 1 slides, {bs} patches/slide, per epoch \n')
                f.write(f'{model_name}: {kg_carbon/data_size} [kg]\n')
                f.write(f'{model_name}: {kg_carbon*1000/data_size} [g]\n')
                f.write(f'==================================================================================== \n')


