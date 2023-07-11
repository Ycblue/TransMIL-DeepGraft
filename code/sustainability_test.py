import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from tqdm import tqdm
# from torchvision.datasets import Dataset
from torch.utils.data import random_split, Dataset, DataLoader
from experiment_impact_tracker.compute_tracker import ImpactTracker

from models.ResNet import resnet50
from models import TransMIL, CLAM_MB
import timm

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
        return bag, label

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='TransMIL', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    from utils.custom_resnet50 import resnet50_baseline
    from torchvision import models

    tracker = ImpactTracker(f'co2log/')
    tracker.launch_impact_monitor()


    args = make_parse()

    model_name = args.model
    n_classes = 2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_name == 'TransMIL':
        in_features = 2048
        model = TransMIL(n_classes=n_classes, in_features=2048)
        mode = 'features'
        feature_size = in_features
    elif model_name == 'RetCCL':
        model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
        model.fc = torch.nn.Identity()
        model.load_state_dict(torch.load('models/ckpt/retccl_best_ckpt.pth'), strict=False)
        mode = 'images'
        feature_size = 224
    elif model_name == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, 1024)
        for param in model.parameters():
            param.requires_grad = False
        # model.load_state_dict(torch.load('models/ckpt/retccl_best_ckpt.pth'), strict=False)
        mode = 'images'
        feature_size = 224
    elif model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=n_classes)
        for param in model.parameters():
            param.requires_grad = False
        outputs_attrs = n_classes
        num_inputs = model.head.in_features
        last_layer = nn.Linear(num_inputs, outputs_attrs)
        model.head = last_layer
        mode = 'images'
        feature_size = 224
    elif model_name == 'inception':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights='Inception_V3_Weights.DEFAULT')
        model.aux_logits = False
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        mode = 'images'
        feature_size = 384
    elif model_name == 'clam':
        model = CLAM_MB(n_classes = n_classes)
        mode = 'features'
        feature_size = 1024
    model = model.to(device)
    # test_data = torch.rand([1000,3,224,224]).to(device)

    dataset = CustomImageDataset(data_size=1000, device=device, mode=mode, feature_size=feature_size)
    dataloader = DataLoader(dataset, batch_size=1)

    epochs = 1
    for e in range(epochs):
        start = time.time()
        for item in tqdm(dataloader):
            # print(i)
            bag, label = item
            bag.to(device)
            if len(bag.shape) > 4:
                bag = bag.squeeze()
            label.to(device)
            if model_name == 'clam':
                bag = bag.squeeze(0)
            with torch.cuda.amp.autocast():
                features = model(bag)
        end = time.time()
        print('Epoch Time: ', end-start)

    tracker.stop()
    from experiment_impact_tracker.data_interface import DataInterface
    data_interface = DataInterface(['co2log'])
    kg_carbon = data_interface.kg_carbon
    print(kg_carbon)

    '''Netherlands'''

    txt_path = f'../test/co2_emission/{model_name}.txt'
    with open(txt_path, 'w') as f:
        f.write('Emissions calculated for 1000 slides, 1000 patches/slide. \n')
        f.write(f'{model_name}: {kg_carbon}kg\n')
    