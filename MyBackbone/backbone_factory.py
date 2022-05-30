import torch
import torch.nn as nn

from transformers import AutoFeatureExtractor, ViTModel
from torchvision import models

def init_backbone(**kargs):
    
    backbone = kargs['backbone']
    n_classes = kargs['n_classes']
    out_features = kargs['out_features']

    if backbone == 'dino' or backbone == 'vit':

        if backbone == 'dino':        
            feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/dino-vitb16')
            model_ft = ViTModel.from_pretrained('facebook/dino-vitb16', num_labels=n_classes)

        def model_ft(input):
            input = feature_extractor(input, return_tensors='pt')
            features = model_ft(**input)


    elif kargs['backbone'] == 'resnet18':
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())[:-1]
        # model_ft.fc = nn.Linear(512, out_features)

        res18 = nn.Sequential(
            *modules,
        )
        for param in res18.parameters():
            param.requires_grad = False
        self.model_ft = nn.Sequential(
            res18,
            nn.AdaptiveAvgPool2d(1),
            View((-1, 512)),
            nn.Linear(512, self.out_features),
            nn.GELU(),
        )
    elif kargs['backbone'] == 'resnet50':

        resnet50 = models.resnet50(pretrained=True)    
        # model_ft.fc = nn.Linear(1024, out_features)
        modules = list(resnet50.children())[:-3]
        res50 = nn.Sequential(
            *modules,     
        )
        for param in res50.parameters():
            param.requires_grad = False
        self.model_ft = nn.Sequential(
            res50,
            nn.AdaptiveAvgPool2d(1),
            View((-1, 1024)),
            nn.Linear(1024, self.out_features),
            nn.GELU()
        )
    elif kargs['backbone'] == 'efficientnet':
        efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b0', pretrained=True)
        for param in efficientnet.parameters():
            param.requires_grad = False
        self.model_ft = nn.Sequential(
            efficientnet,
            nn.Linear(1000, 512),
            nn.GELU(),
        )
    elif kargs['backbone'] == 'simple': #mil-ab attention
        feature_extracting = False
        self.model_ft = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            View((-1, 1024)),
            nn.Linear(1024, self.out_features),
            nn.ReLU(),
        )

