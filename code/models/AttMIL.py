import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    import apex
    apex_available=True
except ModuleNotFoundError:
    # Error handling
    apex_available = False
    pass


class AttMIL(nn.Module): #gated attention
    def __init__(self, n_classes, in_features=2048, out_features=512):
        super(AttMIL, self).__init__()
        self.L = out_features
        self.D = 128
        self.K = 1
        self.n_classes = n_classes

        if apex_available: 
            norm_layer = apex.normalization.FusedLayerNorm
        else:
            norm_layer = nn.LayerNorm


        # resnet50 = models.resnet50(pretrained=True)    
        # modules = list(resnet50.children())[:-3]

        # self.resnet_extractor = nn.Sequential(
        #     *modules,
        #     nn.AdaptiveAvgPool2d(1),
        #     View((-1, 1024)),
        #     nn.Linear(1024, self.L)
        # )

        # self.feature_extractor1 = nn.Sequential(
        #     nn.Conv2d(3, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),

        #     # View((-1, 50 * 4 * 4)),
        #     # nn.Linear(50 * 4 * 4, self.L),
        #     # nn.ReLU(),
        # )
        if in_features == 2048:
            self._fc1 = nn.Sequential(
                nn.Linear(in_features, int(in_features/2)), nn.GELU(), nn.Dropout(p=0.6), norm_layer(int(in_features/2)),
                nn.Linear(int(in_features/2), out_features), nn.GELU(),
                ) 
        elif in_features == 1024:
            self._fc1 = nn.Sequential(
                # nn.Linear(in_features, int(in_features/2)), nn.GELU(), nn.Dropout(p=0.2), norm_layer(out_features),
                nn.Linear(in_features, out_features), nn.GELU(), nn.Dropout(p=0.6), norm_layer(out_features)
                ) 

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(in_features, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.n_classes),
        )    

    def forward(self, x):
        # H = kwargs['data'].float().squeeze(0)
        # H = x.float().squeeze(0).squeeze(0)
        # H = x.float().squeeze()
        # print(H.shape)
        # H = self.feature_extractor_part2(H)
        # print(H.shape)
        # print(x.shape)
        x = x.squeeze()
        H = self._fc1(x)
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        # out_A = A
        if len(A.shape) < 2:
            A = A.unsqueeze(0)
        # print(A.shape)
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxL
        logits = self.classifier(M)
       
        return logits

