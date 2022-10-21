import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

'''
Chowder implementation from Courtiol 2018(https://openreview.net/pdf?id=ryserbZR-) 
'''


class Chowder(nn.Module): 
    def __init__(self, n_classes, features=512, r=5):
        super(Chowder, self).__init__()
        self.L = features
        self.n_classes = n_classes
        self.R = r


        self.f1 = nn.Sequential(
            nn.Conv1d(self.L, 1, 1),
        )
        self.f2 = nn.Sequential(
            nn.Linear(r*2, 200),
            nn.Linear(200, 100),
            nn.Linear(100, self.n_classes),
            # nn.Sigmoid()
        )
        
    def forward(self, x):

        x = x.float()
        x = torch.transpose(x, 1, 2)

        x = self.f1(x)
        max_indices = torch.topk(x, self.R).values
        min_indices = torch.topk(x, self.R, largest=False).values

        cat_minmax = torch.cat((min_indices, max_indices), dim=2)

        out = self.f2(cat_minmax).squeeze(0)
        
        return out, None

if __name__ == '__main__':

    data = torch.randn((1, 6000, 512)).cuda()
    model = Chowder(n_classes=2).cuda()
    print(model.eval())
    logits, _ = model(data)

    print(logits.shape)