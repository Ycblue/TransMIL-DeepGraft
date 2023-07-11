import sys
import numpy as np
import re
import inspect
import importlib
import random
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from pytorch_pretrained_vit import ViT
from pprint import pprint

#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import cross_entropy_torch
from utils.custom_resnet50 import resnet50_baseline

from timm.loss import AsymmetricLossSingleLabel
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from transformers import ViTForImageClassification
from libauc.losses import AUCMLoss, AUCM_MultiLabel, CompositionalAUCLoss
from libauc.optimizers import PESG, PDSCA
#---->
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.functional import stat_scores
from torchmetrics.functional.classification import binary_auroc, multiclass_auroc, binary_precision_recall_curve, multiclass_precision_recall_curve, confusion_matrix
from torchmetrics.utilities.compute import _auc_compute_without_check, _auc_compute


from torch import optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from monai.config import KeysCollection
from monai.data import Dataset, load_decathlon_datalist
from monai.data.wsi_reader import WSIReader
from monai.metrics import Cumulative, CumulativeAverage
from monai.networks.nets import milmodel

# from sklearn.metrics import roc_curve, auc, roc_curve_score


#---->
import pytorch_lightning as pl
from .vision_transformer import vit_small
import torchvision
from torchvision import models
from torchvision.models import resnet
from transformers import AutoFeatureExtractor, ViTModel, SwinModel

from pytorch_grad_cam import GradCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from captum.attr import LayerGradCam
import models.ResNet as ResNet
import timm

LABEL_MAP = {
    # 'bin': {'0': 0, '1': 1, '2': 1, '3': 1, '4': 1, '5': None},
    # 'tcmr_viral': {'0': None, '1': 0, '2': None, '3': None, '4': 1, '5': None},
    # 'no_viral': {'0': 0, '1': 1, '2': 2, '3': 3, '4': None, '5': None},
    # 'no_other': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': None},
    # 'no_stable': {'0': None, '1': 1, '2': 2, '3': 3, '4': None, '5': None},
    # 'all': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5},
    'rejections': {'0': 'TCMR', '1': 'ABMR', '2': 'Mixed'},
    'norm_rest': {'0': 'Normal', '1': 'Disease'},
    'rej_rest': {'0': 'Rejection', '1': 'Rest'},
    'norm_rej_rest': {'0': 'Normal', '1': 'Rejection', '2': 'Rest'},

}
COLOR_MAP = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


class VisionTransformerLogit():
    def __init__(self, variant='vit_base_patch16_224', pretrained=True):
        self.model = timm.create_model(variant, pretrained=pretrained)
    
    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, pre_logits=True) 
        return x

class FeatureExtractor(pl.LightningDataModule):
    def __init__(self, model_name, n_classes):
        self.n_classes = n_classes
        
        self.model_ft = ResNet.resnet50(num_classes=self.n_classes, mlp=False, two_branch=False, normlinear=True)
        home = Path.cwd().parts[1]
        self.model_ft.load_state_dict(torch.load(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
        # self.model_ft.load_state_dict(torch.load(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
        for param in self.model_ft.parameters():
            param.requires_grad = False
        self.model_ft.fc = nn.Linear(2048, self.out_features)

    def forward(self,x):
        return self.model_ft(x)

class ModelInterface_Classic(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface_Classic, self).__init__()
        self.save_hyperparameters()
        self.n_classes = model.n_classes
        
        # if self.n_classes>2:
        #     self.aucm_loss = AUCM_MultiLabel(num_classes = self.n_classes, device=self.device)
        # else:
        #     self.aucm_loss = CompositionalAUCLoss()
        # self.asl = AsymmetricLossSingleLabel()
        # self.loss = LabelSmoothingCrossEntropy(smoothing=0.1)
        self.loss = create_loss(loss, model.n_classes)

        # self.loss = 
        # print(self.model)
        self.model_name = model.name
        
        
        # self.ecam = EigenGradCAM(model = self.model, target_layers = target_layers, use_cuda=True, reshape_transform=self.reshape_transform)
        self.optimizer = optimizer
        
        self.save_path = kargs['log']
        if Path(self.save_path).parts[3] == 'tcmr':
            temp = list(Path(self.save_path).parts)
            # print(temp)
            temp[3] = 'tcmr_viral'
            self.save_path = '/'.join(temp)

        # if kargs['task']:
        self.task = kargs['task']
        # self.task = Path(self.save_path).parts[3]


        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.data_patient = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        # print(self.experiment)
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(task='multiclass', num_classes = self.n_classes, average=None)
            self.PRC = torchmetrics.PrecisionRecallCurve(task='multiclass', num_classes = self.n_classes)
            self.ROC = torchmetrics.ROC(task='multiclass', num_classes=self.n_classes)
            self.confusion_matrix = torchmetrics.ConfusionMatrix(task='multiclass', num_classes = self.n_classes) 
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='multiclass', num_classes = self.n_classes,
                                                                           average='weighted'),
                                                     torchmetrics.CohenKappa(task='multiclass', num_classes = self.n_classes),
                                                     torchmetrics.F1Score(task='multiclass', num_classes = self.n_classes,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(task='multiclass', average = 'macro',
                                                                         num_classes = self.n_classes),
                                                     torchmetrics.Precision(task='multiclass', average = 'macro',
                                                                            num_classes = self.n_classes),
                                                     torchmetrics.Specificity(task='multiclass', average = 'macro',
                                                                            num_classes = self.n_classes)])
                                                                            
        else : 
            self.AUROC = torchmetrics.AUROC(task='binary')
            # self.AUROC = torchmetrics.AUROC(num_classes=self.n_classes, average = 'weighted')
            self.PRC = torchmetrics.PrecisionRecallCurve(task='binary')
            self.ROC = torchmetrics.ROC(task='binary')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='binary'),
                                                     torchmetrics.CohenKappa(task='binary'),
                                                     torchmetrics.F1Score(task='binary'),
                                                     torchmetrics.Recall(task='binary'),
                                                     torchmetrics.Precision(task='binary')
                                                     ])
            self.confusion_matrix = torchmetrics.ConfusionMatrix(task='binary')                                                                    
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.valid_patient_metrics = metrics.clone(prefix = 'val_patient_')
        self.test_metrics = metrics.clone(prefix = 'test_')
        self.test_patient_metrics = metrics.clone(prefix = 'test_patient')

        #--->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0
        # self.model_name = kargs['backbone']


        if self.model_name == 'features':
            self.model = None
        elif self.model_name == 'inception':
            # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights='Inception_V3_Weights.DEFAULT')
            self.model.aux_logits = False
            ct = 0
            for child in self.model.children():
                ct += 1
                if ct < 15:
                    for parameter in child.parameters():
                        parameter.requires_grad=False
            # for parameter in self.model.parameters():
                # parameter.requires_grad = False

            
            # self.model.AuxLogits.fc = nn.Linear(768, self.n_classes)
            self.model.fc = nn.Linear(self.model.fc.in_features, self.n_classes)
        elif self.model_name == 'resnet18':
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            # modules = list(resnet18.children())[:-1]
            # frozen_layers = 8
            # for child in self.model.children():

            # ct = 0
            # for child in self.model.children():
            #     ct += 1
            #     if ct < 7:
            #         for parameter in child.parameters():
            #             parameter.requires_grad=False
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, self.n_classes),
            )
        elif self.model_name == 'retccl':
            # import models.ResNet as ResNet
            self.model = ResNet.resnet50(num_classes=self.n_classes, mlp=False, two_branch=False, normlinear=True)
            home = Path.cwd().parts[1]
            # pre_model = 
            # self.model.fc = nn.Identity()
            # self.model.load_from_checkpoint(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth', strict=False)
            self.model.load_state_dict(torch.load(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.LayerNorm(1024),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Linear(512, self.n_classes)
            )
        elif self.model_name == 'vit':
            home = Path.cwd().parts[1]
            # self.model = ViT('B_32_imagenet1k', pretrained = True) #vis=vis
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=self.n_classes)
            # model = timm.create_model(“vit_base_patch16_224”, pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = False
            outputs_attrs = self.n_classes
            num_inputs = self.model.head.in_features
            last_layer = nn.Linear(num_inputs, outputs_attrs)
            self.model.head = last_layer
            # model_name_or_path = f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/vit-base-patch16-224/'
            # 'code/models/ckpt/vit-base-patch16-224'
            # self.model = ViTForImageClassification.from_pretrained(
            #     model_name_or_path,
            #     num_labels=self.n_classes,
            #     # id2label={str(i): c for i, c in enumerate(self.n_classes)},
            #     # label2id={c: str(i) for i, c in enumerate(self.n_classes)}
            # )
           # for param in self.model.parameters():
            #     param.requires_grad = False
            # self.model.fc = nn.Linear(self.model.fc.in_features, self.n_classes)
            # print(self.model)
            # input_size = 384

        elif self.model_name == 'resnet50':
        
            # self.model = resnet50_baseline(pretrained=True)
            self.model = models.resnet50(weights='IMAGENET1K_V1')
            # ct = 0
            # for child in self.model.children():
            #     ct += 1
            #     if ct < len(list(self.model.children())) - 2:
            #         for parameter in child.parameters():
            #             parameter.requires_grad=False
            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, self.n_classes),
            )
            
        elif self.model_name == 'efficientnet':
            self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b0', pretrained=True)
            # for param in efficientnet.parameters():
            #     param.requires_grad = False
            # efn = list(efficientnet.children())[:-1]
            self.model.classifier.fc = nn.Linear(1280, self.n_classes)
            # self.model = nn.Sequential(
            #     efficientnet,
            # )
        elif self.model_name == 'simple': #mil-ab attention
            feature_extracting = False
            self.model = nn.Sequential(
                nn.Conv2d(3, 20, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                View((-1, 53*53)),
                nn.Linear(53*53, self.out_features),
                nn.ReLU(),
            )

    # def __build_

    def forward(self, x):
        # print(x.shape)
        if len(x.shape) > 4:
            x = x.squeeze(0)
        # print(x.shape)
        return self.model(x)

    def step(self, input):

        input = input.float()
        # input = input
        # logits, _ = self(input.contiguous()) 

        logits = self(input.contiguous())
        # logits = logits
        # print(F.softmax(logits))
        # print(torch.argmax(logits, dim=0))
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        # Y_hat = torch.argmax(logits, dim=0).unsqueeze(0)
        # Y_prob = F.softmax(logits, dim = 0)

        # print(Y_hat)
        # print(Y_prob)


        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim=1)
        
        return logits, Y_prob, Y_hat

    def training_step(self, batch, batch_idx):

        input, label, _= batch

        # label_filled = torch.full([input.shape[1]], label.item(), device=self.device)

        logits, Y_prob, Y_hat = self.step(input) 

        loss = self.loss(logits, label)
        
        for y, y_hat in zip(label, Y_hat):    
            y = int(y)
            # print(Y_hat)
            self.data[y]["count"] += 1
            self.data[y]["correct"] += (int(y_hat) == y)

        self.log('loss', loss, prog_bar=True, on_epoch=True, logger=True, batch_size=1, sync_dist=True)

        return {'loss': loss, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label} 

    def training_epoch_end(self, training_step_outputs):

        # logits = torch.cat([x['logits'] for x in training_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in training_step_outputs])
        max_probs = torch.cat([x['Y_hat'] for x in training_step_outputs])
        target = torch.cat([x['label'] for x in training_step_outputs])

        # probs = torch.cat([x['Y_prob'] for x in training_step_outputs])
        # probs = torch.stack([x['Y_prob'] for x in training_step_outputs], dim=0)
        # max_probs = torch.stack([x['Y_hat'] for x in training_step_outputs])
        # target = torch.stack([x['label'] for x in training_step_outputs], dim=0).int()

        if self.current_epoch % 5 == 0:
            for c in range(self.n_classes):
                count = self.data[c]["count"]
                correct = self.data[c]["correct"]
                if count == 0: 
                    acc = None
                else:
                    acc = float(correct) / count
                print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        if self.n_classes <=2:
            out_probs = probs[:,1]
        else: out_probs = probs


        if self.current_epoch % 10 == 0:
            self.log_confusion_matrix(out_probs, target.squeeze(), stage='train', comment='slide')

        # print(probs)
        # print(target)
        # print(probs.shape)
        # print(target.shape)
        if self.n_classes <=2:
            out_probs = probs[:,1] 
        self.log('Train/auc', self.AUROC(out_probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):

        input, label, (wsi_name, tile_name, patient) = batch
        # label_filled = torch.full([input.shape[1]], label.item(), device=self.device)
        
        logits, Y_prob, Y_hat = self.step(input) 
        logits = logits.detach()
        Y_prob = Y_prob.detach()
        Y_hat = Y_hat.detach()

        # loss = self.loss(logits, label)
        loss = cross_entropy_torch(logits, label)

        for y, y_hat in zip(label, Y_hat):    
            y = int(y)
            # print(Y_hat)
            self.data[y]["count"] += 1
            self.data[y]["correct"] += (int(y_hat) == y)
        
        # self.data[Y]["correct"] += (Y_hat.item() == Y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
        # print(Y_hat)
        # print(label)
        # self.log('val_aucm_loss', aucm_loss, prog_bar=True, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label, 'name': wsi_name, 'patient': patient, 'tile_name': tile_name, 'loss': loss}


    def validation_epoch_end(self, val_step_outputs):
        
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs])
        max_probs = torch.cat([x['Y_hat'] for x in val_step_outputs])
        target = torch.cat([x['label'] for x in val_step_outputs])
        # slide_names = [list(x['name']) for x in val_step_outputs]
        slide_names = []
        for x in val_step_outputs:
            slide_names += list(x['name'])
        # patients = [list(x['patient']) for x in val_step_outputs]
        patients = []
        for x in val_step_outputs:
            patients += list(x['patient'])
        tile_name = []
        for x in val_step_outputs:
            tile_name += list(x['tile_name'])

        loss = torch.stack([x['loss'] for x in val_step_outputs])

        self.log_dict(self.valid_metrics(max_probs.squeeze(), target.squeeze()),
                          on_epoch = True, logger = True, sync_dist=True)

        if self.n_classes <=2:
            out_probs = probs[:,1] 
        else: out_probs = probs

        self.log_confusion_matrix(out_probs, target, stage='val', comment='slide')
        if len(target.unique()) != 1:
            self.log('val_auc', self.AUROC(out_probs, target).squeeze().mean(), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
            # self.log('val_patient_auc', self.AUROC(patient_score, patient_target), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        else:    
            self.log('val_auc', 0.0, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)



        # self.log_confusion_matrix(max_probs, target.squeeze(), stage='val')

        #----> log per patient metrics
        complete_patient_dict = {}
        patient_list = []            
        patient_score = []      
        patient_target = []

        for p, s, pr, t in zip(patients, slide_names, probs, target):

            if p not in complete_patient_dict.keys():
                complete_patient_dict[p] = {s:[]}
                patient_target.append(t)
                
            elif s not in complete_patient_dict[p].keys():
                complete_patient_dict[p][s] = []
            complete_patient_dict[p][s].append(pr)
            
        for p in complete_patient_dict.keys():
            score = []
            for slide in complete_patient_dict[p].keys():

                slide_score = torch.stack(complete_patient_dict[p][slide])
                if self.n_classes == 2:
                    positive_positions = (slide_score.argmax(dim=1) == 1).nonzero().squeeze()
                    if positive_positions.numel() != 0:
                        slide_score = slide_score[positive_positions]
                if len(slide_score.shape)>1:
                    slide_score = torch.mean(slide_score, dim=0)

                score.append(slide_score)
            score = torch.stack(score)
            if self.n_classes == 2:
                positive_positions = (score.argmax(dim=1) == 1).nonzero().squeeze()
                if positive_positions.numel() != 0:
                    score = score[positive_positions]
            if len(score.shape) > 1:
                score = torch.mean(score, dim=0)
            patient_score.append(score)    

        patient_score = torch.stack(patient_score)
        patient_target = torch.stack(patient_target)
        if self.n_classes <=2:
            patient_score = patient_score[:,1]

        self.log_confusion_matrix(patient_score, patient_target, stage='val', comment='patient')

        self.log_roc_curve(patient_score, patient_target.squeeze(), stage='val', comment='patient')
        self.log_pr_curve(patient_score, patient_target.squeeze(), stage='val', comment='patient')

        
        if len(patient_target.unique()) != 1:
            self.log('val_patient_auc', self.AUROC(patient_score, patient_target.squeeze()).mean(), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        else:    
            self.log('val_patient_auc', 0.0, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        
        self.log_dict(self.valid_patient_metrics(patient_score, patient_target),
                          on_epoch = True, logger = True, sync_dist=True)
        
            

        # precision, recall, thresholds = self.PRC(probs, target)

        

        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('val class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->random, if shuffle data, change seed
        # if self.shuffle == True:
        #     self.count = self.count+1
        #     random.seed(self.count*50)



    def test_step(self, batch, batch_idx):

        input, label, (wsi_name, batch_names, patient) = batch
        label = label.float()
        
        logits, Y_prob, Y_hat = self.step(input) 

        #---->acc log
        # Y = int(label)
        # Y = torch.argmax(label)
        for y, y_hat in zip(label, Y_hat):    
            y = int(y)
            # print(Y_hat)
            self.data[y]["count"] += 1
            self.data[y]["correct"] += (int(y_hat) == y)
        # # print(Y_hat)
        # self.data[Y]["count"] += 1
        # self.data[Y]["correct"] += (int(Y_hat) == Y)
        # self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label, 'name': wsi_name, 'patient': patient, 'tile_name': batch_names}

    def test_epoch_end(self, output_results):

        logits = torch.cat([x['logits'] for x in output_results], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in output_results])
        max_probs = torch.cat([x['Y_hat'] for x in output_results])
        target = torch.cat([x['label'] for x in output_results])
        slide_names = []
        for x in output_results:
            slide_names += list(x['name'])
        patients = []
        for x in output_results:
            patients += list(x['patient'])
        tile_name = []
        # for x in output_results:
        #     tile_name += list(x['tile_name'])

        # logits = torch.cat([x['logits'] for x in output_results], dim = 0)
        # probs = torch.cat([x['Y_prob'] for x in output_results])
        # max_probs = torch.stack([x['Y_hat'] for x in output_results])
        # target = torch.stack([x['label'] for x in output_results]).int()
        # slide_names = [x['name'] for x in output_results]
        # patients = [x['patient'] for x in output_results]
        
        self.log_dict(self.test_metrics(max_probs.squeeze(), target.squeeze()),
                          on_epoch = True, logger = True, sync_dist=True)
        # self.log('test_loss', cross_entropy_torch(logits.squeeze(), target.squeeze()), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

        # if self.n_classes <=2:
        #     out_probs = probs[:,1] 
            # max_probs = max_probs[:,1]


        if self.n_classes <=2:
            out_probs = probs[:,1] 
        else: out_probs = probs

        if len(target.unique()) != 1:
        
            # print(target.shape)
            target = target.long()
            self.log('test_auc', self.AUROC(out_probs, target.squeeze()).mean(), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
            # self.log('val_patient_auc', self.AUROC(patient_score, patient_target), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        else:    
            self.log('test_auc', 0.0, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

        

        #----> log confusion matrix
        self.log_confusion_matrix(out_probs, target, stage='test', comment='slide')

        #----> log per patient metrics
        complete_patient_dict = {}
        patient_list = []            
        patient_score = []      
        patient_target = []

        for p, s, pr, t in zip(patients, slide_names, probs, target):

            if p not in complete_patient_dict.keys():
                complete_patient_dict[p] = {s:[]}
                patient_target.append(t)
                
            elif s not in complete_patient_dict[p].keys():
                complete_patient_dict[p][s] = []
            complete_patient_dict[p][s].append(pr)
            

        for p in complete_patient_dict.keys():
            score = []
            for slide in complete_patient_dict[p].keys():

                slide_score = torch.stack(complete_patient_dict[p][slide])
                if self.n_classes == 2:
                    positive_positions = (slide_score.argmax(dim=1) == 1).nonzero().squeeze()
                    if positive_positions.numel() != 0:
                        slide_score = slide_score[positive_positions]
                if len(slide_score.shape)>1:
                    slide_score = torch.mean(slide_score, dim=0)

                score.append(slide_score)
            score = torch.stack(score)
            if self.n_classes == 2:
                positive_positions = (score.argmax(dim=1) == 1).nonzero().squeeze()
                if positive_positions.numel() != 0:
                    score = score[positive_positions]
            if len(score.shape) > 1:
                score = torch.mean(score, dim=0)
            patient_score.append(score)    
            complete_patient_dict[p]['patient_score'] = score


        # print(complete_patient_dict)
        self.save_results(complete_patient_dict, patient_target)

        opt_threshold = self.load_thresholds(torch.stack(patient_score), torch.stack(patient_target), stage='test', comment='patient')
        
        if self.n_classes > 2:
            opt_threshold = [0.5] * self.n_classes 
        else: 
            opt_threshold = [1-opt_threshold, opt_threshold]
        # print(opt_threshold[1])
        self.log_topk_patients(list(complete_patient_dict.keys()), patient_score, thresh=opt_threshold, stage='test')


        patient_score = torch.stack(patient_score)
        patient_target = torch.stack(patient_target)
        if self.n_classes <=2:
            patient_score = patient_score[:,1]
        
        self.log_confusion_matrix(patient_score, patient_target, stage='test', comment='patient')
        # log roc curve
        self.log_roc_curve(patient_score, patient_target.squeeze(), stage='test', comment='patient')
        # log pr curve
        self.log_pr_curve(patient_score, patient_target.squeeze(), stage='test')

        print('Test Patient AUC: ', self.AUROC(patient_score, patient_target.squeeze()))

        if len(patient_target.unique()) != 1:
            self.log('test_patient_auc', self.AUROC(patient_score, patient_target.squeeze()).mean(), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        else:    
            self.log('test_patient_auc', 0.0, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        
        self.log_dict(self.test_patient_metrics(patient_score, patient_target),
                          on_epoch = True, logger = True, sync_dist=True)
        
            

        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('test class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        

        #---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)

    def configure_optimizers(self):
        # optimizer_ft = optim.Adam(self.model_ft.parameters(), lr=self.optimizer.lr*0.1)
        optimizer = create_optimizer(self.optimizer, self.model)
        # optimizer_aucm = PESG(self.model, loss_fn=self.aucm_loss, lr=self.optimizer.lr, margin=1.0, epoch_decay=2e-3, weight_decay=1e-5, device=self.device)
        # optimizer_aucm = PDSCA(self.model, loss_fn=self.aucm_loss, lr=0.005, margin=1.0, epoch_decay=2e-3, weight_decay=1e-4, beta0=0.9, beta1=0.9, device=self.device)
        # optimizer = PDSCA(self.model, loss_fn=self.loss, lr=self.optimizer.lr, margin=1.0, epoch_decay=2e-3, weight_decay=1e-5, device=self.device)
        # scheduler = {'scheduler': CosineAnnearlingLR(optimizer, mode='min', factor=0.5), 'monitor': 'val_loss', 'frequency': 5}
        scheduler = {'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1), 'monitor': 'val_loss', 'frequency': 5}
        # scheduler_aucm = {'scheduler': CosineAnnealingWarmRestarts(optimizer_aucm, T_0=20)}
        
        # return [optimizer_adam, optimizer_aucm], [scheduler_adam, scheduler_aucm]     
        # return [optimizer_aucm], [scheduler_aucm]     
        return [optimizer], [scheduler]

    # def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
    #     optimizer.zero_grad(set_to_none=True)

    def reshape_transform(self, tensor):
        # print(tensor.shape)
        H = tensor.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        tensor = torch.cat([tensor, tensor[:,:add_length,:]],dim = 1)
        result = tensor[:, :, :].reshape(tensor.size(0), _H, _W, tensor.size(2))
        result = result.transpose(2,3).transpose(1,2)
        # print(result.shape)
        return result

    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if name == 'ViT':
            self.model = ViT

        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
                
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)

        # if backbone == 'retccl':

        #     self.model_ft = ResNet.resnet50(num_classes=self.n_classes, mlp=False, two_branch=False, normlinear=True)
        #     home = Path.cwd().parts[1]
        #     # self.model_ft.fc = nn.Identity()
        #     # self.model_ft.load_from_checkpoint(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth', strict=False)
        #     self.model_ft.load_state_dict(torch.load(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
        #     for param in self.model_ft.parameters():
        #         param.requires_grad = False
        #     self.model_ft.fc = nn.Linear(2048, self.out_features)
        
        # elif backbone == 'resnet50':
        #     self.model_ft = resnet50_baseline(pretrained=True)
        #     for param in self.model_ft.parameters():
        #         param.requires_grad = False

        pass


    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)


        return Model(**args1)

    def log_image(self, tensor, stage, name):
        
        tile = tile.cpu().numpy().transpose(1,2,0)
        tile = (tile - tile.min())/ (tile.max() - tile.min()) * 255
        tile = tile.astype(np.uint8)
        img = Image.fromarray(tile)
        self.loggers[0].experiment.add_figure(f'{stage}/{name}', img, self.current_epoch)

    def get_optimal_operating_point(self, fpr, tpr, thresholds):
        '''
        Returns: 
            optimal_fpr [Tensor]
            optimal_tpr [Tensor]
            optimal_threshold [Float]
        '''

        youden_j = tpr - fpr
        optimal_idx = torch.argmax(youden_j)
        # print(youden_j[optimal_idx])
        optimal_threshold = thresholds[optimal_idx].item()
        # print(optimal_threshold)
        optimal_tpr = tpr[optimal_idx]
        optimal_fpr = fpr[optimal_idx]

        return optimal_fpr, optimal_tpr, optimal_threshold

    def log_topk_patients(self, patient_list, patient_scores, thresh=[], stage='val',  k=10):
        
        for n in range(self.n_classes):
            # p_scores = 
            p_score = [x[n] for x in patient_scores]
            topk_csv_path = f'{self.loggers[0].log_dir}/{stage}_c{n}_top_patients.csv'


            topk_scores, topk_indices = torch.topk(torch.Tensor(p_score), k, dim=0)

            # print(topk_indices)
            # print(patient_list) 
            
            topk_scores = [i for i in topk_scores if i > thresh[n]]
            topk_indices = topk_indices[:len(topk_scores)]
            topk_patients = [patient_list[i] for i in topk_indices]

            topk_df = pd.DataFrame({'Patient': topk_patients, 'Scores': topk_scores})
            topk_df.to_csv(topk_csv_path, index=False)


    def load_thresholds(self, probs, target, stage, comment=''):
        threshold_csv_path = f'{self.loggers[0].log_dir}/val_thresholds.csv'
        if not Path(threshold_csv_path).is_file():
            thresh_df = pd.DataFrame({'slide': [0.5], 'patient': [0.5]})
            thresh_df.to_csv(threshold_csv_path, index=False)

        thresh_df = pd.read_csv(threshold_csv_path)
        if stage != 'test':
            if self.n_classes <= 2:
                fpr_list, tpr_list, thresholds = self.ROC(probs, target)
                optimal_fpr, optimal_tpr, optimal_threshold = self.get_optimal_operating_point(fpr_list, tpr_list, thresholds)
                print(f'Optimal Threshold {stage} {comment}: ', optimal_threshold)
                thresh_df.at[0, comment] =  optimal_threshold
                thresh_df.to_csv(threshold_csv_path, index=False)
            else: 
                optimal_threshold = 0.5
        elif stage == 'test': 
            
            optimal_threshold = thresh_df.at[0, comment]
            print(f'Optimal Threshold {stage} {comment}: ', optimal_threshold)

        return optimal_threshold

    def log_confusion_matrix(self, probs, target, stage, comment=''):
        # confmat = self.confusion_matrix(max_probs, target)
        # print(confmat)
        # df_cm = pd.DataFrame(confmat.cpu().numpy(), index=range(self.n_classes), columns=range(self.n_classes))
        # fig_ = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Spectral').get_figure()
        # if stage == 'train':
        #     self.loggers[0].experiment.add_figure(f'{stage}/Confusion matrix', fig_, self.current_epoch)
        # else:
        #     fig_.savefig(f'{self.loggers[0].log_dir}/cm_{stage}.png', dpi=400)

        # fig_.clf()
        threshold_csv_path = f'{self.loggers[0].log_dir}/val_thresholds.csv'
        if not Path(threshold_csv_path).is_file():
            # thresh_dict = {'index': ['train', 'val'], 'columns': , 'data': [[0.5, 0.5], [0.5, 0.5]]}
            thresh_df = pd.DataFrame({'slide': [0.5], 'patient': [0.5]})
            thresh_df.to_csv(threshold_csv_path, index=False)

        thresh_df = pd.read_csv(threshold_csv_path)
        if stage != 'test':
            if self.n_classes <= 2:
                fpr_list, tpr_list, thresholds = self.ROC(probs, target)
                optimal_fpr, optimal_tpr, optimal_threshold = self.get_optimal_operating_point(fpr_list, tpr_list, thresholds)
                # print(f'Optimal Threshold {stage} {comment}: ', optimal_threshold)
                thresh_df.at[0, comment] =  optimal_threshold
                thresh_df.to_csv(threshold_csv_path, index=False)
            else: 
                optimal_threshold = 0.5
        elif stage == 'test': 
            
            optimal_threshold = thresh_df.at[0, comment]
            print(f'Optimal Threshold {stage} {comment}: ', optimal_threshold)
        if self.n_classes <= 2:
            confmat = confusion_matrix(probs, target, task='binary', threshold=optimal_threshold)
            
        else:
            confmat = confusion_matrix(probs, target, task='multiclass', num_classes=self.n_classes)
        print(stage, comment)
        print(confmat)
        cm_labels = LABEL_MAP[self.task].values()

        fig, ax = plt.subplots()

        # df_cm = pd.DataFrame(confmat.cpu().numpy(), index=range(self.n_classes), columns=range(self.n_classes))
        df_cm = pd.DataFrame(confmat.cpu().numpy(), index=cm_labels, columns=cm_labels)
        # fig_ = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Spectral').get_figure()
        cm_plot = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Spectral')
        
        # cm_plot = 
        if stage == 'train':
            self.loggers[0].experiment.add_figure(f'{stage}/Confusion matrix', cm_plot.figure, self.current_epoch)
            if len(self.loggers) > 2:
                self.loggers[2].log_image(key=f'{stage}/Confusion matrix', images=[cm_plot.figure], caption=[self.current_epoch])
            # self.loggers[0].experiment.add_figure(f'{stage}/Confusion matrix', cm_plot.figure, self.current_epoch)
        else:
            ax.set_title(f'{stage}_{comment}')
            if comment: 
                stage += f'_{comment}'
            # fig_.savefig(f'{self.loggers[0].log_dir}/cm_{stage}.png', dpi=400)
            cm_plot.figure.savefig(f'{self.loggers[0].log_dir}/{stage}_cm.png', dpi=400)

        # fig.clf()
        cm_plot.figure.clf()

    def log_roc_curve(self, probs, target, stage, comment=''):
        
        
        fpr_list, tpr_list, thresholds = self.ROC(probs, target)

        task_label_map = LABEL_MAP[self.task]

        fig, ax = plt.subplots(figsize=(6,6))

        if self.n_classes > 2:
            auroc_score = multiclass_auroc(probs, target, num_classes=self.n_classes, average=None)
            for i in range(len(fpr_list)):

                class_label = task_label_map[str(i)]
                color = COLOR_MAP[i]
                
                fpr = fpr_list[i].cpu().numpy()
                tpr = tpr_list[i].cpu().numpy()
                # ax.plot(fpr, tpr, label=f'class_{i}, AUROC={auroc_score[i]}')
                df = pd.DataFrame(data = {'fpr': fpr, 'tpr': tpr})
                line_plot = sns.lineplot(data=df, x='fpr', y='tpr', label=f'{class_label}={auroc_score[i]:.2f}', legend='full', color=color)
            
        else: 
            auroc_score = binary_auroc(probs, target)
            color = COLOR_MAP[0]
            
            
            # thresholds = thresholds.cpu().numpy()

            optimal_fpr, optimal_tpr, optimal_threshold = self.get_optimal_operating_point(fpr_list, tpr_list, thresholds)
            # 
            fpr = fpr_list.cpu().numpy()
            tpr = tpr_list.cpu().numpy()
            optimal_fpr = optimal_fpr.cpu().numpy()
            optimal_tpr = optimal_tpr.cpu().numpy()
            # youden_j = tpr - fpr
            # optimal_idx = np.argmax(youden_j)
            # # print(youden_j[optimal_idx])
            # # optimal_threshold = thresholds[optimal_idx]
            # optimal_tpr = tpr[optimal_idx]
            # optimal_fpr = fpr[optimal_idx]
            


            df = pd.DataFrame(data = {'fpr': fpr, 'tpr': tpr})
            line_plot = sns.lineplot(data=df, x='fpr', y='tpr', label=f'AUROC={auroc_score:.2f}', legend='full', color=color)
            # ax.plot(fpr, tpr, label=f'AUROC={auroc_score:.2f}', color=color)
            # ax.plot([optimal_fpr, optimal_fpr], [0,1], linestyle='--', color='black', label=f'OOP={optimal_threshold}')
            ax.plot([0, 1], [optimal_tpr, optimal_tpr], linestyle='--', color='black', label=f'OOP={optimal_threshold}')
            ax.plot([optimal_fpr, optimal_fpr], [0, 1], linestyle='--', color='black')
            # ax.plot(fpr, tpr, label=f'AUROC={auroc_score}')
        ax.plot([0,1], [0,1], linestyle='--', color='red')
        
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xlabel('False positive rate (1-specificity)')
        ax.set_ylabel('True positive rate (sensitivity)')
        ax.set_title('ROC curve')
        ax.legend(loc='lower right')
        # plt.savefig(f'{self.loggers[0].log_dir}/roc.jpg')

        if stage == 'train':
            self.loggers[0].experiment.add_figure(f'{stage}/ROC_{stage}', plt, self.current_epoch)
        else:
            # plt.savefig(f'{self.loggers[0].log_dir}/{stage}_roc.png', dpi=400)
            line_plot.figure.savefig(f'{self.loggers[0].log_dir}/{stage}_roc_{comment}.png', dpi=400)
            line_plot.figure.savefig(f'{self.loggers[0].log_dir}/{stage}_roc_{comment}.svg', format='svg')

        line_plot.figure.clf()

    def log_pr_curve(self, probs, target, stage, comment=''):

        # fpr_list, tpr_list, thresholds = self.ROC(probs, target)
        # precision, recall, thresholds = torchmetrics.functional.classification.multiclass_precision_recall_curve(probs, target, num_classes=self.n_classes)
        # print(precision)
        # print(recall)

        # baseline = len(target[target==1]) / len(target)

        # plt.figure(1)
        fig, ax = plt.subplots(figsize=(6,6))
        
        if self.n_classes > 2:

            precision, recall, thresholds = multiclass_precision_recall_curve(probs, target, num_classes=self.n_classes)
            task_label_map = LABEL_MAP[self.task]
            
            for i in range(len(precision)):

                class_label = task_label_map[str(i)]
                color = COLOR_MAP[i]

                re = recall[i]
                pr = precision[i]
                
                partial_auc = _auc_compute(re, pr, 1.0)
                df = pd.DataFrame(data = {'re': re.cpu().numpy(), 'pr': pr.cpu().numpy()})
                line_plot = sns.lineplot(data=df, x='re', y='pr', label=f'{class_label}={partial_auc:.2f}', legend='full', color=color)

                baseline = len(target[target==i]) / len(target)
                ax.plot([0,1],[baseline, baseline], linestyle='--', label=f'Baseline={baseline:.2f}', color=color)

        else: 
            # print(fpr_list)
            color = COLOR_MAP[0]
            precision, recall, thresholds = binary_precision_recall_curve(probs, target)
            baseline = len(target[target==1]) / len(target)
            
            pr = precision
            re = recall
            partial_auc = _auc_compute(re, pr, 1.0)
            # ax.plot(re, pr)
            df = pd.DataFrame(data = {'re': re.cpu().numpy(), 'pr': pr.cpu().numpy()})
            line_plot = sns.lineplot(data=df, x='re', y='pr', label=f'{partial_auc:.2f}', legend='full', color=color)
            
        
            ax.plot([0,1], [baseline, baseline], linestyle='--', label=f'Baseline={baseline:.2f}', color=color)

        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('PR curve')
        ax.legend(loc='lower right')
        # plt.savefig(f'{self.loggers[0].log_dir}/pr_{stage}.jpg')

        if stage == 'train':
            self.loggers[0].experiment.add_figure(f'{stage}/PR_{stage}', fig, self.current_epoch)
        else:
            # fig.savefig(f'{self.loggers[0].log_dir}/pr_{stage}.jpg', dpi=400)
            line_plot.figure.savefig(f'{self.loggers[0].log_dir}/{stage}_pr.png', dpi=400)
            line_plot.figure.savefig(f'{self.loggers[0].log_dir}/{stage}_pr.svg', format='svg')

        line_plot.figure.clf()

    def save_results(self, complete_patient_dict, patient_target):

        patient_output_dict = {'PATIENT': [], 'yTrue': []}
        slide_output_dict = {'SLIDE': [], 'yTrue': []}
        for i in range(self.n_classes):
            # print(LABEL_MAP[self.task])
            class_label = LABEL_MAP[self.task][str(i)]
            # if class_label not in patient_output_dict.keys():
                # patient_output_dict[class_label] = []
            class_scores = [complete_patient_dict[k]['patient_score'][i].cpu().numpy().item() for k in complete_patient_dict.keys()]
            patient_output_dict[class_label] = class_scores
            slide_output_dict[class_label] = []
            
        patient_output_dict['PATIENT'] = list(complete_patient_dict.keys())
        patient_output_dict['yTrue'] = [int(t.cpu().numpy()) for t in patient_target]

        # json.dump(patient_output_dict, open(f'{self.loggers[0].log_dir}/results.json', 'w'))
        out_df = pd.DataFrame.from_dict(patient_output_dict)
        out_df.to_csv(f'{self.loggers[0].log_dir}/TEST_RESULT_PATIENT.csv')

        
        label_mapping = LABEL_MAP[self.task]
    
        for v in label_mapping.values():
            slide_output_dict[v] = []
        for p, t in zip(list(complete_patient_dict.keys()), patient_target):
            # print(complete_patient_dict[p])
            slide_target = label_mapping[str(t.item())]

            for slide in complete_patient_dict[p].keys():
                if slide != 'patient_score':
                    slide_output_dict['SLIDE'].append(slide)
                    slide_output_dict['yTrue'].append(slide_target)
                    
                    # if len(complete_patient_dict[p][slide]) > 1:    
                    slide_score = torch.stack(complete_patient_dict[p][slide]).mean(dim=0).cpu().numpy()
                    # else: slide_score = complete_patient_dict[p][slide][0]
                    # score = score.cpu().numpy()[[]
                    # score = score.cpu().numpy()[[]
                    # print(score)]
                    for j in range(len(slide_score)):
                        class_label = label_mapping[str(j)]
                        slide_output_dict[class_label].append(slide_score[j])
                        
                    


       
        out_df = pd.DataFrame.from_dict(slide_output_dict)
        out_df.to_csv(f'{self.loggers[0].log_dir}/TEST_RESULT_SLIDE.csv')
    

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        out = input.view(*self.shape)
        return out

