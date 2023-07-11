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
plt.style.use('tableau-colorblind10')
import pandas as pd
import cv2
from PIL import Image
from pytorch_pretrained_vit import ViT
# import wandb
from scipy.stats import bootstrap
# import sklearn
# import json
# from pprint import pprint
#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import cross_entropy_torch
from utils.custom_resnet50 import resnet50_baseline

from timm.loss import AsymmetricLossSingleLabel
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
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

LABEL_MAP = {
    # 'bin': {'0': 0, '1': 1, '2': 1, '3': 1, '4': 1, '5': None},
    # 'tcmr_viral': {'0': None, '1': 0, '2': None, '3': None, '4': 1, '5': None},
    # 'no_viral': {'0': 0, '1': 1, '2': 2, '3': 3, '4': None, '5': None},
    'no_other': {'0': 'Normal', '1': 'TCMR', '2': 'ABMR', '3': 'Mixed', '4': 'Viral'},
    # 'no_stable': {'0': None, '1': 1, '2': 2, '3': 3, '4': None, '5': None},
    # 'all': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5},
    'rejections': {'0': 'TCMR', '1': 'ABMR', '2': 'Mixed'},
    'norm_rest': {'0': 'Normal', '1': 'Disease'},
    'rej_rest': {'0': 'Rejection', '1': 'Rest'},
    'rest_rej': {'0': 'Rest', '1': 'Rejection'},
    'norm_rej_rest': {'0': 'Normal', '1': 'Rejection', '2': 'Rest'},

}
COLOR_MAP = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

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

class ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters() #ignore=kargs.keys()
        self.n_classes = model.n_classes
        self.lr = optimizer.lr
        # if 'in_features' in kargs.keys():
        #     self.in_features = kargs['in_features']
        # else: self.in_features = 2048
        # print(self.in_features)
        # print(model.in_features)
        self.in_features = model.in_features
        # self.bag_size = int(kargs['bag_size'])
        if 'bag_size' in kargs.keys():
            self.bag_size = int(kargs['bag_size'])
        else: self.bag_size = 200
        
        if model.name == 'AttTrans':
            self.model = milmodel.MILModel(num_classes=self.n_classes, pretrained=True, mil_mode='att_trans')
        elif model.name == 'vit':
            self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=self.n_classes)
            self.model.patch_embed = nn.Sequential(nn.Linear(self.in_features, 768), nn.Identity())
        elif model.name == 'resnet50':
            self.model = models.resnet50(weights='IMAGENET1K_V1')
            self.model.conv1 = torch.nn.Conv2d(self.in_features, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
            # print(self.model)
            self.model.fc = torch.nn.Sequential(
                torch.nn.Linear(self.model.fc.in_features, self.n_classes),
            )
        else: self.load_model()
        if self.n_classes>2:
            # self.aucm_loss = AUCM_MultiLabel(num_classes = self.n_classes, device=self.device)
            # self.loss = LabelSmoothingCrossEntropy(smoothing=0.1)
            self.loss = create_loss(loss, model.n_classes)
        else:
            # self.loss = CompositionalAUCLoss()
            self.loss = create_loss(loss, model.n_classes)
        # self.asl = AsymmetricLossSingleLabel()
        self.lsce_loss = LabelSmoothingCrossEntropy(smoothing=0.2)

        self.model_name = model.name
        
        self.optimizer = optimizer
        # print(kargs)
        self.save_path = kargs['log']

        
        # # self.out_features = kargs['out_features']
        # self.in_features = 2048
        self.out_features = 512
        if Path(self.save_path).parts[3] == 'tcmr':
            temp = list(Path(self.save_path).parts)
            # print(temp)
            temp[3] = 'tcmr_viral'
            self.save_path = '/'.join(temp)

        # print(kargs['task'])
        if kargs['task']:
            self.task = kargs['task']
        # self.task = Path(self.save_path).parts[3]
        # self.task = kargs['task']
        self.test_slide_OOP = 0
        self.test_patient_OOP = 0


        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.data_patient = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        # print(self.experiment)
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(task='multiclass', num_classes = self.n_classes, average=None)
            self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes = self.n_classes, average='weighted')
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
            self.accuracy = torchmetrics.Accuracy(task='binary')
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
        self.train_metrics = metrics.clone(prefix = 'train_')                                                                
        self.val_metrics = metrics.clone(prefix = 'val_')
        self.val_patient_metrics = metrics.clone(prefix = 'val_patient')
        self.test_metrics = metrics.clone(prefix = 'test_')
        self.test_patient_metrics = metrics.clone(prefix = 'test_patient')

        #--->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0
        self.backbone = kargs['backbone']


        if self.backbone == 'features':
            self.model_ft = None
            
        elif self.backbone == 'resnet18':
            self.model_ft = models.resnet18(weights='IMAGENET1K_V1')
            # modules = list(resnet18.children())[:-1]
            # frozen_layers = 8
            # for child in self.model_ft.children():

            for param in self.model_ft.parameters():
                param.requires_grad = False
            self.model_ft.fc = nn.Linear(512, self.out_features)


        elif self.backbone == 'retccl':
            # import models.ResNet as ResNet
            self.model_ft = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
            home = Path.cwd().parts[1]
            # pre_model = 
            # self.model_ft.fc = nn.Identity()
            # self.model_ft.load_from_checkpoint(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth', strict=False)
            self.model_ft.load_state_dict(torch.load(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
            for param in self.model_ft.parameters():
                param.requires_grad = False
            self.model_ft.fc = torch.nn.Identity()
            # self.model_ft.eval()
            # self.model_ft = FeatureExtractor('retccl', self.n_classes)


        elif self.backbone == 'resnet50':
            
            self.model_ft = resnet50_baseline(pretrained=True)
            # self.model_ft.fc = torch.linear()
            # for param in self.model_ft.parameters():
            #     param.requires_grad = False

            
        elif self.backbone == 'efficientnet':
            efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b0', pretrained=True)
            for param in efficientnet.parameters():
                param.requires_grad = False
            # efn = list(efficientnet.children())[:-1]
            efficientnet.classifier.fc = nn.Linear(1280, self.out_features)
            self.model_ft = nn.Sequential(
                efficientnet,
                nn.GELU(),
            )
        elif self.backbone == 'simple': #mil-ab attention
            self.model_ft = nn.Sequential(
                nn.Conv2d(3, 20, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                View((-1, 53*53*50)),
                nn.Linear(53*53*50, 1024),
                nn.ReLU(),
            )

        # print('Bag_size: ', self.bag_size)
        if self.model_ft:
            self.example_input_array = torch.rand([1,self.bag_size,3,224,224])
        elif self.model_name == 'resnet50' or self.model_name == 'CTMIL':
            self.example_input_array = torch.rand([5,self.in_features,50,50])
        
        else:
            self.example_input_array = torch.rand([1,self.bag_size,self.in_features])
        # self.example_input_array = torch.rand([1,self.bag_size,self.in_features])

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        if self.model_name == 'AttTrans' or self.model_name == 'MonaiMILModel':
            return self.model(x)
        if self.model_ft:
            # x = x.squeeze(0)
            # if x.dim() == 5:
            batch_size = x.shape[0]
            bag_size = x.shape[1]
            x = x.view(batch_size*bag_size, x.shape[2], x.shape[3], x.shape[4])

            feats = self.model_ft(x).unsqueeze(0)
            # print('feats: ', feats.shape)
            # if feats.dim() == 3:
            feats = feats.view(batch_size, bag_size, -1)
        else: 
            feats = x.unsqueeze(0)
        del x
        if self.model_name == 'resnet50':
            feats = feats.squeeze(0)
        return self.model(feats)
        # return self.model(x)

    def step(self, input):

        input = input.float()
        logits = self(input.contiguous())
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        # Y_prob = torch.sigmoid(logits)


        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat

    def training_step(self, batch):

        # print()
        # print(batch)
        # print(len(batch))
        input, label, _ = batch


        logits, Y_prob, Y_hat = self.step(input) 

        #---->loss
        # loss = self.loss(logits, label)

        one_hot_label = torch.nn.functional.one_hot(label, num_classes=self.n_classes)
        loss = self.loss(logits, one_hot_label.float())
        if loss.ndim == 0:
            loss = loss.unsqueeze(0)
        # if self.n_classes > 2: 
        #     aucm_loss = loss
        

        # total_loss = (aucm_loss + loss)/2
        for y, y_hat in zip(label, Y_hat):
            
            y = int(y)
            # print(Y_hat)
            self.data[y]["count"] += 1
            self.data[y]["correct"] += (int(y_hat) == y)


        # self.log('total_loss', total_loss, prog_bar=True, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
        self.log('loss', loss.item(), prog_bar=True, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
        # wandb.log('loss', loss, prog_bar=True, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
        # self.log('lsce_loss', loss, prog_bar=True, on_epoch=True, logger=True, batch_size=1, sync_dist=True)

        # if self.current_epoch % 10 == 0:

        #     # images = input.squeeze()[:10, :, :, :]
        #     # for i in range(10):
        #     img = input.squeeze(0)[:10, :, :, :]
        #     img = (img - torch.min(img)/(torch.max(img)-torch.min(img)))*255.0
            
        #     # mg = img.cpu().numpy()
        #     grid = torchvision.utils.make_grid(img, normalize=True, value_range=(0, 255), scale_each=True)
        #     # grid = img.detach().cpu().numpy()
        # # log input images 
        #     self.loggers[0].experiment.add_image(f'{self.current_epoch}/input', grid)


        self.train_step_outputs.append({'loss': loss, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label}) 
        return loss

    def on_training_epoch_end(self):

        # for t in training_step_outputs:
        # probs = torch.cat([torch.cat(x[0]['Y_prob'], x[1]['Y_prob']) for x in training_step_outputs])
        # max_probs = torch.stack([torch.stack(x[0]['Y_hat'], x[1]['Y_hat']) for x in training_step_outputs])
        # target = torch.stack([torch.stack(x[0]['label'], x[1]['label']) for x in training_step_outputs])
            # print(t)

        probs = torch.cat([x['Y_prob'] for x in self.train_step_outputs])
        max_probs = torch.cat([x['Y_hat'] for x in self.train_step_outputs])
        # print(max_probs)
        target = torch.cat([x['label'] for x in self.train_step_outputs], dim=0).int()

        # logits = torch.cat([x['logits'] for x in training_step_outputs], dim = 0)
        # probs = torch.cat([x['Y_prob'] for x in training_step_outputs])
        # max_probs = torch.stack([x['Y_hat'] for x in training_step_outputs])
        # # target = torch.stack([x['label'] for x in training_step_outputs], dim = 0)
        # target = torch.stack([x['label'] for x in training_step_outputs])
        # target = torch.argmax(target, dim=1)

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

        # print('max_probs: ', max_probs)
        # print('probs: ', probs)
        self.log_dict(self.train_metrics(max_probs.squeeze(), target.squeeze()),
                          on_epoch = True, logger = True, sync_dist=True)

        
        if self.n_classes <=2:
            out_probs = probs[:,1] 
        else: out_probs = probs

        if self.current_epoch % 10 == 0:
            # self.log_confusion_matrix(max_probs, target, stage='train')
            self.log_confusion_matrix(out_probs, target, stage='train', comment='slide')

        self.log('train/auc', self.AUROC(out_probs, target.squeeze()).mean(), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):

        input, label, (wsi_name, _, patient) = batch
        # label = label.float()
        
        logits, Y_prob, Y_hat = self.step(input) 
        logits = logits.detach()
        Y_prob = Y_prob.detach()
        Y_hat = Y_hat.detach()

        #---->acc log
        # Y = int(label[0][1])
        # Y = torch.argmax(label)
        # loss = self.lsce_loss(logits, label)
        loss = cross_entropy_torch(logits, label)
        # one_hot_label = torch.nn.functional.one_hot(label, num_classes=self.n_classes)
        # print(logits)
        # print(label)
        # print(one_hot_label)
        # aucm_loss = self.aucm_loss(logits, one_hot_label.float())
        # if aucm_loss.ndim == 0:
        #     aucm_loss = aucm_loss.unsqueeze(0)
        # print(aucm_loss)
        # loss = self.loss(logits, label)
        # total_loss = (aucm_loss + loss)/2
        # print(loss)

        for y, y_hat in zip(label, Y_hat):
            y = int(y)
            # print(Y_hat)
            self.data[y]["count"] += 1
            self.data[y]["correct"] += (int(y_hat) == y)
        
        # self.data[Y]["correct"] += (Y_hat.item() == Y)
        self.log('val_loss', loss.item(), prog_bar=True, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
        # print(wsi_name)
        # self.log('val_aucm_loss', aucm_loss, prog_bar=True, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
        self.validation_step_outputs.append({'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label.int(), 'name': wsi_name, 'patient': patient, 'loss':loss})


    def on_validation_epoch_end(self):

        logits = torch.cat([x['logits'] for x in self.validation_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in self.validation_step_outputs])
        max_probs = torch.cat([x['Y_hat'] for x in self.validation_step_outputs])
        target = torch.cat([x['label'] for x in self.validation_step_outputs])
        slide_names = [x['name'] for x in self.validation_step_outputs]
        slide_names = [item for sublist in slide_names for item in sublist]
        # slide_names = list(sum(slide_names, ()))
        # slide_names = 
        # for x in self.validation_step_outputs:
        #     print(x['name'])
        patients = [x['patient'] for x in self.validation_step_outputs]
        patients = [item for sublist in patients for item in sublist]
        loss = torch.stack([x['loss'] for x in self.validation_step_outputs])
        
        # if len(max_probs.shape) <2:
        #     max_probs = max_probs.unsqueeze(0).unsqueeze(0)
        #     target = target.unsqueeze(0).unsqueeze(0)
        
        # self.log_dict(self.val_metrics(max_probs.squeeze(0), target.squeeze(0)),
        #                   on_epoch = True, logger = True, sync_dist=True)

        if self.n_classes <=2:
            out_probs = probs[:,1] 
        else: out_probs = probs

        # self.log_confusion_matrix(out_probs, target, stage='val', comment='slide')
        if len(target.unique()) != 1:
            self.log('val_auc', self.AUROC(out_probs, target.squeeze()).mean(), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        else:    
            self.log('val_auc', 0.0, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        

        #----> log confusion matrix
        # print(max_probs, target)
        # self.log_confusion_matrix(max_probs, target, stage='val')
        

        # print(slide_names)
        # print(patients)
        # print(len(patients))
        # print(len(slide_names))
        # print(len(probs))
        # print(len(target))

        #----> log per patient metrics
        complete_patient_dict = {}
        patient_list = []            
        patient_score = []      
        patient_target = []
        patient_class_score = 0

        for p, s, pr, t in zip(patients, slide_names, probs, target):
            # p = p[0]
            # print(p)
            if p not in complete_patient_dict.keys():
                complete_patient_dict[p] = {'scores':[(s[0], pr)], 'patient_score': 0}
                patient_target.append(t)
            else:
                complete_patient_dict[p]['scores'].append((s[0], pr))

        for p in complete_patient_dict.keys():
            score = []
            for (slide, probs) in complete_patient_dict[p]['scores']:
                score.append(probs)
            score = torch.stack(score)
            if self.n_classes == 2:
                positive_positions = (score.argmax(dim=1) == 1).nonzero().squeeze()
                if positive_positions.numel() != 0:
                    score = score[positive_positions]
            if len(score.shape) > 1:
                score = torch.mean(score, dim=0) #.cpu().detach().numpy()

            patient_score.append(score)  
            complete_patient_dict[p]['patient_score'] = score

        # print(complete_patient_dict)
        # correct_patients = []
        # false_patients = []

        # for patient, label in zip(complete_patient_dict.keys(), patient_target):
        #     # if label == 0:
        #     p_score =  complete_patient_dict[patient]['patient_score']
        #     # print(torch.argmax(patient_score))
        #     if torch.argmax(p_score) == label:
        #         correct_patients.append(patient)
        #     else: 
        #         false_patients.append(patient)

        patient_score = torch.stack(patient_score)
        patient_target = torch.stack(patient_target)
        
        if self.n_classes <=2:
            patient_score = patient_score[:,1] 

        # self.log_confusion_matrix(patient_score, patient_target, stage='val', comment='patient')
        
        # print(patient_score)
        # print(patient_target)
        # print(patient_target.squeeze())
        # print(self.AUROC(patient_score, patient_target.squeeze()))
        
        # print('patient_score: ', patient_score)
        # print('patient_target: ', patient_target)
        

        # self.log_roc_curve(patient_score, patient_target.squeeze(), stage='val', comment='patient')
        # self.log_pr_curve(patient_score, patient_target.squeeze(), stage='val', comment='patient')

        if len(patient_target.unique()) != 1:
            self.log('val_patient_auc', self.AUROC(patient_score, patient_target.squeeze()).mean(), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        else:    
            self.log('val_patient_auc', 0.0, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        
        self.log_dict(self.val_patient_metrics(patient_score, patient_target),
                          on_epoch = True, logger = True, sync_dist=True)

        self.log('val_accuracy', self.accuracy(patient_score, patient_target), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
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
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)

        self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):

        input, label, (wsi_name, batch_names, patient) = batch
        # 
        # get attention ##
        # input = input.float()
        # logits = self(input.contiguous())
        # logits, attn = self(input.contiguous())
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = torch.sigmoid(logits)

        # norm_result = self.reshape_transform(h)
        # print(norm_result)

        # print('input: ', input.shape)
        # print('attn: ', attn)
        # print('attn: ', attn.softmax(dim=1))
        # for i in range(attn.shape[1]):
        
        #     attn_mask = attn[0, i, ...].cpu().numpy()
        # #     print(attn_mask)
        #     attn_mask = (attn_mask - attn_mask.min())/(attn_mask.max() - attn_mask.min())*255
        #     attn_mask = attn_mask.astype(np.uint8)
        #     # print(attn_mask.shape)
        #     # print(attn_mask)
            

        #     # mask = Image.fromarray(attn_mask)
        #     cv2.imwrite(f'/home/ylan/workspace/TransMIL-DeepGraft/test/attention_maps_3/{wsi_name[0]}_{i}.png', attn_mask)
        #     # mask = mask.convert('RGB')
            # mask.save(f'/home/ylan/workspace/TransMIL-DeepGraft/test/attention_maps_2/{i}.png')

            # print(attn_mask)
        ##################
        logits, Y_prob, Y_hat = self.step(input) # Use this for standard inference.

        loss = self.loss(logits, label)
        #---->acc log
        # Y = int(label)
        for y, y_hat in zip(label, Y_hat):
            
            y = int(y)
            # print(Y_hat)
            self.data[y]["count"] += 1
            self.data[y]["correct"] += (int(y_hat) == y)

        # Y = torch.argmax(label)

        # # print(Y_hat)
        # self.data[Y]["count"] += 1
        # self.data[Y]["correct"] += (int(Y_hat) == Y)
        # self.data[Y]["correct"] += (Y_hat.item() == Y)
        step_output_dict = {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label.int(), 'loss': loss, 'name': wsi_name, 'patient': patient}
        self.test_step_outputs.append(step_output_dict)

    # def test_epoch_end(self, output_results):
    def on_test_epoch_end(self):

        logits = torch.cat([x['logits'] for x in self.test_step_outputs], dim = 0)
        # probs = torch.cat([x['Y_prob'] for x in self.test_step_outputs])
        probs = F.softmax(logits, dim = 1)
        # Y_hat = torch.argmax(logits, dim=1)

        # max_probs = torch.cat([x['Y_hat'] for x in self.test_step_outputs])
        max_probs = torch.argmax(logits, dim=1)


        target = torch.cat([x['label'] for x in self.test_step_outputs])
        # slide_names = [x['name'] for x in self.test_step_outputs]
        # patients = [x['patient'] for x in self.test_step_outputs]
        slide_names = [x['name'] for x in self.test_step_outputs]
        slide_names = [item for sublist in slide_names for item in sublist]
        patients = [x['patient'] for x in self.test_step_outputs]
        patients = [item for sublist in patients for item in sublist]
        loss = torch.stack([x['loss'] for x in self.test_step_outputs])


        self.log_dict(self.test_metrics(max_probs.squeeze(), target.squeeze()),
                          on_epoch = True, logger = True, sync_dist=True)
        self.log('test_loss', loss.mean(), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

        if self.n_classes <=2:
            out_probs = probs[:,1] 
        else: out_probs = probs

        if len(target.unique()) != 1:
                self.log('test_auc', self.AUROC(out_probs, target.squeeze()).mean(), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        else:    
            self.log('test_auc', 0.0, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        print('Test Slide AUROC: ', self.AUROC(out_probs, target.squeeze()).mean())

        #----> log confusion matrix
        self.log_confusion_matrix(out_probs, target, stage='test', comment='slide')

        #----> log per patient metrics
        complete_patient_dict = {}
        patient_list = []            
        patient_score = []      
        patient_target = []
        
        patient_class_score = 0

        for p, s, pr, t in zip(patients, slide_names, probs, target):
            # p = p[0]
            if p not in complete_patient_dict.keys():
                complete_patient_dict[p] = {'scores':[(s, pr)], 'patient_score': 0}
                patient_target.append(t)
            else:
                complete_patient_dict[p]['scores'].append((s, pr))

        for p in complete_patient_dict.keys():
            score = []
            for (slide, probs) in complete_patient_dict[p]['scores']:
                score.append(probs)
            
            score = torch.stack(score)
            # print(score)
            if self.n_classes <= 2:
                positive_positions = (score.argmax(dim=1) == 1).nonzero().squeeze()
                if positive_positions.numel() != 0:
                    score = score[positive_positions]
            # elif self.n_classes > 2: 
            #     positive_positions = (score.argmax(dim=1) > 0).nonzero().squeeze()
            #     if positive_positions.numel() == 1:
            #         score = score[positive_positions]
            #     else: 

            #         values, indices = score[positive_positions].max(dim=1)
            #         values = values.squeeze().argmax()
            #         score = score[positive_positions[]]
            # positive_positions = (score.argmax(dim=1) > 0).nonzero().squeeze()
            # if positive_positions.numel() != 0:
            #     score = score[positive_positions]

                
            if len(score.shape) > 1:
                
                # print('before: ', score)
                score = torch.mean(score, dim=0) #.cpu().detach().numpy()
                # print('after: ', score)

            patient_score.append(score)  
            

            complete_patient_dict[p]['patient_score'] = score

        

        
        self.save_results(complete_patient_dict, patient_target)
        
        

        opt_threshold = self.load_thresholds(torch.stack(patient_score), torch.stack(patient_target), stage='test', comment='patient')
        # print(opt_threshold)
        if self.n_classes > 2:
            opt_threshold = [0.5] * self.n_classes 
        else: 
            opt_threshold = [1-opt_threshold, opt_threshold]
        # print(opt_threshold[1])
        # self.log_topk_patients(complete_patient_dict, patient_target, thresh=opt_threshold, stage='test')
        self.log_topk_patients(list(complete_patient_dict.keys()), patient_score, patient_target, thresh=opt_threshold, stage='test')


        # get topk patients
        # correct_patients = []
        # false_patients = []


        # for patient, label in zip(complete_patient_dict.keys(), patient_target):
        #     if label == 0:
        #         p_score =  complete_patient_dict[patient]['patient_score']
        #         # print(torch.argmax(patient_score))
        #         if torch.argmax(p_score) == label:
        #             correct_patients.append((patient, p_score))
        #         else: 
        #             false_patients.append((patient, p_score))
        
        # print('correct_patients: ')
        # print(correct_patients)
        # print('false_patients: ')
        # print(false_patients)

        


        patient_score = torch.stack(patient_score)
        patient_target = torch.stack(patient_target)
        # max_patient_score = torch.argmax(patient_score, dim=1)
        if self.n_classes <=2:
            patient_score = patient_score[:,1] 

        
        self.log_confusion_matrix(patient_score, patient_target, stage='test', comment='patient')
        # log roc curve

        # print(patient_score.shape)
        # print(patient_target.shape)
        self.log_roc_curve(patient_score, patient_target.squeeze(), stage='test', comment='patient')
        # log pr curve
        self.log_pr_curve(patient_score, patient_target.squeeze(), stage='test')


        # print(patient_score)
        
        

        # res = bootstrap((patient_score.cpu().numpy(), patient_target.cpu().numpy()), sklearn.metrics.roc_auc_score, confidence_level=0.95, paired=True, vectorized=False)

        # print('bootstrap AUC: ', res)
        # patient_AUC = self.AUROC(patient_score, patient_target.squeeze())



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

        self.test_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        # optimizer_ft = optim.Adam(self.model_ft.parameters(), lr=self.optimizer.lr*0.1)
        # optimizer = create_optimizer(self.optimizer, self.model)
        if self.n_classes > 2:
            # optimizer = PESG(self.model, loss_fn=self.aucm_loss, lr=self.optimizer.lr, margin=1.0, epoch_decay=2e-3, weight_decay=1e-5, device=self.device)
            optimizer = create_optimizer(self.optimizer, self.model)
        else:
            # optimizer = PDSCA(self.model, loss_fn=self.loss, lr=0.005, margin=1.0, epoch_decay=2e-3, weight_decay=1e-4, beta0=0.9, beta1=0.9, device=self.device)
            optimizer = create_optimizer(self.optimizer, self.model)
        # optimizer = PDSCA(self.model, loss_fn=self.loss, lr=self.optimizer.lr, margin=1.0, epoch_decay=2e-3, weight_decay=1e-5, device=self.device)
        # scheduler = {'scheduler': CosineAnnearlingLR(optimizer, mode='min', factor=0.5), 'monitor': 'val_loss', 'frequency': 5}
        scheduler = {'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5), 'monitor': 'val_loss', 'frequency': 10}
        # scheduler_aucm = {'scheduler': CosineAnnealingWarmRestarts(optimizer_aucm, T_0=20)}
        
        # return [optimizer_adam, optimizer_aucm], [scheduler_adam, scheduler_aucm]     
        return [optimizer], [scheduler]     
        # return optimizer_aucm
        # return [optimizer], [scheduler]

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

    
    def save_results(self, complete_patient_dict, patient_target):

        # print(complete_patient_dict)
        label_mapping = LABEL_MAP[self.task]


        patient_output_dict = {'PATIENT': [], 'yTrue': []}
        for i in range(self.n_classes):
            # print(LABEL_MAP[self.task])
            class_label = LABEL_MAP[self.task][str(i)]
            # if class_label not in patient_output_dict.keys():
                # patient_output_dict[class_label] = []
            class_scores = [complete_patient_dict[k]['patient_score'][i].cpu().numpy().item() for k in complete_patient_dict.keys()]
            patient_output_dict[class_label] = class_scores
            
        patient_output_dict['PATIENT'] = list(complete_patient_dict.keys())
        patient_output_dict['yTrue'] = [int(t.cpu().numpy()) for t in patient_target]

        # json.dump(patient_output_dict, open(f'{self.loggers[0].log_dir}/results.json', 'w'))
        out_df = pd.DataFrame.from_dict(patient_output_dict)
        out_df.to_csv(f'{self.loggers[0].log_dir}/TEST_RESULT_PATIENT.csv')

        slide_output_dict = {'SLIDE': [], 'yTrue': []}
        
    
        for v in label_mapping.values():
            slide_output_dict[v] = []
        # print(slide_output_dict)
        for p, t in zip(list(complete_patient_dict.keys()), patient_target):
            # print(complete_patient_dict[p])
            # target_label = label_mapping[str(t.item())]?

            # print(complete_patient_dict[p])
            for i in complete_patient_dict[p]['scores']:
                # pprint(complete_patient_dict[p][i])
                # for x in complete_patient_dict[p][i]:
                # print(i)
                slide_name, score = i

                score = score.cpu().numpy()
                slide_output_dict['SLIDE'].append(slide_name)
                # print(score)
                for j in range(len(score)):
                    class_label = label_mapping[str(j)]
                    slide_output_dict[class_label].append(score[j])
                    # print(score[j])

                slide_output_dict['yTrue'].append(t.item())

        out_df = pd.DataFrame.from_dict(slide_output_dict)
        out_df.to_csv(f'{self.loggers[0].log_dir}/TEST_RESULT_SLIDE.csv')


        # slide_output_dict['SLIDE'] = list(complete_patient_dict)

        # slide_output_dict = {'Slide': [], 'yPred': [], 'yTrue': []}

        # for patient in complete_patient_dict: 
            



    def log_image(self, tensor, stage, name):
        
        tile = tile.cpu().numpy().transpose(1,2,0)
        tile = (tile - tile.min())/ (tile.max() - tile.min()) * 255
        tile = tile.astype(np.uint8)
        img = Image.fromarray(tile)
        self.loggers[0].experiment.add_figure(f'{stage}/{name}', img, self.current_epoch)
        # self.logger.log_image(key=f'{stage}/{name}', images=img, caption=self.current_epoch)

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

    def log_topk_patients(self, patient_list, patient_scores, patient_target, thresh=[], stage='val',  k=5):
        
        # patient_target = np.array([i.item() for i in patient_target])
        patient_target = torch.Tensor(patient_target)
        patient_list = np.array(patient_list)

        patient_scores = np.array([i.cpu().numpy() for i in patient_scores])
        patient_scores = torch.Tensor(patient_scores)

        for n in range(self.n_classes):

            n_patients = patient_list[patient_target == n]
            n_scores = [s[n] for s in patient_scores[patient_target == n]]
            # print(n_patients)

            topk_csv_path = f'{self.loggers[0].log_dir}/{stage}_c{n}_top_patients.csv'

            topk_scores, topk_indices = torch.topk(torch.Tensor(n_scores), k, dim=0)

        #     # print(topk_indices)
        #     # print(patient_list) 
            
            topk_scores = [i for i in topk_scores if i > thresh[n]]
            topk_indices = topk_indices[:len(topk_scores)]
            topk_patients = [n_patients[i] for i in topk_indices]

            topk_df = pd.DataFrame({'Patient': topk_patients, 'Scores': topk_scores})
            topk_df.to_csv(topk_csv_path, index=False)


    def load_thresholds(self, probs, target, stage, comment=''):
        threshold_csv_path = f'{self.loggers[0].log_dir}/val_thresholds.csv'
        optimal_threshold = 1/self.n_classes
        if not Path(threshold_csv_path).is_file():
            
            thresh_df = pd.DataFrame({'slide': [optimal_threshold], 'patient': [optimal_threshold]})
            thresh_df.to_csv(threshold_csv_path, index=False)

        thresh_df = pd.read_csv(threshold_csv_path)
        if stage != 'test':
            if self.n_classes <= 2:
                fpr_list, tpr_list, thresholds = self.ROC(probs, target)
                optimal_fpr, optimal_tpr, optimal_threshold = self.get_optimal_operating_point(fpr_list, tpr_list, thresholds)
                print(f'Optimal Threshold {stage} {comment}: ', optimal_threshold)
                
            else: 
                optimal_threshold = 1/self.n_classes
            # thresh_df.at[0, comment] =  optimal_threshold
            # thresh_df.to_csv(threshold_csv_path, index=False)

        elif stage == 'test': 
            optimal_threshold = thresh_df.at[0, comment]
            print(f'Optimal Threshold {stage} {comment}: ', optimal_threshold)

        return optimal_threshold

    def log_confusion_matrix(self, probs, target, stage, comment=''): # threshold

        # preds = torch.argmax(probs, dim=1)
        # if self.n_classes <= 2:
        #     probs = probs[:,1] 

        # read threshold file
        threshold_csv_path = f'{self.loggers[0].log_dir}/val_thresholds.csv'
        # print(self.loggers[0].log_dir)
        # print(threshold_csv_path)
        
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
                # fpr_list, tpr_list, thresholds = multiclass_roc(probs, target)
                # optimal_fpr, optimal_tpr, optimal_threshold = self.get_optimal_operating_point(fpr_list, tpr_list, thresholds)

                optimal_threshold = 1/self.n_classes
        elif stage == 'test': 
            
            optimal_threshold = thresh_df.at[0, comment]
            print(f'Optimal Threshold {stage} {comment}: ', optimal_threshold)
            # optimal_threshold = 0.5 # manually change to val_optimal_threshold for testing

        # print(confmat)
        # confmat = self.confusion_matrix(preds, target, threshold=optimal_threshold)
        if self.n_classes <= 2:
            confmat = confusion_matrix(probs, target, task='binary', threshold=optimal_threshold)
            
        else:
            confmat = confusion_matrix(probs, target, task='multiclass', num_classes=self.n_classes)
        # print(stage, comment)
        # print(confmat)
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

        # self.AUROC(out_probs, target.squeeze())

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
            line_plot.figure.savefig(f'{self.loggers[0].log_dir}/{stage}_{comment}_roc.png', dpi=400)
            line_plot.figure.savefig(f'{self.loggers[0].log_dir}/{stage}_{comment}_roc.svg', format='svg')

        line_plot.figure.clf()
        # fig.clf()
        


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

class RETCCL_FE(pl.LightningModule):
    def __init__(self):
        super(RETCCL_FE, self).__init__()
        self.model_ft = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
        home = Path.cwd().parts[1]
        self.model_ft.load_state_dict(torch.load(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
        for param in self.model_ft.parameters():
            param.requires_grad = False
        self.model_ft.fc = torch.nn.Identity()
    
    def forward(self, x):
        return self.model_ft(x)

    
