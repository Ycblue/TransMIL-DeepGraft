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
from torch import optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
        self.save_hyperparameters()
        self.n_classes = model.n_classes
        
        if model.name == 'AttTrans':
            self.model = milmodel.MILModel(num_classes=self.n_classes, pretrained=True, mil_mode='att_trans', backbone_num_features=1024)
        else: self.load_model()
        # self.loss = create_loss(loss, model.n_classes)
        # self.loss = 
        if self.n_classes>2:
            self.aucm_loss = AUCM_MultiLabel(num_classes = model.n_classes, device=self.device)
        else:
            self.aucm_loss = AUCMLoss()
        # self.asl = AsymmetricLossSingleLabel()
        self.loss = LabelSmoothingCrossEntropy(smoothing=0.1)

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
        #     self.task = kargs['task']
        self.task = Path(self.save_path).parts[3]


        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        # print(self.experiment)
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(num_classes = self.n_classes, average='macro')
            
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = self.n_classes,
                                                                           average='weighted'),
                                                     torchmetrics.CohenKappa(num_classes = self.n_classes),
                                                     torchmetrics.F1Score(num_classes = self.n_classes,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = self.n_classes),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = self.n_classes),
                                                     torchmetrics.Specificity(average = 'macro',
                                                                            num_classes = self.n_classes)])
                                                                            
        else : 
            self.AUROC = torchmetrics.AUROC(num_classes=self.n_classes, average='weighted')
            # self.AUROC = torchmetrics.AUROC(num_classes=self.n_classes, average = 'weighted')

            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = 2,
                                                                           average = 'weighted'),
                                                     torchmetrics.CohenKappa(num_classes = 2),
                                                     torchmetrics.F1Score(num_classes = 2,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = 2),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = 2)])
        self.PRC = torchmetrics.PrecisionRecallCurve(num_classes = self.n_classes)
        self.ROC = torchmetrics.ROC(num_classes=self.n_classes)
        # self.pr_curve = torchmetrics.BinnedPrecisionRecallCurve(num_classes = self.n_classes, thresholds=10)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes = self.n_classes)                                                                    
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.valid_patient_metrics = metrics.clone(prefix = 'val_patient_')
        self.test_metrics = metrics.clone(prefix = 'test_')
        self.test_patient_metrics = metrics.clone(prefix = 'test_patient')

        #--->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0
        self.backbone = kargs['backbone']

        self.out_features = 1024

        if self.backbone == 'features':
            self.model_ft = None
        elif self.backbone == 'dino':
            self.feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/dino-vitb16')
            self.model_ft = ViTModel.from_pretrained('facebook/dino-vitb16')
        elif self.backbone == 'resnet18':
            self.model_ft = models.resnet18(weights='IMAGENET1K_V1')
            # modules = list(resnet18.children())[:-1]
            # frozen_layers = 8
            # for child in self.model_ft.children():

            for param in self.model_ft.parameters():
                param.requires_grad = False
            self.model_ft.fc = nn.Linear(512, self.out_features)


            # res18 = nn.Sequential(
            #     *modules,
            # )
            # for param in res18.parameters():
            #     param.requires_grad = False
            # self.model_ft = nn.Sequential(
            #     res18,
            #     nn.AdaptiveAvgPool2d(1),
            #     View((-1, 512)),
            #     nn.Linear(512, self.out_features),
            #     nn.GELU(),
            # )
        elif self.backbone == 'retccl':
            # import models.ResNet as ResNet
            self.model_ft = ResNet.resnet50(num_classes=self.n_classes, mlp=False, two_branch=False, normlinear=True)
            home = Path.cwd().parts[1]
            # pre_model = 
            # self.model_ft.fc = nn.Identity()
            # self.model_ft.load_from_checkpoint(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth', strict=False)
            self.model_ft.load_state_dict(torch.load(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
            for param in self.model_ft.parameters():
                param.requires_grad = False
            self.model_ft.fc = nn.Linear(2048, self.out_features)
            
            # self.model_ft = FeatureExtractor('retccl', self.n_classes)


        elif self.backbone == 'resnet50':
            
            self.model_ft = resnet50_baseline(pretrained=True)
            for param in self.model_ft.parameters():
                param.requires_grad = False

            # self.model_ft = models.resnet50(pretrained=True)
            # for param in self.model_ft.parameters():
            #     param.requires_grad = False
            # self.model_ft.fc = nn.Linear(2048, self.out_features)


            # modules = list(resnet50.children())[:-3]
            # res50 = nn.Sequential(
            #     *modules,     
            # )
            
            # self.model_ft = nn.Sequential(
            #     res50,
            #     nn.AdaptiveAvgPool2d(1),
            #     View((-1, 1024)),
            #     nn.Linear(1024, self.out_features),
            #     # nn.GELU()
            # )
        # elif kargs
            
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
            feature_extracting = False
            self.model_ft = nn.Sequential(
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
        # print(self.model_ft[0].features[-1])
        # print(self.model_ft)

    # def __build_

    def forward(self, x):
        # print(x.shape)
        if self.model_name == 'AttTrans':
            return self.model(x)
        if self.model_ft:
            x = x.squeeze(0)
            feats = self.model_ft(x).unsqueeze(0)
        else: 
            feats = x.unsqueeze(0)
        
        return self.model(feats)
        # return self.model(x)

    def step(self, input):

        input = input.float()
        # logits, _ = self(input.contiguous()) 
        logits = self(input.contiguous())
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)


        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat

    def training_step(self, batch, batch_idx):

        input, label, _= batch

        #random image dropout

        # bag_size = input.squeeze().shape[0] * 0.7
        # bag_idxs = torch.randperm(input.squeeze(0).shape[0])[:bag_size]
        # input = input.squeeze(0)[bag_idxs].unsqueeze(0)

        # label = label.float()
        
        logits, Y_prob, Y_hat = self.step(input) 

        #---->loss
        loss = self.loss(logits, label)

        one_hot_label = torch.nn.functional.one_hot(label, num_classes=self.n_classes)
        # aucm_loss = self.aucm_loss(torch.sigmoid(logits), one_hot_label)
        # total_loss = torch.mean(loss + aucm_loss)
        Y = int(label)
        # print(logits, label)
        # loss = cross_entropy_torch(logits.squeeze(0), label)
        # loss = self.asl(logits, label.squeeze())

        #---->acc log
        # print(label)
        # Y_hat = int(Y_hat)
        # if self.n_classes == 2:
        #     Y = int(label[0][1])
        # else: 
        # Y = torch.argmax(label)
        
            # Y = int(label[0])
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (int(Y_hat) == Y)
        # self.log('total_loss', total_loss, prog_bar=True, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
        # self.log('aucm_loss', aucm_loss, prog_bar=True, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
        self.log('lsce_loss', loss, prog_bar=True, on_epoch=True, logger=True, batch_size=1, sync_dist=True)

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


        return {'loss': loss, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label} 

    def training_epoch_end(self, training_step_outputs):
        # logits = torch.cat([x['logits'] for x in training_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in training_step_outputs])
        max_probs = torch.stack([x['Y_hat'] for x in training_step_outputs])
        # target = torch.stack([x['label'] for x in training_step_outputs], dim = 0)
        target = torch.stack([x['label'] for x in training_step_outputs])
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
        if self.current_epoch % 10 == 0:
            self.log_confusion_matrix(max_probs, target, stage='train')

        self.log('Train/auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):

        input, label, (wsi_name, batch_names, patient) = batch
        # label = label.float()
        
        logits, Y_prob, Y_hat = self.step(input) 

        #---->acc log
        # Y = int(label[0][1])
        # Y = torch.argmax(label)
        loss = self.loss(logits, label)
        # loss = self.loss(logits, label)
        # print(loss)
        Y = int(label)

        # print(Y_hat)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (int(Y_hat) == Y)
        
        # self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label, 'name': wsi_name, 'patient': patient, 'loss':loss}


    def validation_epoch_end(self, val_step_outputs):

        # print(val_step_outputs)
        # print(torch.cat([x['Y_prob'] for x in val_step_outputs], dim=0))
        # print(torch.stack([x['Y_prob'] for x in val_step_outputs]))
        
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs])
        max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
        target = torch.stack([x['label'] for x in val_step_outputs], dim=0).int()
        slide_names = [x['name'] for x in val_step_outputs]
        patients = [x['patient'] for x in val_step_outputs]

        loss = torch.stack([x['loss'] for x in val_step_outputs])
        # loss = torch.cat([x['loss'] for x in val_step_outputs])
        # print(loss.shape)
        

        # self.log('val_loss', cross_entropy_torch(logits.squeeze(), target.squeeze()), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        
        # print(logits)
        # print(target)
        self.log_dict(self.valid_metrics(max_probs.squeeze(), target.squeeze()),
                          on_epoch = True, logger = True, sync_dist=True)
        

        if len(target.unique()) != 1:
            self.log('val_auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
            # self.log('val_patient_auc', self.AUROC(patient_score, patient_target), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        else:    
            self.log('val_auc', 0.0, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)



        # print(max_probs.squeeze(0).shape)
        # print(target.shape)
        

        #----> log confusion matrix
        self.log_confusion_matrix(max_probs, target, stage='val')

        #----> log per patient metrics
        complete_patient_dict = {}
        patient_list = []            
        patient_score = []      
        patient_target = []

        for p, s, pr, t in zip(patients, slide_names, probs, target):
            if p not in complete_patient_dict.keys():
                complete_patient_dict[p] = [(s, pr)]
                patient_target.append(t)
            else:
                complete_patient_dict[p].append((s, pr))

       

        for p in complete_patient_dict.keys():
            score = []
            for (slide, probs) in complete_patient_dict[p]:
                # max_probs = torch.argmax(probs)
                # if self.n_classes == 2:
                #     score.append(max_probs)
                # else: score.append(probs)
                score.append(probs)

            # if self.n_classes == 2:
                # score =
            score = torch.mean(torch.stack(score), dim=0) #.cpu().detach().numpy()
            # complete_patient_dict[p]['score'] = score
            # print(p, score)
            # patient_list.append(p)    
            patient_score.append(score)    

        patient_score = torch.stack(patient_score)
        # print(patient_target)
        # print(torch.cat(patient_target))
        # print(self.AUROC(patient_score.squeeze(), torch.cat(patient_target)))

        
        patient_target = torch.cat(patient_target)

        # print(patient_score.shape)
        # print(patient_target.shape)
        
        if len(patient_target.unique()) != 1:
            self.log('val_patient_auc', self.AUROC(patient_score.squeeze(), patient_target.squeeze()), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
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
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)



    def test_step(self, batch, batch_idx):

        input, label, (wsi_name, batch_names, patient) = batch
        label = label.float()
        
        logits, Y_prob, Y_hat = self.step(input) 

        #---->acc log
        Y = int(label)
        # Y = torch.argmax(label)

        # print(Y_hat)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (int(Y_hat) == Y)
        # self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label, 'name': wsi_name, 'patient': patient}

    def test_epoch_end(self, output_results):
        logits = torch.cat([x['logits'] for x in output_results], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in output_results])
        max_probs = torch.stack([x['Y_hat'] for x in output_results])
        target = torch.stack([x['label'] for x in output_results]).int()
        slide_names = [x['name'] for x in output_results]
        patients = [x['patient'] for x in output_results]
        
        self.log_dict(self.test_metrics(max_probs.squeeze(), target.squeeze()),
                          on_epoch = True, logger = True, sync_dist=True)
        self.log('test_loss', cross_entropy_torch(logits.squeeze(), target.squeeze()), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

        if len(target.unique()) != 1:
            self.log('test_auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
            # self.log('val_patient_auc', self.AUROC(patient_score, patient_target), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        else:    
            self.log('test_auc', 0.0, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)



        #----> log confusion matrix
        self.log_confusion_matrix(max_probs, target, stage='test')

        #----> log per patient metrics
        complete_patient_dict = {}
        patient_list = []            
        patient_score = []      
        patient_target = []
        patient_class_score = 0

        for p, s, pr, t in zip(patients, slide_names, probs, target):
            if p not in complete_patient_dict.keys():
                complete_patient_dict[p] = [(s, pr)]
                patient_target.append(t)
            else:
                complete_patient_dict[p].append((s, pr))

       

        for p in complete_patient_dict.keys():
            score = []
            for (slide, probs) in complete_patient_dict[p]:
                # if self.n_classes == 2:
                #     if probs.argmax().item() == 1: # only if binary and if class 1 is more important!!! Normal vs Diseased or Rejection vs Other
                #         score.append(probs)
                    
                # else: 
                score.append(probs)
            # print(score)
            score = torch.stack(score)
            # print(score)
            if self.n_classes == 2:
                positive_positions = (score.argmax(dim=1) == 1).nonzero().squeeze()
                if positive_positions.numel() != 0:
                    score = score[positive_positions]
            else:
            # score = torch.stack(torch.score)
            ## get scores that predict class 1:
            # positive_scores = score.argmax(dim=1)
            # score = torch.sum(score.argmax(dim=1))

            # if score.item() == 1:
            #     patient_class_score = 1
                score = torch.mean(score) #.cpu().detach().numpy()
            # complete_patient_dict[p]['score'] = score
            # print(p, score)
            # patient_list.append(p)    
            patient_score.append(score)    

        print(patient_score)

        patient_score = torch.stack(patient_score)
        # patient_target = torch.stack(patient_target)
        patient_target = torch.cat(patient_target)

        
        if len(patient_target.unique()) != 1:
            self.log('test_patient_auc', self.AUROC(patient_score.squeeze(), patient_target.squeeze()), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        else:    
            self.log('test_patient_auc', 0.0, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        
        self.log_dict(self.test_patient_metrics(patient_score, patient_target),
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
            print('test class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)

    def configure_optimizers(self):
        # optimizer_ft = optim.Adam(self.model_ft.parameters(), lr=self.optimizer.lr*0.1)
        optimizer = create_optimizer(self.optimizer, self.model)
        # optimizer = PESG(self.model, loss_fn=self.aucm_loss, lr=self.optimizer.lr, margin=1.0, epoch_decay=2e-3, weight_decay=1e-5, device=self.device)
        # optimizer = PDSCA(self.model, loss_fn=self.loss, lr=self.optimizer.lr, margin=1.0, epoch_decay=2e-3, weight_decay=1e-5, device=self.device)
        scheduler = {'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5), 'monitor': 'val_loss', 'frequency': 5}
        
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


    def log_confusion_matrix(self, max_probs, target, stage):
        confmat = self.confusion_matrix(max_probs, target)
        print(confmat)
        df_cm = pd.DataFrame(confmat.cpu().numpy(), index=range(self.n_classes), columns=range(self.n_classes))
        fig_ = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Spectral').get_figure()
        if stage == 'train':
            self.loggers[0].experiment.add_figure(f'{stage}/Confusion matrix', fig_, self.current_epoch)
        else:
            fig_.savefig(f'{self.loggers[0].log_dir}/cm_{stage}.png', dpi=400)

        fig_.clf()

    def log_roc_curve(self, probs, target, stage):

        fpr_list, tpr_list, thresholds = self.ROC(probs, target)

        plt.figure(1)
        if self.n_classes > 2:
            for i in range(len(fpr_list)):
                fpr = fpr_list[i].cpu().numpy()
                tpr = tpr_list[i].cpu().numpy()
                plt.plot(fpr, tpr, label=f'class_{i}')
        else: 
            print(fpr_list)
            fpr = fpr_list.cpu().numpy()
            tpr = tpr_list.cpu().numpy()
            plt.plot(fpr, tpr)
        
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.savefig(f'{self.loggers[0].log_dir}/roc.jpg')

        if stage == 'train':
            self.loggers[0].experiment.add_figure(f'{stage}/ROC', plt, self.current_epoch)
        else:
            plt.savefig(f'{self.loggers[0].log_dir}/roc.jpg', dpi=400)

    

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

