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
from torchmetrics.functional.classification import binary_auroc, multiclass_auroc, binary_precision_recall_curve, multiclass_precision_recall_curve
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
        
        self.save_path = kargs['log']
        
        # self.in_features = kargs['in_features']
        # self.out_features = kargs['out_features']
        self.in_features = 2048
        self.out_features = 512
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
        self.data_patient = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        # print(self.experiment)
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(task='multiclass', num_classes = self.n_classes, average='weighted')
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
        self.backbone = kargs['backbone']


        if self.backbone == 'features':
            self.model_ft = None
            
        elif self.backbone == 'dino':
            self.feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/dino-vitb16')
            self.model_ft = ViTModel.from_pretrained('facebook/dino-vitb16')
        # elif self.backbone == 'inception':
        #     self.model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        #     self.model_ft.aux_logits = False
        #     for parameter in self.model_ft.parameters():
        #         parameter.requires_grad = False

        #     self.model_ft.fc = nn.Sequential(nn.Linear(model.fc.in_features, 10),
        #                                     nn.Linear(10, self)
        #     )

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
            for param in self.model_ft.parameters():
                param.requires_grad = False

            
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
        if self.model_ft:
            self.example_input_array = torch.rand([1,1,3,224,224])
        else:
            self.example_input_array = torch.rand([1,1000,self.in_features])
        # print(self.model_ft[0].features[-1])
        # print(self.model_ft)

    # def __build_

    def forward(self, x):
        # print(x.shape)
        if self.model_name == 'AttTrans':
            return self.model(x)
        if self.model_ft:
            # x = x.squeeze(0)
            # if x.dim() == 5:
            batch_size = x.shape[0]
            bag_size = x.shape[1]
            x = x.view(batch_size*bag_size, x.shape[2], x.shape[3], x.shape[4])
            feats = self.model_ft(x).unsqueeze(0)
            # print(feats.shape)
            # print(x.shape)
            # if feats.dim() == 3:
            feats = feats.view(batch_size, bag_size, -1)
        else: 
            feats = x.unsqueeze(0)
        del x
        return self.model(feats)
        # return self.model(x)

    def step(self, input):

        input = input.float()
        # print(input.shape)
        # logits, _ = self(input.contiguous()) 
        logits = self(input.contiguous())
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)


        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat

    def training_step(self, batch):

        input, label, _= batch


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
        self.log('loss', loss, prog_bar=True, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
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


        return {'loss': loss, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label} 

    def training_epoch_end(self, training_step_outputs):

        # for t in training_step_outputs:
        # probs = torch.cat([torch.cat(x[0]['Y_prob'], x[1]['Y_prob']) for x in training_step_outputs])
        # max_probs = torch.stack([torch.stack(x[0]['Y_hat'], x[1]['Y_hat']) for x in training_step_outputs])
        # target = torch.stack([torch.stack(x[0]['label'], x[1]['label']) for x in training_step_outputs])
            # print(t)

        probs = torch.cat([x['Y_prob'] for x in training_step_outputs])
        max_probs = torch.cat([x['Y_hat'] for x in training_step_outputs])
        # print(max_probs)
        target = torch.cat([x['label'] for x in training_step_outputs], dim=0).int()

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
        if self.current_epoch % 10 == 0:
            self.log_confusion_matrix(max_probs, target, stage='train')
        if self.n_classes <=2:
            out_probs = probs[:,1] 
        else: out_probs = probs

        self.log('Train/auc', self.AUROC(out_probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):

        input, label, (wsi_name, patient) = batch
        # label = label.float()
        
        logits, Y_prob, Y_hat = self.step(input) 
        logits = logits.detach()
        Y_prob = Y_prob.detach()
        Y_hat = Y_hat.detach()

        #---->acc log
        # Y = int(label[0][1])
        # Y = torch.argmax(label)
        loss = self.lsce_loss(logits, label)
        # loss = cross_entropy_torch(logits, label)
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
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
        # self.log('val_aucm_loss', aucm_loss, prog_bar=True, on_epoch=True, logger=True, batch_size=1, sync_dist=True)
        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label.int(), 'name': wsi_name, 'patient': patient, 'loss':loss}


    def validation_epoch_end(self, val_step_outputs):

        # print(val_step_outputs)
        # print(torch.cat([x['Y_prob'] for x in val_step_outputs], dim=0))
        # print(torch.stack([x['Y_prob'] for x in val_step_outputs]))
        
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs])
        max_probs = torch.cat([x['Y_hat'] for x in val_step_outputs])
        # print(max_probs)
        target = torch.cat([x['label'] for x in val_step_outputs])
        slide_names = [x['name'] for x in val_step_outputs]
        patients = [x['patient'] for x in val_step_outputs]

        loss = torch.stack([x['loss'] for x in val_step_outputs])
        
        # print(loss)
        # print(loss.mean())
        # print(loss.shape)
        # loss = torch.cat([x['loss'] for x in val_step_outputs])
        # print(loss.shape)
        

        # self.log('val_loss', cross_entropy_torch(logits.squeeze(), target.squeeze()), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        # self.log('val_loss', loss.mean(), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        
        # print(logits)
        # print(target)
        self.log_dict(self.valid_metrics(max_probs.squeeze(), target.squeeze()),
                          on_epoch = True, logger = True, sync_dist=True)
        

        if self.n_classes <=2:
            out_probs = probs[:,1] 
        else: out_probs = probs

        bin_auroc = binary_auroc(out_probs, target.squeeze())
        # print('val_bin_auroc: ', bin_auroc)

        # print(target.unique())
        if len(target.unique()) != 1:
            self.log('val_auc', self.AUROC(out_probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
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
        patient_class_score = 0

        for p, s, pr, t in zip(patients, slide_names, probs, target):
            p = p[0]
            # print(s[0])
            # print(pr)
            if p not in complete_patient_dict.keys():
                complete_patient_dict[p] = {'scores':[(s[0], pr)], 'patient_score': 0}
                # print((s,pr))
                # complete_patient_dict[p]['scores'] = []
                # print(t)
                patient_target.append(t)
            else:
                complete_patient_dict[p]['scores'].append((s[0], pr))

        # print(complete_patient_dict)

        for p in complete_patient_dict.keys():
            # complete_patient_dict[p] = 0
            score = []
            for (slide, probs) in complete_patient_dict[p]['scores']:
                score.append(probs)
            # print(score)
            score = torch.stack(score)
            # print(score)
            if self.n_classes == 2:
                positive_positions = (score.argmax(dim=1) == 1).nonzero().squeeze()
                # print(positive_positions)
                if positive_positions.numel() != 0:
                    score = score[positive_positions]
            if len(score.shape) > 1:
                score = torch.mean(score, dim=0) #.cpu().detach().numpy()

            patient_score.append(score)  
            complete_patient_dict[p]['patient_score'] = score
        correct_patients = []
        false_patients = []

        for patient, label in zip(complete_patient_dict.keys(), patient_target):
            if label == 0:
                p_score =  complete_patient_dict[patient]['patient_score']
                # print(torch.argmax(patient_score))
                if torch.argmax(p_score) == label:
                    correct_patients.append(patient)
                else: 
                    false_patients.append(patient)

        patient_score = torch.stack(patient_score)
        
        if self.n_classes <=2:
            patient_score = patient_score[:,1] 
        patient_target = torch.stack(patient_target)
        # print(patient_target)
        # patient_target = torch.cat(patient_target)
        # self.log_confusion_matrix(max_probs, target, stage='test', comment='patient')
        # print(patient_score.shape)
        # print(patient_target.shape)
        if len(patient_target.shape) >1:
            patient_target = patient_target.squeeze()
        self.log_roc_curve(patient_score, patient_target, stage='val')
        # self.log_roc_curve(patient_score, patient_target.squeeze(), stage='test')

        # if self.current_epoch < 20:
        #     self.log('val_patient_auc', 0.0, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        if len(patient_target.unique()) != 1:
            self.log('val_patient_auc', self.AUROC(patient_score, patient_target.squeeze()), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        else:    
            self.log('val_patient_auc', 0.0, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        
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
            print('val class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)



    def test_step(self, batch, batch_idx):

        input, label, (wsi_name, patient) = batch
        # input, label, (wsi_name, batch_names, patient) = batch
        # label = label.float()
        # 
        logits, Y_prob, Y_hat = self.step(input) 
        loss = self.lsce_loss(logits, label)
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

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label.int(), 'loss': loss, 'name': wsi_name, 'patient': patient}

    def test_epoch_end(self, output_results):
        logits = torch.cat([x['logits'] for x in output_results], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in output_results])
        # max_probs = torch.stack([x['Y_hat'] for x in output_results])
        max_probs = torch.cat([x['Y_hat'] for x in output_results])
        target = torch.cat([x['label'] for x in output_results])
        slide_names = [x['name'] for x in output_results]
        patients = [x['patient'] for x in output_results]
        loss = torch.stack([x['loss'] for x in output_results])
        
        self.log_dict(self.test_metrics(max_probs.squeeze(), target.squeeze()),
                          on_epoch = True, logger = True, sync_dist=True)
        self.log('test_loss', loss.mean(), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)

        if self.n_classes <=2:
            out_probs = probs[:,1] 
        else: out_probs = probs
            # max_probs = max_probs[:,1]

        if len(target.unique()) != 1:
                self.log('test_auc', self.AUROC(out_probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
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
            p = p[0]
            # print(s[0])
            # print(pr)
            if p not in complete_patient_dict.keys():
                complete_patient_dict[p] = {'scores':[(s[0], pr)], 'patient_score': 0}
                # print((s,pr))
                # complete_patient_dict[p]['scores'] = []
                # print(t)
                patient_target.append(t)
            else:
                complete_patient_dict[p]['scores'].append((s[0], pr))

        # print(complete_patient_dict)

        for p in complete_patient_dict.keys():
            # complete_patient_dict[p] = 0
            score = []
            for (slide, probs) in complete_patient_dict[p]['scores']:
                score.append(probs)
            # print(score)
            score = torch.stack(score)
            # print(score)
            if self.n_classes == 2:
                positive_positions = (score.argmax(dim=1) == 1).nonzero().squeeze()
                # print(positive_positions)
                if positive_positions.numel() != 0:
                    score = score[positive_positions]
            if len(score.shape) > 1:
                score = torch.mean(score, dim=0) #.cpu().detach().numpy()

            patient_score.append(score)  
            complete_patient_dict[p]['patient_score'] = score
        correct_patients = []
        false_patients = []

        for patient, label in zip(complete_patient_dict.keys(), patient_target):
            if label == 0:
                p_score =  complete_patient_dict[patient]['patient_score']
                # print(torch.argmax(patient_score))
                if torch.argmax(p_score) == label:
                    correct_patients.append(patient)
                else: 
                    false_patients.append(patient)
        # print('Label 0:')
        # print('Correct Patients: ')
        # print(correct_patients)
        # print('False Patients: ')
        # print(false_patients)

        # print('True positive slides: ')
        # for p in correct_patients: 
        #     print(complete_patient_dict[p]['scores'])
        
        # print('False Negative Slides')
        # for p in false_patients: 
        #     print(complete_patient_dict[p]['scores'])
        
        

        patient_score = torch.stack(patient_score)
        
        # complete_patient_dict[p]['patient_score'] = patient_score

        # print(patient_score)
        if self.n_classes <=2:
            patient_score = patient_score[:,1] 
        patient_target = torch.stack(patient_target)
        # print(patient_target)
        # patient_target = torch.cat(patient_target)
        # self.log_confusion_matrix(max_probs, target, stage='test', comment='patient')
        self.log_roc_curve(patient_score, patient_target.squeeze(), stage='test')

        
        if len(patient_target.unique()) != 1:
            self.log('test_patient_auc', self.AUROC(patient_score, patient_target.squeeze()), prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        else:    
            self.log('test_patient_auc', 0.0, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        
        self.log_dict(self.test_patient_metrics(patient_score, patient_target),
                          on_epoch = True, logger = True, sync_dist=True)
        
        
        self.log_pr_curve(patient_score, patient_target.squeeze(), stage='test')
        
        
        

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
        # optimizer = create_optimizer(self.optimizer, self.model)
        if self.n_classes > 2:
            # optimizer = PESG(self.model, loss_fn=self.aucm_loss, lr=self.optimizer.lr, margin=1.0, epoch_decay=2e-3, weight_decay=1e-5, device=self.device)
            optimizer = create_optimizer(self.optimizer, self.model)
        else:
            # optimizer = PDSCA(self.model, loss_fn=self.loss, lr=0.005, margin=1.0, epoch_decay=2e-3, weight_decay=1e-4, beta0=0.9, beta1=0.9, device=self.device)
            optimizer = create_optimizer(self.optimizer, self.model)
        # optimizer = PDSCA(self.model, loss_fn=self.loss, lr=self.optimizer.lr, margin=1.0, epoch_decay=2e-3, weight_decay=1e-5, device=self.device)
        # scheduler = {'scheduler': CosineAnnearlingLR(optimizer, mode='min', factor=0.5), 'monitor': 'val_loss', 'frequency': 5}
        scheduler = {'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1), 'monitor': 'val_loss', 'frequency': 10}
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

    def init_backbone(self):
        self.backbone = 'retccl'
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
        self.model_ft.to(self.device)

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

    def log_roc_curve(self, probs, target, stage, comment=''):

        fpr_list, tpr_list, thresholds = self.ROC(probs, target)

        # self.AUROC(out_probs, target.squeeze())

        fig, ax = plt.subplots(figsize=(6,6))
        if self.n_classes > 2:
            auroc_score = multiclass_auroc(probs, target.squeeze(), num_classes=self.n_classes, average=None)
            for i in range(len(fpr_list)):
                
                fpr = fpr_list[i].cpu().numpy()
                tpr = tpr_list[i].cpu().numpy()
                ax.plot(fpr, tpr, label=f'class_{i}, AUROC={auroc_score[i]}')
        else: 
            # print(fpr_list)
            auroc_score = binary_auroc(probs, target.squeeze())

            fpr = fpr_list.cpu().numpy()
            tpr = tpr_list.cpu().numpy()

            # df = pd.DataFrame(data = {'fpr': fpr, 'tpr': tpr})
            # line_plot = sns.lineplot(data=df, x='fpr', y='tpr', label=f'AUROC={auroc_score}', legend='full')
            # sfig = line_plot.get_figure()

            ax.plot(fpr, tpr, label=f'AUROC={auroc_score}')


        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title('ROC curve')
        ax.legend(loc='lower right')
        # plt.savefig(f'{self.loggers[0].log_dir}/roc.jpg')

        if stage == 'train':
            self.loggers[0].experiment.add_figure(f'{stage}/ROC_{stage}', plt, self.current_epoch)
        else:
            plt.savefig(f'{self.loggers[0].log_dir}/roc_{stage}.jpg', dpi=400)
            # line_plot.figure.savefig(f'{self.loggers[0].log_dir}/roc_{stage}_sb.jpg')

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
            
            # print(precision)
            # print(recall)
            
            for i in range(len(precision)):
                pr = precision[i].cpu().numpy()
                re = recall[i].cpu().numpy()
                ax.plot(re, pr, label=f'class_{i}')
                baseline = len(target[target==i]) / len(target)
                ax.plot([0,1],[baseline, baseline], linestyle='--', label=f'Baseline_{i}')

        else: 
            # print(fpr_list)
            precision, recall, thresholds = binary_precision_recall_curve(probs, target)
            baseline = len(target[target==1]) / len(target)
            pr = precision.cpu().numpy()
            re = recall.cpu().numpy()
            ax.plot(re, pr)
        
            ax.plot([0,1], [baseline, baseline], linestyle='--', label='Baseline')

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
            fig.savefig(f'{self.loggers[0].log_dir}/pr_{stage}.jpg', dpi=400)

    

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

