import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt

#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import cross_entropy_torch
from timm.loss import AsymmetricLossSingleLabel
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy

#---->
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch import optim as optim

#---->
import pytorch_lightning as pl
from .vision_transformer import vit_small
from torchvision import models
from torchvision.models import resnet
from transformers import AutoFeatureExtractor, ViTModel

from captum.attr import LayerGradCam

class ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = create_loss(loss)
        # self.asl = AsymmetricLossSingleLabel()
        # self.loss = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        # self.loss = 
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']

        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        # print(self.experiment)
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(num_classes = self.n_classes, average = 'weighted')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = self.n_classes,
                                                                           average='micro'),
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
            self.AUROC = torchmetrics.AUROC(num_classes=2, average = 'weighted')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes = 2,
                                                                           average = 'micro'),
                                                     torchmetrics.CohenKappa(num_classes = 2),
                                                     torchmetrics.F1Score(num_classes = 2,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(average = 'macro',
                                                                         num_classes = 2),
                                                     torchmetrics.Precision(average = 'macro',
                                                                            num_classes = 2)])
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes = self.n_classes)                                                                    
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        #--->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0
        self.backbone = kargs['backbone']

        self.out_features = 512
        if kargs['backbone'] == 'dino':
            self.feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/dino-vitb16')
            self.model_ft = ViTModel.from_pretrained('facebook/dino-vitb16')
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
                # nn.GELU()
            )
        elif kargs['backbone'] == 'efficientnet':
            efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_widese_b0', pretrained=True)
            for param in efficientnet.parameters():
                param.requires_grad = False
            # efn = list(efficientnet.children())[:-1]
            efficientnet.classifier.fc = nn.Linear(1280, self.out_features)
            self.model_ft = nn.Sequential(
                efficientnet,
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

    def training_step(self, batch, batch_idx):
        #---->inference
        
        data, label, _ = batch
        label = label.float()
        data = data.squeeze(0).float()
        # print(data)
        # print(data.shape)
        if self.backbone == 'dino':
            features = self.model_ft(**data)
            features = features.last_hidden_state
        else:
            features = self.model_ft(data)
        features = features.unsqueeze(0)
        # print(features.shape)
        # features = features.squeeze()
        results_dict = self.model(data=features) 
        # results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->loss
        loss = self.loss(logits, label)
        # loss = self.asl(logits, label.squeeze())

        #---->acc log
        # print(label)
        # Y_hat = int(Y_hat)
        # if self.n_classes == 2:
        #     Y = int(label[0][1])
        # else: 
        Y = torch.argmax(label)
            # Y = int(label[0])
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (int(Y_hat) == Y)

        return {'loss': loss, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label} 

    def training_epoch_end(self, training_step_outputs):
        # logits = torch.cat([x['logits'] for x in training_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in training_step_outputs])
        max_probs = torch.stack([x['Y_hat'] for x in training_step_outputs])
        # target = torch.stack([x['label'] for x in training_step_outputs], dim = 0)
        target = torch.cat([x['label'] for x in training_step_outputs])
        target = torch.argmax(target, dim=1)
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
            self.log_confusion_matrix(probs, target, stage='train')

        self.log('Train/auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx):

        data, label, _ = batch

        label = label.float()
        data = data.squeeze(0).float()
        features = self.model_ft(data)
        features = features.unsqueeze(0)

        results_dict = self.model(data=features)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']


        #---->acc log
        # Y = int(label[0][1])
        Y = torch.argmax(label)

        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}


    def validation_epoch_end(self, val_step_outputs):
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        # probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs])
        max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
        # target = torch.stack([x['label'] for x in val_step_outputs], dim = 0)
        target = torch.cat([x['label'] for x in val_step_outputs])
        target = torch.argmax(target, dim=1)
        #---->
        # logits = logits.long()
        # target = target.squeeze().long()
        # logits = logits.squeeze(0)
        self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
        self.log('val_auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)

        # print(max_probs.squeeze(0).shape)
        # print(target.shape)
        self.log_dict(self.valid_metrics(max_probs.squeeze() , target),
                          on_epoch = True, logger = True)

        #----> log confusion matrix
        self.log_confusion_matrix(probs, target, stage='val')
        

        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->random, if shuffle data, change seed
        if self.shuffle == True:
            self.count = self.count+1
            random.seed(self.count*50)

    def test_step(self, batch, batch_idx):

        data, label, _ = batch
        label = label.float()
        data = data.squeeze(0).float()
        features = self.model_ft(data)
        features = features.unsqueeze(0)

        results_dict = self.model(data=features, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        #---->acc log
        Y = torch.argmax(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    def test_epoch_end(self, output_results):
        probs = torch.cat([x['Y_prob'] for x in output_results])
        max_probs = torch.stack([x['Y_hat'] for x in output_results])
        # target = torch.stack([x['label'] for x in output_results], dim = 0)
        target = torch.cat([x['label'] for x in output_results])
        target = torch.argmax(target, dim=1)
        
        #---->
        auc = self.AUROC(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze() , target)


        # metrics = self.test_metrics(max_probs.squeeze() , torch.argmax(target.squeeze(), dim=1))
        metrics['test_auc'] = auc

        # self.log('auc', auc, prog_bar=True, on_epoch=True, logger=True)

        # print(max_probs.squeeze(0).shape)
        # print(target.shape)
        # self.log_dict(metrics, logger = True)
        for keys, values in metrics.items():
            print(f'{keys} = {values}')
            metrics[keys] = values.cpu().numpy()
        #---->acc log
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            if count == 0: 
                acc = None
            else:
                acc = float(correct) / count
            print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

        self.log_confusion_matrix(probs, target, stage='test')
        #---->
        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path / 'result.csv')

    def configure_optimizers(self):
        # optimizer_ft = optim.Adam(self.model_ft.parameters(), lr=self.optimizer.lr*0.1)
        optimizer = create_optimizer(self.optimizer, self.model)
        return optimizer     


    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
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

    def log_confusion_matrix(self, max_probs, target, stage):
        confmat = self.confusion_matrix(max_probs.squeeze(), target)
        df_cm = pd.DataFrame(confmat.cpu().numpy(), index=range(self.n_classes), columns=range(self.n_classes))
        plt.figure()
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        # plt.close(fig_)
        # plt.savefig(f'{self.log_path}/cm_e{self.current_epoch}')
        self.loggers[0].experiment.add_figure(f'{stage}/Confusion matrix', fig_, self.current_epoch)

        if stage == 'test':
            plt.savefig(f'{self.log_path}/cm_test')
        plt.close(fig_)
        # self.logger[0].experiment.add_figure('Confusion matrix', fig_, self.current_epoch)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        # batch_size = input.size(0)
        # shape = (batch_size, *self.shape)
        out = input.view(*self.shape)
        return out

