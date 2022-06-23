import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
import cv2
from PIL import Image

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
from torchmetrics.functional import stat_scores
from torch import optim as optim
# from sklearn.metrics import roc_curve, auc, roc_curve_score


#---->
import pytorch_lightning as pl
from .vision_transformer import vit_small
from torchvision import models
from torchvision.models import resnet
from transformers import AutoFeatureExtractor, ViTModel

from pytorch_grad_cam import GradCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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
        # print(self.model)
        
        
        # self.ecam = EigenGradCAM(model = self.model, target_layers = target_layers, use_cuda=True, reshape_transform=self.reshape_transform)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        print(self.n_classes)
        self.save_path = kargs['log']
        if Path(self.save_path).parts[3] == 'tcmr':
            temp = list(Path(self.save_path).parts)
            # print(temp)
            temp[3] = 'tcmr_viral'
            self.save_path = '/'.join(temp)

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
        self.PRC = torchmetrics.PrecisionRecallCurve(num_classes = self.n_classes)
        # self.pr_curve = torchmetrics.BinnedPrecisionRecallCurve(num_classes = self.n_classes, thresholds=10)
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
        # print(self.model_ft[0].features[-1])
        # print(self.model_ft)
        if model.name == 'TransMIL':
            target_layers = [self.model.layer2.norm] # 32x32
            # target_layers = [self.model_ft[0].features[-1]] # 32x32
            self.cam = GradCAM(model=self.model, target_layers = target_layers, use_cuda=True, reshape_transform=self.reshape_transform) #, reshape_transform=self.reshape_transform
            # self.cam_ft = GradCAM(model=self.model, target_layers = target_layers_ft, use_cuda=True) #, reshape_transform=self.reshape_transform
        else:
            target_layers = [self.model.attention_weights]
            self.cam = GradCAM(model = self.model, target_layers = target_layers, use_cuda=True)

    def forward(self, x):
        
        feats = self.model_ft(x).unsqueeze(0)
        return self.model(feats)

    def step(self, input):

        input = input.squeeze(0).float()
        logits = self(input) 

        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        return logits, Y_prob, Y_hat

    def training_step(self, batch, batch_idx):
        #---->inference
        

        input, label, _= batch
        label = label.float()
        
        logits, Y_prob, Y_hat = self.step(input) 

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
        self.log('loss', loss, prog_bar=True, on_epoch=True, logger=True)

        if self.current_epoch % 10 == 0:

            grid = torchvision.utils.make_grid(images)
        # log input images 
        # self.loggers[0].experiment.add_figure(f'{stage}/input', , self.current_epoch)


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

        input, label, _ = batch
        label = label.float()
        
        logits, Y_prob, Y_hat = self.step(input) 

        #---->acc log
        # Y = int(label[0][1])
        Y = torch.argmax(label)

        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}


    def validation_epoch_end(self, val_step_outputs):
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs])
        max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
        target = torch.cat([x['label'] for x in val_step_outputs])
        target = torch.argmax(target, dim=1)
        #---->
        # logits = logits.long()
        # target = target.squeeze().long()
        # logits = logits.squeeze(0)
        if len(target.unique()) != 1:
            self.log('val_auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
        else:    
            self.log('val_auc', 0.0, prog_bar=True, on_epoch=True, logger=True)

        

        self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
        

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
        torch.set_grad_enabled(True)
        data, label, name = batch
        label = label.float()
        # logits, Y_prob, Y_hat = self.step(data) 
        # print(data.shape)
        data = data.squeeze(0).float()
        logits = self(data).detach() 

        Y = torch.argmax(label)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        
        #----> Get Topk tiles 

        target = [ClassifierOutputTarget(Y)]

        data_ft = self.model_ft(data).unsqueeze(0).float()
        # data_ft = self.model_ft(data).unsqueeze(0).float()
        # print(data_ft.shape)
        # print(target)
        grayscale_cam = self.cam(input_tensor=data_ft, targets=target)
        # grayscale_ecam = self.ecam(input_tensor=data_ft, targets=target)

        # print(grayscale_cam)

        summed = torch.mean(torch.Tensor(grayscale_cam), dim=2)
        print(summed)
        print(summed.shape)
        topk_tiles, topk_indices = torch.topk(summed.squeeze(0), 5, dim=0)
        topk_data = data[topk_indices].detach()
        
        # target_ft = 
        # grayscale_cam_ft = self.cam_ft(input_tensor=data, )
        # for i in range(data.shape[0]):
            
            # vis_img = data[i, :, :, :].cpu().numpy()
            # vis_img = np.transpose(vis_img, (1,2,0))
            # print(vis_img.shape)
            # cam_img = grayscale_cam.squeeze(0)
        # cam_img = self.reshape_transform(grayscale_cam)

        # print(cam_img.shape)
            
            # visualization = show_cam_on_image(vis_img, cam_img, use_rgb=True)
            # visualization = ((visualization/visualization.max())*255.0).astype(np.uint8)
            # print(visualization)
        # cv2.imwrite(f'{test_path}/{Y}/{name}/gradcam.jpg', cam_img)

        #---->acc log
        Y = torch.argmax(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label, 'name': name, 'topk_data': topk_data} #
        # return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label, 'name': name} #, 'topk_data': topk_data

    def test_epoch_end(self, output_results):
        probs = torch.cat([x['Y_prob'] for x in output_results])
        max_probs = torch.stack([x['Y_hat'] for x in output_results])
        # target = torch.stack([x['label'] for x in output_results], dim = 0)
        target = torch.cat([x['label'] for x in output_results])
        target = torch.argmax(target, dim=1)
        patients = [x['name'] for x in output_results]
        topk_tiles = [x['topk_data'] for x in output_results]
        #---->
        auc = self.AUROC(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze() , target)


        # metrics = self.test_metrics(max_probs.squeeze() , torch.argmax(target.squeeze(), dim=1))
        metrics['test_auc'] = auc

        # self.log('auc', auc, prog_bar=True, on_epoch=True, logger=True)

        #---->get highest scoring patients for each class
        test_path = Path(self.save_path) / 'most_predictive'
        topk, topk_indices = torch.topk(probs.squeeze(0), 5, dim=0)
        for n in range(self.n_classes):
            print('class: ', n)
            topk_patients = [patients[i[n]] for i in topk_indices]
            topk_patient_tiles = [topk_tiles[i[n]] for i in topk_indices]
            for x, p, t in zip(topk, topk_patients, topk_patient_tiles):
                print(p, x[n])
                patient = p[0]
                outpath = test_path / str(n) / patient 
                outpath.mkdir(parents=True, exist_ok=True)
                for i in range(len(t)):
                    tile = t[i]
                    tile = tile.cpu().numpy().transpose(1,2,0)
                    tile = (tile - tile.min())/ (tile.max() - tile.min()) * 255
                    tile = tile.astype(np.uint8)
                    img = Image.fromarray(tile)
                    
                    img.save(f'{test_path}/{n}/{patient}/{i}_gradcam.jpg')

            
            
        #----->visualize top predictive tiles
        
        

        
                # img = img.squeeze(0).cpu().numpy()
                # img = np.transpose(img, (1,2,0))
                # # print(img)
                # # print(grayscale_cam.shape)
                # visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)


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

        #---->plot auroc curve
        # stats = stat_scores(probs, target, reduce='macro', num_classes=self.n_classes)
        # fpr = {}
        # tpr = {}
        # for n in self.n_classes: 

        # fpr, tpr, thresh = roc_curve(target.cpu().numpy(), probs.cpu().numpy())
        #[tp, fp, tn, fn, tp+fn]


        self.log_confusion_matrix(probs, target, stage='test')
        #---->
        result = pd.DataFrame([metrics])
        result.to_csv(Path(self.save_path) / f'test_result.csv', mode='a', header=not Path(self.save_path).exists())

        # with open(f'{self.save_path}/test_metrics.txt', 'a') as f:

        #     f.write([metrics])

    def configure_optimizers(self):
        # optimizer_ft = optim.Adam(self.model_ft.parameters(), lr=self.optimizer.lr*0.1)
        optimizer = create_optimizer(self.optimizer, self.model)
        return optimizer     

    def reshape_transform(self, tensor, h=32, w=32):
        result = tensor[:, 1:, :].reshape(tensor.size(0), h, w, tensor.size(2))
        result = result.transpose(2,3).transpose(1,2)
        # print(result.shape)
        return result

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

    def log_image(self, tensor, stage, name):
        
        tile = tile.cpu().numpy().transpose(1,2,0)
        tile = (tile - tile.min())/ (tile.max() - tile.min()) * 255
        tile = tile.astype(np.uint8)
        img = Image.fromarray(tile)
        self.loggers[0].experiment.add_figure(f'{stage}/{name}', img, self.current_epoch)


    def log_confusion_matrix(self, max_probs, target, stage):
        confmat = self.confusion_matrix(max_probs.squeeze(), target)
        print(confmat)
        df_cm = pd.DataFrame(confmat.cpu().numpy(), index=range(self.n_classes), columns=range(self.n_classes))
        # plt.figure()
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        # plt.close(fig_)
        # plt.savefig(f'{self.save_path}/cm_e{self.current_epoch}')
        

        if stage == 'train':
            # print(self.save_path)
            # plt.savefig(f'{self.save_path}/cm_test')

            self.loggers[0].experiment.add_figure(f'{stage}/Confusion matrix', fig_, self.current_epoch)
        else:
            fig_.savefig(f'{self.save_path}/cm_test.png', dpi=400)
        # plt.close(fig_)
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

