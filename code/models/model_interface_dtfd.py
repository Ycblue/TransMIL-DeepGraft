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
import torchvision
from torchvision import models
from torchvision.models import resnet
from transformers import AutoFeatureExtractor, ViTModel

from pytorch_grad_cam import GradCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from captum.attr import LayerGradCam
from models.DTFDMIL import Attention_Gated, Classifier_1fc, DimReduction, Attention_with_Classifier

class ModelInterface_DTFD(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface_DTFD, self).__init__()
        self.save_hyperparameters()
        # self.load_model()
        self.loss = create_loss(loss)
        # self.asl = AsymmetricLossSingleLabel()
        # self.loss = LabelSmoothingCrossEntropy(smoothing=0.1)
        # self.loss = 
        # print(self.model)
        self.model_name = model.name
        
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.save_path = kargs['log']
        if Path(self.save_path).parts[3] == 'tcmr':
            temp = list(Path(self.save_path).parts)
            temp[3] = 'tcmr_viral'
            self.save_path = '/'.join(temp)

        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
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
            self.AUROC = torchmetrics.AUROC(num_classes=self.n_classes, average = 'weighted')

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
        self.ROC = torchmetrics.ROC(num_classes=self.n_classes)
        # self.pr_curve = torchmetrics.BinnedPrecisionRecallCurve(num_classes = self.n_classes, thresholds=10)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes = self.n_classes)                                                                    
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        #--->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0
        self.backbone = kargs['backbone']

        self.out_features = 1024
        if kargs['backbone'] == 'dino':
            self.feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/dino-vitb16')
            self.model_ft = ViTModel.from_pretrained('facebook/dino-vitb16')
        elif kargs['backbone'] == 'resnet18':
            self.model_ft = models.resnet18(pretrained=True)
            # modules = list(resnet18.children())[:-1]
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
        elif kargs['backbone'] == 'resnet50':

            self.model_ft = models.resnet50(pretrained=True)    
            for param in self.model_ft.parameters():
                param.requires_grad = False
            self.model_ft.fc = nn.Linear(2048, self.out_features)

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
        self.classifier = Classifier_1fc(n_channels=512, n_classes=self.n_classes)
        self.attention = Attention_Gated(features=512)
        self.dimreduction = DimReduction(n_channels=self.out_features, m_dim=512)
        self.attCls = Attention_with_Classifier(L=512, num_cls=self.n_classes)
        self.trainable_parameters = []
        self.trainable_parameters += list(self.classifier.parameters())
        self.trainable_parameters += list(self.attention.parameters())
        self.trainable_parameters += list(self.dimreduction.parameters())
        
        # print(self.model_ft[0].features[-1])
        # print(self.model_ft)

    def forward(self, x, bag_size=120):
        # print(x.shape)
        x = x.float()
        max_pseudo_bags = x.squeeze(0).shape[0] // bag_size
        max_pseudo_bags = min(8, max_pseudo_bags)
        
        slide_pseudo_feat = []
        sub_predictions = []

        input = x.squeeze(0)
        features = self.model_ft(input) # max_pseudo_bags, 512
        features = self.dimreduction(features)
        randomized_idx = torch.randperm(features.shape[0])

        
        for n in range(max_pseudo_bags):

            bag_idxs = randomized_idx[bag_size*n:bag_size*(n+1)] #torch.randperm(x.squeeze(0).shape[0])
            bag_features = features.squeeze(0)[bag_idxs]
            
            t1AA = self.attention(bag_features).squeeze(0)
            # print('features: ', features.shape)
            # print('t1AA: ', t1AA.shape)
            t1attFeats = torch.einsum('ns, n->ns', bag_features, t1AA)
            # print('t1attFeats: ', t1attFeats.shape)
            t1attFeats_tensor = torch.sum(t1attFeats, dim=0).unsqueeze(0)
            # print('t1attFeats_tensor: ', t1attFeats_tensor.shape)
            t1Predict = self.classifier(t1attFeats_tensor)
            sub_predictions.append(t1Predict)

            patch_pred_logits = get_cam_1d(self.classifier, t1attFeats.unsqueeze(0)).squeeze(0)
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            # patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            af_inst_feat = t1attFeats_tensor
            slide_pseudo_feat.append(af_inst_feat)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)

        ## optimization for first tier
        sub_predictions = torch.cat(sub_predictions, dim=0)

        ## optimization for second tier
        slide_prediction = self.attCls(slide_pseudo_feat)

        Y_hat = torch.argmax(slide_prediction, dim=1)
        Y_prob = F.softmax(slide_prediction, dim=1)

        
        
        return sub_predictions, slide_prediction, Y_prob, Y_hat


    def training_step(self, batch, batch_idx, optimizer_idx):

        input, label, _= batch


        sub_predictions, slide_prediction, Y_prob, Y_hat = self(input)

        # print(sub_predictions.size(0))
        label = label.float()
        sub_labels = [label] * sub_predictions.size(0)         
        sub_labels = torch.cat(sub_labels, dim=0)
        
        
        sub_loss = self.loss(sub_predictions, sub_labels)
        slide_loss = self.loss(slide_prediction, label)

        
        Y = torch.argmax(label)
            # Y = int(label[0])
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (int(Y_hat) == Y)
        self.log('sub_loss', sub_loss, prog_bar=True, on_epoch=True, logger=True, batch_size=1)
        self.log('slide_loss', slide_loss, prog_bar=True, on_epoch=True, logger=True, batch_size=1)

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


        # print(Y_prob)
        # print(Y_prob.shape)
        
        total_loss = (sub_loss + slide_loss)/2
        # print(sub_predictions)
        # print(sub_labels)
        # sub_probs = sub_predictions
        # sub_targets = torch.argmax(sub_labels, dim=1)
        # if len(sub_targets.unique()) != 1:
        #     self.log('Train/sub_auc', self.AUROC(sub_predictions, sub_targets), prog_bar=True, on_epoch=True, logger=True)

        # else:    
        #     self.log('Train/sub_auc', 0.0, prog_bar=True, on_epoch=True, logger=True)

        return {'loss': total_loss, 'Y_prob': Y_prob.detach(), 'Y_hat': Y_hat.detach(), 'label': Y} 

    def training_epoch_end(self, training_step_outputs):
        # print(training_step_outputs)
        # for x in training_step_outputs:
        #     print(x)
            # print(x['Y_prob'])
        # logits = torch.cat([x['logits'] for x in training_step_outputs], dim = 0)
        probs = torch.cat([x[0]['Y_prob'] for x in training_step_outputs])
        max_probs = torch.stack([x[0]['Y_hat'] for x in training_step_outputs])
        # target = torch.stack([x['label'] for x in training_step_outputs], dim = 0)
        target = torch.stack([x[0]['label'] for x in training_step_outputs])

        # sub_probs = torch.cat(x[0]['sub_probs'] for x in training_step_outputs)
        # sub_targets = torch.cat(x[0]['sub_targets'] for x in training_step_outputs)
        # target = torch.argmax(target, dim=1)
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

        self.log('Train/auc', self.AUROC(probs, target), prog_bar=True, on_epoch=True, logger=True)
        # self.log('Train/sub_auc', self.AUROC(sub_probs, sub_targets), prog_bar=True, on_epoch=True, logger=True)

    def validation_step(self, batch, batch_idx):

        input, label, _= batch
        label = label.float()

        sub_predictions, slide_prediction, Y_prob, Y_hat = self(input)

        # print(sub_predictions.size(0))
        sub_labels = [label] * sub_predictions.size(0)         

        
        sub_labels = torch.stack(sub_labels).squeeze()
        # print(sub_labels.shape)
        # print(sub_predictions.shape)
        
        sub_loss = self.loss(sub_predictions, sub_labels)
        slide_loss = self.loss(slide_prediction, label)

        
        Y = torch.argmax(label)

        #---->acc log
        # Y = int(label[0][1])
        Y = torch.argmax(label)

        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        return {'val_sub_loss': sub_loss, 'val_slide_loss': slide_loss, 'logits' : slide_prediction, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : Y}


    def validation_epoch_end(self, val_step_outputs):
        logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in val_step_outputs])
        max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
        target = torch.stack([x['label'] for x in val_step_outputs])
        
        self.log_dict(self.valid_metrics(logits, target),
                          on_epoch = True, logger = True)
        
        #---->
        # logits = logits.long()
        # target = target.squeeze().long()
        # logits = logits.squeeze(0)
        if len(target.unique()) != 1:
            self.log('val_auc', self.AUROC(probs, target), prog_bar=True, on_epoch=True, logger=True)
        else:    
            self.log('val_auc', 0.0, prog_bar=True, on_epoch=True, logger=True)

        

        self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True)
        

        precision, recall, thresholds = self.PRC(probs, target)



        # print(max_probs.squeeze(0).shape)
        # print(target.shape)
        

        #----> log confusion matrix
        self.log_confusion_matrix(max_probs, target, stage='val')
        

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

        torch.set_grad_enabled(True)
        data, label, (wsi_name, batch_names) = batch
        wsi_name = wsi_name[0]
        label = label.float()
        # logits, Y_prob, Y_hat = self.step(data) 
        # print(data.shape)
        data = data.squeeze(0).float()
        logits, attn = self(data)
        attn = attn.detach()
        logits = logits.detach()

        Y = torch.argmax(label)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        
        #----> Get GradCam maps, map each instance to attention value, assemble, overlay on original WSI 
        if self.model_name == 'TransMIL':
           
            target_layers = [self.model.layer2.norm] # 32x32
            # target_layers = [self.model_ft[0].features[-1]] # 32x32
            self.cam = GradCAM(model=self.model, target_layers = target_layers, use_cuda=True, reshape_transform=self.reshape_transform) #, reshape_transform=self.reshape_transform
            # self.cam_ft = GradCAM(model=self.model, target_layers = target_layers_ft, use_cuda=True) #, reshape_transform=self.reshape_transform
        else:
            target_layers = [self.model.attention_weights]
            self.cam = GradCAM(model = self.model, target_layers = target_layers, use_cuda=True)


        data_ft = self.model_ft(data).unsqueeze(0).float()
        instance_count = data.size(0)
        target = [ClassifierOutputTarget(Y)]
        grayscale_cam = self.cam(input_tensor=data_ft, targets=target)
        grayscale_cam = torch.Tensor(grayscale_cam)[:instance_count, :]

        # attention_map = grayscale_cam[:, :, 1].squeeze()
        # attention_map = F.relu(attention_map)
        # mask = torch.zeros((instance_count, 3, 256, 256)).to(self.device)
        # for i, v in enumerate(attention_map):
        #     mask[i, :, :, :] = v

        # mask = self.assemble(mask, batch_names)
        # mask = (mask - mask.min())/(mask.max()-mask.min())
        # mask = mask.cpu().numpy()
        # wsi = self.assemble(data, batch_names)
        # wsi = wsi.cpu().numpy()

        # def show_cam_on_image(img, mask):
        #     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        #     heatmap = np.float32(heatmap) / 255
        #     cam = heatmap*0.4 + np.float32(img)
        #     cam = cam / np.max(cam)
        #     return cam

        # wsi = show_cam_on_image(wsi, mask)
        # wsi = ((wsi-wsi.min())/(wsi.max()-wsi.min()) * 255.0).astype(np.uint8)
        
        # img = Image.fromarray(wsi)
        # img = img.convert('RGB')
        

        # output_path = self.save_path / str(Y.item())
        # output_path.mkdir(parents=True, exist_ok=True)
        # img.save(f'{output_path}/{wsi_name}.jpg')


        #----> Get Topk Tiles and Topk Patients
        summed = torch.mean(grayscale_cam, dim=2)
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

        return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : Y, 'name': wsi_name, 'topk_data': topk_data} #
        # return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label, 'name': name} #, 'topk_data': topk_data

    def test_epoch_end(self, output_results):
        logits = torch.cat([x['logits'] for x in output_results], dim = 0)
        probs = torch.cat([x['Y_prob'] for x in output_results])
        max_probs = torch.stack([x['Y_hat'] for x in output_results])
        # target = torch.stack([x['label'] for x in output_results], dim = 0)
        target = torch.stack([x['label'] for x in output_results])
        # target = torch.argmax(target, dim=1)
        patients = [x['name'] for x in output_results]
        topk_tiles = [x['topk_data'] for x in output_results]
        #---->
        auc = self.AUROC(probs, target)
        fpr, tpr, thresholds = self.ROC(probs, target)
        fpr = fpr.cpu().numpy()
        tpr = tpr.cpu().numpy()

        plt.figure(1)
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.savefig(f'{self.save_path}/roc.jpg')
        # self.loggers[0].experiment.add_figure(f'{stage}/Confusion matrix', fig_, self.current_epoch)

        metrics = self.test_metrics(logits , target)


        # metrics = self.test_metrics(max_probs.squeeze() , torch.argmax(target.squeeze(), dim=1))
        metrics['test_auc'] = auc

        # self.log('auc', auc, prog_bar=True, on_epoch=True, logger=True)

        #---->get highest scoring patients for each class
        # test_path = Path(self.save_path) / 'most_predictive' 
        
        # Path.mkdir(output_path, exist_ok=True)
        topk, topk_indices = torch.topk(probs.squeeze(0), 5, dim=0)
        for n in range(self.n_classes):
            print('class: ', n)
            
            topk_patients = [patients[i[n]] for i in topk_indices]
            topk_patient_tiles = [topk_tiles[i[n]] for i in topk_indices]
            for x, p, t in zip(topk, topk_patients, topk_patient_tiles):
                print(p, x[n])
                patient = p
                # outpath = test_path / str(n) / patient 
                outpath = Path(self.save_path) / str(n) / patient
                outpath.mkdir(parents=True, exist_ok=True)
                for i in range(len(t)):
                    tile = t[i]
                    tile = tile.cpu().numpy().transpose(1,2,0)
                    tile = (tile - tile.min())/ (tile.max() - tile.min()) * 255
                    tile = tile.astype(np.uint8)
                    img = Image.fromarray(tile)
                    
                    img.save(f'{outpath}/{i}.jpg')

            
            
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


        self.log_confusion_matrix(max_probs, target, stage='test')
        #---->
        result = pd.DataFrame([metrics])
        result.to_csv(Path(self.save_path) / f'test_result.csv', mode='a', header=not Path(self.save_path).exists())

        # with open(f'{self.save_path}/test_metrics.txt', 'a') as f:

        #     f.write([metrics])

    def configure_optimizers(self):
        # optimizer_ft = optim.Adam(self.model_ft.parameters(), lr=self.optimizer.lr*0.1)
        optimizer0 = torch.optim.Adam(self.trainable_parameters, lr=1e-4, weight_decay=1e-2)
        optimizer1 = torch.optim.Adam(self.attCls.parameters(), lr=1e-4, weight_decay=1e-2)

        scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer0, [100], gamma=0.2)
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [100], gamma=0.2)
        return [optimizer0, optimizer1], [scheduler0, scheduler1]     

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
        confmat = self.confusion_matrix(max_probs, target)
        print(confmat)
        df_cm = pd.DataFrame(confmat.cpu().numpy(), index=range(self.n_classes), columns=range(self.n_classes))
        fig_ = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Spectral').get_figure()
        if stage == 'train':
            self.loggers[0].experiment.add_figure(f'{stage}/Confusion matrix', fig_, self.current_epoch)
        else:
            fig_.savefig(f'{self.loggers[0].log_dir}/cm_test.png', dpi=400)

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

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

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

