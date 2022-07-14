from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F
from torchmetrics.classification.accuracy import Accuracy
import os.path as osp
from abc import ABC, abstractmethod
from copy import deepcopy
from pytorch_lightning import LightningModule
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from datasets.data_interface import BaseKFoldDataModule
from typing import Any, Dict, List, Optional, Type
import torchmetrics
import numpy as np
from PIL import Image
import cv2
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from test_visualize import custom_test_module
from pytorch_grad_cam import GradCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pathlib import Path



class EnsembleVotingModel(LightningModule):
    def __init__(self, model_cls: Type[LightningModule], checkpoint_paths: List[str], n_classes, log_path) -> None:
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.n_classes = n_classes
        self.log_path = log_path
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        self.test_acc = Accuracy()
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
        self.test_metrics = metrics.clone(prefix = 'test_')
        self.confusion_matrix = torchmetrics.ConfusionMatrix(num_classes = self.n_classes)
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
        grayscale_cam = torch.Tensor(grayscale_cam)[:instance_count, :] #.to(self.device)

        #----------------------------------------------------
        # Get Topk Tiles and Topk Patients
        #----------------------------------------------------
        summed = torch.mean(grayscale_cam, dim=2)
        topk_tiles, topk_indices = torch.topk(summed.squeeze(0), 5, dim=0)
        topk_data = data[topk_indices].detach()
        
        #----------------------------------------------------
        # Log Correct/Count
        #----------------------------------------------------
        Y = torch.argmax(label)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat.item() == Y)

        #----------------------------------------------------
        # Tile Level Attention Maps
        #----------------------------------------------------

        # self.save_attention_map(wsi_name, data, batch_names, grayscale_cam, Y)

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


        # self.log_roc_curve(probs, target, 'test')
        self.log_confusion_matrix(max_probs, target, stage='test')
        #---->
        result = pd.DataFrame([metrics])
        result.to_csv(Path(self.save_path) / f'test_result.csv', mode='a', header=not Path(self.save_path).exists())

    def save_attention_map(self, wsi_name, data, batch_names, grayscale_cam, target):

        def get_coords(batch_names): #ToDO: Change function for precise coords
            coords = []
            
            for tile_name in batch_names: 
                pos = re.findall(r'\((.*?)\)', tile_name[0])
                x, y = pos[0].split('_')

                coords.append((int(x),int(y)))
            return coords

        
        coords = get_coords(batch_names)
        # temp_data = data.cpu()
        print(data.shape)
        wsi = self.assemble(data, coords).cpu().numpy()
        wsi = (wsi-wsi.min())/(wsi.max()-wsi.min())
        # wsi = wsi

        #--> Get interpolated mask from GradCam
        W, H = wsi.shape[0], wsi.shape[1]
        
        
        attention_map = grayscale_cam[:, :, 1].squeeze()
        attention_map = F.relu(attention_map)
        # print(attention_map)
        input_h = 256
        
        mask = torch.ones(( int(W/input_h), int(H/input_h))).to(self.device)

        for i, (x,y) in enumerate(coords):
            mask[y][x] = attention_map[i]
        mask = mask.unsqueeze(0).unsqueeze(0)
        # mask = torch.stack([mask, mask, mask]).unsqueeze(0)

        mask = F.interpolate(mask, (W,H), mode='bilinear')
        mask = mask.squeeze(0).permute(1,2,0)

        mask = (mask - mask.min())/(mask.max()-mask.min())
        mask = mask.cpu().numpy()
        
        def show_cam_on_image(img, mask):
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap*0.4 + np.float32(img)
            cam = cam / np.max(cam)
            return cam

        wsi_cam = show_cam_on_image(wsi, mask)
        wsi_cam = ((wsi_cam-wsi_cam.min())/(wsi_cam.max()-wsi_cam.min()) * 255.0).astype(np.uint8)
        
        img = Image.fromarray(wsi_cam)
        img = img.convert('RGB')
        output_path = self.save_path / str(target)
        output_path.mkdir(parents=True, exist_ok=True)
        img.save(f'{output_path}/{wsi_name}_gradcam.jpg')

        wsi = ((wsi-wsi.min())/(wsi.max()-wsi.min()) * 255.0).astype(np.uint8)
        img = Image.fromarray(wsi)
        img = img.convert('RGB')
        output_path = self.save_path / str(target)
        output_path.mkdir(parents=True, exist_ok=True)
        img.save(f'{output_path}/{wsi_name}.jpg')


    def assemble(self, tiles, coords): # with coordinates (x-y)
        
        def getPosition(img_name):
            pos = re.findall(r'\((.*?)\)', img_name) #get strings in brackets (0-0)
            a = int(pos[0].split('-')[0])
            b = int(pos[0].split('-')[1])
            return a, b

        position_dict = {}
        assembled = []
        y_max = 0
        # for tile in self.predictions:
        count = 0
        max_x = max(coords, key = lambda t: t[0])[0]
        d = 0
        white_value = 0

        for i, (x,y) in enumerate(coords):

            # name = n[0]
            image = tiles[i,:,:,:].permute(1,2,0)
            
            d = image.shape
            # print(image.min())
            # print(image.max())
            # if image.max() > white_value:
            #     white_value = image.max()
            # # print(image.shape)
            
            # tile_position = '-'.join(name.split('_')[-2:])
            # x,y = getPosition(tile_position)
            
            y_max = y if y > y_max else y_max
            if x not in position_dict.keys():
                position_dict[x] = [(y, image)]
            else: position_dict[x].append((y, image))
            count += 1
        

        for i in range(max_x+1):
            column = [None]*(int(y_max+1))
            # if len(d) == 3:
            #     empty_tile = torch.zeros(d).to(self.device)
            # else:
            empty_tile = torch.ones(d)
            empty_tile = torch.ones(d).to(self.device)
            if i in position_dict.keys():
                for j in position_dict[i]:
                    sample = j[1]
                    column[int(j[0])] = sample
            column = [empty_tile if i is None else i for i in column]
            # for c in column:
            #     print(c.shape)
            # column = torch.vstack(column)
            column = torch.stack(column)
            assembled.append((i, column))



        # for key in position_dict.keys():
        #     column = [None]*(int(y_max+1))
        #     # print(key)
        #     for i in position_dict[key]:
        #         sample = i[1]
        #         d = sample.shape
        #         # print(d) # [3,256,256]
        #         if len(d) == 3:
        #             empty_tile = torch.ones(d).to(self.device)
        #         else:
        #             empty_tile = torch.zeros(d).to(self.device)
        #         column[int(i[0])] = sample
        #     column = [empty_tile if i is None else i for i in column]
        #     column = torch.vstack(column)
        #     assembled.append((key, column))
        # print(len(assembled))
        
        assembled = sorted(assembled, key=lambda x: x[0])

        stack = [i[1] for i in assembled]
        # print(stack)
        img_compl = torch.hstack(stack)
        return img_compl
    # def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
    #     # Compute the averaged predictions over the `num_folds` models.
    #     # print(batch[0].shape)
    #     input, label, _ = batch
    #     label = label.float()
    #     input = input.squeeze(0).float()

            
    #     logits = torch.stack([m(input) for m in self.models]).mean(0)
    #     Y_hat = torch.argmax(logits, dim=1)
    #     Y_prob = F.softmax(logits, dim = 1)
    #     # #---->acc log
    #     Y = torch.argmax(label)
    #     self.data[Y]["count"] += 1
    #     self.data[Y]["correct"] += (Y_hat.item() == Y)

    #     return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    # def test_epoch_end(self, output_results):
    #     probs = torch.cat([x['Y_prob'] for x in output_results])
    #     max_probs = torch.stack([x['Y_hat'] for x in output_results])
    #     # target = torch.stack([x['label'] for x in output_results], dim = 0)
    #     target = torch.cat([x['label'] for x in output_results])
    #     target = torch.argmax(target, dim=1)
        
    #     #---->
    #     auc = self.AUROC(probs, target.squeeze())
    #     metrics = self.test_metrics(max_probs.squeeze() , target)


    #     # metrics = self.test_metrics(max_probs.squeeze() , torch.argmax(target.squeeze(), dim=1))
    #     metrics['test_auc'] = auc

    #     # self.log('auc', auc, prog_bar=True, on_epoch=True, logger=True)

    #     # print(max_probs.squeeze(0).shape)
    #     # print(target.shape)
    #     # self.log_dict(metrics, logger = True)
    #     for keys, values in metrics.items():
    #         print(f'{keys} = {values}')
    #         metrics[keys] = values.cpu().numpy()
    #     #---->acc log
    #     for c in range(self.n_classes):
    #         count = self.data[c]["count"]
    #         correct = self.data[c]["correct"]
    #         if count == 0: 
    #             acc = None
    #         else:
    #             acc = float(correct) / count
    #         print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
    #     self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    #     self.log_confusion_matrix(probs, target, stage='test')
    #     #---->
    #     result = pd.DataFrame([metrics])
    #     result.to_csv(self.log_path / 'result.csv')


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

class KFoldLoop(Loop):
    def __init__(self, num_folds: int, export_path: str, **kargs) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path
        self.n_classes = kargs["model"].n_classes
        self.log_path = kargs["log"]

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths, n_classes=self.n_classes, log_path=self.log_path)
        voting_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)