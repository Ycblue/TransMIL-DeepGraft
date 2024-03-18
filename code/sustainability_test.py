import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from tqdm import tqdm
import sys
# from torchvision.datasets import Dataset
from torch.utils.data import random_split, Dataset, DataLoader

from models.ResNet import resnet50
from models import TransMIL, CLAM_MB
import timm
from pathlib import Path
from torchvision import transforms as T
from experiment_impact_tracker.compute_tracker import ImpactTracker
from experiment_impact_tracker.data_interface import DataInterface
import multiprocessing as mp
import torch.nn.functional as F

from utils.custom_resnet50 import resnet50_baseline
from torchvision import models
from codecarbon import track_emissions, OfflineEmissionsTracker
from codecarbon.output import LoggerOutput
import logging
import csv
import pandas as pd

class CustomImageDataset(Dataset):
    def __init__(self, data_size=1000, bag_size=1000, feature_size=2048, device='cpu', mode='features'):
        self.data_size = data_size
        self.bag_size = bag_size
        self.device = device
        self.mode = mode
        self.feature_size = feature_size
        # self.data = torch.rand([self.bag_size, 3, 224, 224])

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):

        label = torch.randint(1, (1,1))#.to(self.device)
        if self.mode == 'features':
            bag = torch.rand([self.bag_size, self.feature_size])#.to(self.device)
        else:
            bag = torch.rand([self.bag_size, 3, self.feature_size, self.feature_size])#.to(self.device)
            # bag = T.Resize(224)(bag)
        return bag, label

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='transmil', type=str)
    parser.add_argument('--feature_size', default=224, type=int)
    parser.add_argument('--bag_size', default=250, type=int)

    args = parser.parse_args()
    return args

# @track_emissions(offline=True, country_iso_code='DEU')
def compute(args):
    model_name = args.model
    feature_size = args.feature_size
    n_classes = 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_list = ['resnet50', 'retccl', 'transmil', 'resnet50', 'vit', 'inception']
    model_list = ['transmil', 'vit', 'clam', 'inception']
    model_list = ['transmil']
    # for model_name in [args.model]:
    for model_name in model_list:
    

        if model_name == 'transmil':
            # in_features = 2048
            f_model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
            f_model.fc = torch.nn.Identity()
            f_model.load_state_dict(torch.load('/homeStor1/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
            f_model.eval()

            model_ckpt_path = '/homeStor1/ylan/workspace/TransMIL-DeepGraft/logs/DeepGraft/TransMIL/norm_rej_rest/_features_CrossEntropyLoss/lightning_logs/version_53/checkpoints/epoch=17-val_loss=0.9646-val_auc= 0.7541-val_patient_auc=0.0000.ckpt'
            model_ckpt = torch.load(model_ckpt_path)
            model_weights = model_ckpt['state_dict']
            
            model = TransMIL(n_classes=n_classes, in_features=2048)

            for key in list(model_weights):
                model_weights[key.replace('model.', '')] = model_weights.pop(key)
            
            model.load_state_dict(model_weights)
            
            mode = 'images'
            feature_size = 224
            f_model.to(device)
        elif model_name == 'retccl':
            model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
            model.fc = torch.nn.Identity()
            model.load_state_dict(torch.load('/homeStor1/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
            mode = 'images'
            feature_size = 224
        elif model_name == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1')
            model.fc = nn.Linear(model.fc.in_features, 1024)
            for param in model.parameters():
                param.requires_grad = False
            # model.load_state_dict(torch.load('models/ckpt/retccl_best_ckpt.pth'), strict=False)

            mode = 'images'
            # feature_size = 224
        elif model_name == 'vit':
            model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=n_classes)
            for param in model.parameters():
                param.requires_grad = False
            outputs_attrs = n_classes
            num_inputs = model.head.in_features
            last_layer = nn.Linear(num_inputs, outputs_attrs)
            model.head = last_layer
            mode = 'images'
            # feature_size = 224
        elif model_name == 'inception':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights='Inception_V3_Weights.DEFAULT')
            model.aux_logits = False
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(model.fc.in_features, n_classes)
            mode = 'images'
            feature_size = 299
        elif model_name == 'clam':

            f_model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
            # f_model.fc = torch.nn.Identity()
            f_model.load_state_dict(torch.load('/homeStor1/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
            f_model.fc = torch.nn.Linear(f_model.fc.in_features, 1024)

            # f_model = 


            model = CLAM_MB(n_classes = n_classes)
            mode = 'images'
            feature_size = 224
            # epochs = 100
            f_model.eval()

            f_model.to(device)

        
        external_logger = logging.getLogger("codecarbon")
        # while logger.hasHandlers():
        #     logger.removeHandler(logger.handlers[0])

        # Define a log formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)-12s: %(levelname)-8s %(message)s"
        )
        channel = logging.FileHandler("/homeStor1/ylan/workspace/TransMIL-DeepGraft/cc_co2log/codecarbon.log")
        external_logger.addHandler(channel)


        external_logger.setLevel(logging.DEBUG)

        consoleHandler = logging.StreamHandler(sys.stdout)
        # consoleHandler.setFormatter(formatter)
        consoleHandler.setLevel(logging.INFO)
        external_logger.addHandler(consoleHandler)


        external_logger = LoggerOutput(external_logger, logging.INFO)
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)
        

        # logger.debug("GO!")
        # if f_model: 
        #     f_model.to(device)
        model = model.to(device)
        model.eval()
        # test_data = torch.rand([1000,3,224,224]).to(device)
        epochs=100

        data_size = 1#how many slides are processed

        # bag_size = args.bag_size
        
        # epochs = 1
        # bs_array = [bag_size]*10
        repetition = 10
        # bs_array = [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900]
        # bs_array = [1, 10, 100]
        # bs_array = [100, 200, 300, 400, 500, 600, 700, 800, 900]
        bs_array = [600, 700, 800, 900, 1000]
        # bs_array = [1000]
        data_col = []
        for i, bs in enumerate(bs_array):
            for r in range(repetition):
                # epochs = 950-bs
                # epochs = 100 if bs < 300 else 50
                bag_size= bs
                dataset = CustomImageDataset(data_size=data_size, bag_size=bag_size, device=device, mode=mode, feature_size=feature_size)
                dataloader = DataLoader(dataset, batch_size=1)

                # optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
                tracker_path = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/co2log/{model_name}'
                Path(tracker_path).mkdir(exist_ok=True)
                tracker = ImpactTracker(tracker_path)
                tracker.launch_impact_monitor()
                
                # cc_tracker = OfflineEmissionsTracker(
                #     output_file=f'{model_name}_{epochs}_{bs}.csv', 
                #     project_name=f'{model_name}_{epochs}_{bs}_{i}', 
                #     gpu_ids=[0],
                #     country_iso_code='DEU', 
                #     output_dir='/home/ylan/workspace/TransMIL-DeepGraft/cc_co2log', 
                #     tracking_mode='process', 
                #     measure_power_secs=1, on_csv_write='append', 
                #     save_to_logger=True, 
                #     logging_logger=external_logger)
                # cc_tracker.start()
                
                with torch.no_grad():
                    for e in tqdm(range(epochs)):
                        loss = 0
                        for item in dataloader:
                            bag, label = item
                            bag = bag.to(device)
                            
                            if len(bag.shape) > 4:
                                bag = bag.squeeze(0)
                            label = label.to(device)
                            # if model_name == 'clam':
                            #     bag = bag.squeeze(0)


                            # print(bag.shape)
                            with torch.cuda.amp.autocast():
                                if model_name == 'transmil' or model_name == 'clam':
                                    # print(bag.shape)
                                    features = f_model(bag)
                                    # print(features.shape)
                                    if model_name == 'clam':
                                        # print(features.shape)
                                        out = model(features)
                                    else:  
                                        out = model(features.unsqueeze(0))
                                else: 
                                    features = model(bag)
                            # logits = torch.argmax(features, dim=1)
                            # prob = F.max(logits, dim=1)
                            
                            # # loss += (prob - label)
                            # loss = prob - label
                            # loss.backward()                
                            bag = bag.cpu().detach()
                            del bag
                            del out
                            del features
                    # cc_tracker.flush()
                # end = time.time()
                        # print('Epoch Time: ', end-start)

                tracker.stop()
                del tracker
                

                # cc_tracker.stop()
                # emissions: float = cc_tracker.stop()
                # print(emissions)
                

                data_interface = DataInterface([tracker_path])

                print('====================')
                print(f'Repetion: {r}, {model_name}, Epochs: {epochs}, Data Size: {data_size}, Bag Size: {bag_size}')
                print('====================')
                print('total_power: ', data_interface.total_power)
                print('average power per slide: ', data_interface.total_power / (epochs*data_size))
                print('PUE: ', data_interface.PUE)
                print('kg_carbon: ', data_interface.kg_carbon)
                print('kg_carbon/epoch: ', data_interface.kg_carbon / epochs)
                print(f'kg_carbon/single slide: ', data_interface.kg_carbon / epochs / data_size)
                print('g_carbon: ', data_interface.kg_carbon * 1000)
                print('g_carbon/epoch: ', data_interface.kg_carbon * 1000 / epochs)
                print(f'g_carbon/single slide: ', data_interface.kg_carbon * 1000 / epochs / data_size)
                kg_carbon = data_interface.kg_carbon / epochs
                # print(kg_carbon)

                # txt_path = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/co2_emission/{model_name}_grad.txt'
                csv_path = f'/homeStor1/ylan/npj_sus_data/co2_data/{model_name}_power_per_slide_{bag_size}.csv'
                # csv_path = f'/homeStor1/ylan/npj_sus_data/co2_data/{model_name}_power_per_slide_{feature_size}_{bag_size}.csv'

                csv_columns = ['Power (kWh)']

                if not Path(csv_path).is_file():
                    df = pd.DataFrame(columns=csv_columns)
                    # df.loc[len(df.index)] = [data_interface.total_power/epochs]
                    df.to_csv(csv_path, index=False)
                # else:
                # df = pd.read_csv(csv_path)
                # df.loc[len(df.index)] = [data_interface.total_power/epochs]

                
                csv_row = [data_interface.total_power/(epochs*data_size)]
                data_col.append(data_interface.total_power/(epochs*data_size))
                
                data_dict = {'Power (kWh)': data_interface.total_power/(epochs*data_size)}
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['Power (kWh)'])
                    writer.writerow(data_dict)
                    f.close()
                print('Saved to: ', csv_path)
            # Path(txt_path).mkdir(exist_ok=True)
            # with open(txt_path, 'a') as f:
            #     f.write(f'==================================================================================== \n')
            #     f.write(f'Emissions calculated for {data_size} slides, {bs} patches/slide, {feature_size} features, per epoch \n')
            #     f.write(f'{model_name}: {kg_carbon} [kg]\n')
            #     f.write(f'{model_name}: {kg_carbon*1000} [g]\n')
            #     f.write(f'Emissions calculated for 1 slides, {bs} patches/slide, per epoch \n')
            #     f.write(f'{model_name}: {kg_carbon/data_size} [kg]\n')
            #     f.write(f'{model_name}: {kg_carbon*1000/data_size} [g]\n')
            #     f.write(f'==================================================================================== \n')



if __name__ == '__main__':

    

    # mp.set_start_method("fork")
    args = make_parse()
    # with OfflineEmmisionsTracker(country_iso_code='GER') as tracker:
    

    compute(args)

    

