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
    def __init__(self, data_size=1000, bag_size=1000, feature_size=2048, num_classes=3, device='cpu', mode='features', cache=True):
        self.data_size = data_size
        self.bag_size = bag_size
        self.device = device
        self.mode = mode
        self.feature_size = feature_size
        self.cache = cache
        self.num_classes = num_classes
        self.data_list = []
        self.label_list = []
        # cache_size = 500
        if self.cache:
            if self.mode == 'features':
                self.data_list = [torch.rand([self.bag_size, self.feature_size]) for i in range(data_size)]
                self.label_list = [torch.randint(1, 3, (1,1)) for i in range(data_size)]
            else:
                # bag = torch.rand([self.bag_size, 3, self.feature_size, self.feature_size]).to(self.device)
                self.data_list = [torch.rand([3, self.feature_size, self.feature_size]) for i in range(data_size)]
                self.label_list = [torch.randint(1, (1,1)) for i in range(data_size)]
            # data = [torch.rand([])]

        # self.data = torch.rand([self.bag_size, 3, 224, 224])
        print('len(self.data_list): ', len(self.data_list))

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        if self.cache:
            return self.data_list[idx], self.label_list[idx]
        label = torch.randint(1, (1,1)).to(self.device)
        if self.mode == 'features':
            bag = torch.rand([self.data_size, self.bag_size, self.feature_size]).to(self.device)
        else:
            bag = torch.rand([self.bag_size, 3, self.feature_size, self.feature_size]).to(self.device)
            # bag = T.Resize(224)(bag)
        return bag, label

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='transmil', type=str)
    parser.add_argument('--feature_size', default=224, type=int)
    parser.add_argument('--bag_size', default=1000, type=int)

    args = parser.parse_args()
    return args

# @track_emissions(offline=True, country_iso_code='DEU')
def compute(args):
    model_name = args.model
    feature_size = args.feature_size
    n_classes = 3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model_list = ['resnet50', 'retccl', 'transmil', 'resnet50', 'vit', 'inception']
    # model_list = ['transmil', 'vit', 'clam', 'inception']
    # model_list = ['transmil']
    # for model_name in model_list:

    for model_name in [args.model]:
    

        if model_name == 'transmil':
            # in_features = 2048
            # f_model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
            # f_model.fc = torch.nn.Identity()
            # f_model.load_state_dict(torch.load('/homeStor1/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
            # f_model.eval()

            model_ckpt_path = '/homeStor1/ylan/workspace/TransMIL-DeepGraft/logs/DeepGraft/TransMIL/norm_rej_rest/_features_CrossEntropyLoss/lightning_logs/version_53/checkpoints/epoch=17-val_loss=0.9646-val_auc= 0.7541-val_patient_auc=0.0000.ckpt'
            model_ckpt = torch.load(model_ckpt_path)
            model_weights = model_ckpt['state_dict']
            
            model = TransMIL(n_classes=n_classes, in_features=2048)

            for key in list(model_weights):
                model_weights[key.replace('model.', '')] = model_weights.pop(key)
            
            model.load_state_dict(model_weights)
            
            mode = 'features'
            feature_size = 2048
            # f_model.to(device)
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

            mode = 'features'
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

            # f_model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
            # # f_model.fc = torch.nn.Identity()
            # f_model.load_state_dict(torch.load('/homeStor1/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=False)
            # f_model.fc = torch.nn.Linear(f_model.fc.in_features, 1024)

            # f_model = 


            model = CLAM_MB(n_classes = n_classes)
            mode = 'features'
            feature_size = 1024
            # epochs = 100
            # f_model.eval()

            # f_model.to(device)

        
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
        # model.eval()
        # test_data = torch.rand([1000,3,224,224]).to(device)
        epochs=300

        data_size = 1041#how many slides are processed

        bag_size = args.bag_size
        
        # epochs = 1
        batch_size=50

        print('total amount of patches: ', bag_size * data_size*epochs)

        if model_name in ['vit', 'inception', 'retccl']:
            batch_size = 2000
            # bag_size = int(bag_size/batch_size)
            epochs = int(epochs * data_size * bag_size / batch_size)
            data_size = batch_size
        elif model_name == 'clam':
            batch_size = 1
        print('batch_size: ', batch_size)
        print('bag_size: ', bag_size)
        print('epochs: ', epochs)

        repetition = 1
        for i in range(repetition):

            dataset = CustomImageDataset(data_size=data_size, bag_size=bag_size, device=device, mode=mode, feature_size=feature_size, cache=True)
            dataloader = DataLoader(dataset, batch_size=batch_size)

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
            
            optimizer = torch.optim.Adam(model.parameters())
            loss_fn = torch.nn.CrossEntropyLoss()
            
            # with torch.no_grad():
            for e in tqdm(range(epochs)):
                loss = 0
                optimizer.zero_grad()
                for item in dataloader:
                    bag, label = item
                    bag = bag.to(device)
                    
                    if len(bag.shape) > 4:
                        bag = bag.squeeze(0)
                    # if model_name == 'clam':
                    #     bag = bag.squeeze(0)
                    # if model_name == 'vit' or model_name == 'inception':
                    #     label = torch.ones([bag.shape[0]], device=device)* label.item()

                        # label = label.to(device)
                    label = label.to(device)
                    
                    # print(bag.shape)
                    with torch.cuda.amp.autocast():
                        if model_name == 'transmil' or model_name == 'clam':
                            # print(bag.shape)
                            # print(bag.shape)
                            # with torch.no_grad():
                            #     features = f_model(bag)
                            # print(features.shape)
                            if model_name == 'clam':
                                # print(bag.shape)
                                out = model(bag.squeeze(0))[0]
                                label = label.squeeze()
                                out_label = torch.nn.functional.one_hot(label, num_classes=3).squeeze().float().unsqueeze(0)
                            else:  
                                out = model(bag.unsqueeze(0))
                                label = label.squeeze()
                                out_label = torch.nn.functional.one_hot(label, num_classes=3).squeeze().float()


                        else: 
                            # print(bag.shape)
                            # print(label)
                            # label = label*bag.shape[0]
                            # print(label)
                            # print(bag.shape)
                            # print(label.shape)
                            # out_label = label.long()
                            out_label = torch.nn.functional.one_hot(label, num_classes=3).squeeze().float()
                            # out_label.to(device)
                            # print(label.shape)
                            
                            out = model(bag.squeeze())
                    # print(out.shape)
                    # print(out_label.shape)
                    
                    
                    loss = loss_fn(out, out_label)
                    loss.backward()
                    optimizer.step()
                    # running_loss += loss.item()
                    # logits = torch.argmax(features, dim=1)
                    # prob = F.max(logits, dim=1)
                    
                    # # loss += (prob - label)
                    # loss = prob - label
                    # loss.backward()                
                    bag = bag.detach()
                    del bag
                    del out_label
                    del label
                # model = model.detach()
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
            print(f'{model_name}, Epochs: {epochs}, Data Size: {data_size}, Bag Size: {bag_size}')
            print('====================')
            print('total_power: ', data_interface.total_power)
            print('PUE: ', data_interface.PUE)
            print('kg_carbon: ', data_interface.kg_carbon)
            print('kg_carbon/epoch: ', data_interface.kg_carbon / epochs)
            print(f'kg_carbon/single slide: ', data_interface.kg_carbon / epochs / data_size)
            print('g_carbon: ', data_interface.kg_carbon * 1000)
            print('g_carbon/epoch: ', data_interface.kg_carbon * 1000 / epochs)
            print(f'g_carbon/single slide: ', data_interface.kg_carbon * 1000 / epochs / data_size)
            kg_carbon = data_interface.kg_carbon / epochs
            # print(kg_carbon)

            '''Netherlands'''

            # txt_path = f'/homeStor1/ylan/workspace/TransMIL-DeepGraft/test/co2_emission/{model_name}_grad.txt'
            # csv_path = f'/homeStor1/ylan/npj_sus_data/co2_data/{model_name}_power_per_slide_{bag_size}.csv'
            csv_path = f'/homeStor1/ylan/npj_sus_data/co2_data/{model_name}_training_e{epochs}_bs_{bag_size}.csv'

            csv_columns = ['Power (kWh)']

            if not Path(csv_path).is_file():
                df = pd.DataFrame(columns=csv_columns)
                # df.loc[len(df.index)] = [data_interface.total_power/epochs]
                df.to_csv(csv_path, index=False)
            # else:
            # df = pd.read_csv(csv_path)
            # df.loc[len(df.index)] = [data_interface.total_power/epochs]

            
            csv_row = [data_interface.total_power]

            with open(csv_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(csv_row)
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

    '''
    TransMIL/CLAM: only evaluate 
    '''


    # mp.set_start_method("fork")
    args = make_parse()
    # with OfflineEmmisionsTracker(country_iso_code='GER') as tracker:
    

    compute(args)

    

