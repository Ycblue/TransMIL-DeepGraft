import pandas as pd

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn.functional import one_hot
from torch.utils import data
from torch.utils.data import random_split, DataLoader
from torchsampler import ImbalancedDatasetSampler
from torchvision import datasets, transforms
import pandas as pd
from sklearn.utils import shuffle
from pathlib import Path
from tqdm import tqdm
import zarr
import json
import cv2
from PIL import Image
# from models import TransMIL



class ZarrFeatureBagLoader(data.Dataset):
    def __init__(self, file_path, label_path, mode, n_classes, cache=False, data_cache_size=50, max_bag_size=1000):
        super().__init__()

        self.data_info = []
        self.data_cache = []
        self.slideLabelDict = {}
        self.files = []
        self.data_cache_size = data_cache_size
        self.mode = mode
        self.file_path = file_path
        # self.csv_path = csv_path
        self.label_path = label_path
        self.n_classes = n_classes
        self.max_bag_size = max_bag_size
        self.min_bag_size = 120
        self.empty_slides = []
        self.corrupt_slides = []
        self.cache = cache
        self.drop_rate=0.1
        self.cache=True
        print('mode: ', self.mode)
        # read labels and slide_path from csv
        with open(self.label_path, 'r') as f:
            temp_slide_label_dict = json.load(f)[self.mode]
            # print(len(temp_slide_label_dict))
            for (x, y) in temp_slide_label_dict:
                x = Path(x).stem
                # x_complete_path = Path(self.file_path)/Path(x)
                for cohort in Path(self.file_path).iterdir():
                    if self.mode == 'test':
                        x_complete_path = Path(self.file_path) / cohort / 'FEATURES_RETCCL_GAN' / (str(x) + '.zarr')
                    else:
                        x_complete_path = Path(self.file_path) / cohort / 'FEATURES_RETCCL' / (str(x) + '.zarr')
                    print(x_complete_path)
                    if x_complete_path.is_dir():
                        # if len(list(x_complete_path.iterdir())) > self.min_bag_size:
                        # # print(x_complete_path)
                        self.slideLabelDict[x] = y
                        self.files.append(x_complete_path)
        
        home = Path.cwd().parts[1]
        self.slide_patient_dict_path = f'/{home}/ylan/DeepGraft/training_tables/slide_patient_dict.json'
        with open(self.slide_patient_dict_path, 'r') as f:
            self.slide_patient_dict = json.load(f)

        self.feature_bags = []
        self.labels = []
        self.wsi_names = []
        self.name_batches = []
        self.patients = []
        for t in tqdm(self.files):
            self._add_data_infos(t, cache=cache)


        print('data_cache_size: ', self.data_cache_size)
        print('data_info: ', len(self.data_info))
        # if self.cache:
        #     print('Loading data into cache.')
        #     for t in tqdm(self.files):
        #         # zarr_t = str(t) + '.zarr'
        #         batch, label, (wsi_name, name_batch, patient) = self.get_data(t)

        #         self.labels.append(label)
        #         self.feature_bags.append(batch)
        #         self.wsi_names.append(wsi_name)
        #         self.name_batches.append(name_batch)
        #         self.patients.append(patient)
        # else: 
            

    def _add_data_infos(self, file_path, cache):

        # if cache:
        wsi_name = Path(file_path).stem
        # if wsi_name in self.slideLabelDict:
        label = self.slideLabelDict[wsi_name]
        patient = self.slide_patient_dict[wsi_name]
        idx = -1
        self.data_info.append({'data_path': file_path, 'label': label, 'name': wsi_name, 'patient': patient, 'cache_idx': idx})

    def get_data(self, i):

        fp = self.data_info[i]['data_path']
        idx = self.data_info[i]['cache_idx']
        if idx == -1:

        # if fp not in self.data_cache:
            self._load_data(fp)
        
        
        cache_idx = self.data_info[i]['cache_idx']
        label = self.data_info[i]['label']
        name = self.data_info[i]['name']
        patient = self.data_info[i]['patient']

        return self.data_cache[cache_idx], label, name, patient
        # return self.data_cache[fp][cache_idx], label, name, patient
        


    def _load_data(self, file_path):
        

        # batch_names=[] #add function for name_batch read out
        # wsi_name = Path(file_path).stem
        # if wsi_name in self.slideLabelDict:
        #     label = self.slideLabelDict[wsi_name]
        #     patient = self.slide_patient_dict[wsi_name]

        z = zarr.open(file_path, 'r')
        np_bag = np.array(z['data'][:])
        # np_bag = np.array(zarr.open(file_path, 'r')).astype(np.uint8)
        wsi_bag = torch.from_numpy(np_bag)
        batch_coords = torch.from_numpy(np.array(z['coords'][:]))

        # print(wsi_bag.shape)
        bag_size = wsi_bag.shape[0]
        
        # random drop 
        bag_idxs = torch.randperm(bag_size)[:int(bag_size*(1-self.drop_rate))]
        wsi_bag = wsi_bag[bag_idxs, :]
        batch_coords = batch_coords[bag_idxs]

        idx = self._add_to_cache((wsi_bag, batch_coords), file_path)
        file_idx = next(i for i, v in enumerate(self.data_info) if v['data_path'] == file_path)
        # print('file_idx: ', file_idx)
        # print('idx: ', idx)
        self.data_info[file_idx]['cache_idx'] = idx
        # print(wsi_bag.shape)
        # name_samples = [batch_names[i] for i in bag_idxs]
        # return wsi_bag, label, (wsi_name, batch_coords, patient)
        
        if len(self.data_cache) > self.data_cache_size:
            # removal_keys = list(self.data_cache)
            # removal_keys.remove(file_path)

            self.data_cache.pop(idx)

            self.data_info = [{'data_path': di['data_path'], 'label': di['label'], 'name': di['name'], 'patient':di['patient'], 'cache_idx':-1} if di['cache_idx'] == idx else di for di in self.data_info]
        


    def _add_to_cache(self, data, data_path):


        # if data_path not in self.data_cache:
        #     self.data_cache[data_path] = [data]
        # else:
        #     self.data_cache[data_path].append(data)
        self.data_cache.append(data)
        # print(len(self.data_cache))
        # return len(self.data_cache)
        return len(self.data_cache) - 1

    
    def get_labels(self, indices):
        # return [self.labels[i] for i in indices]
        return [self.data_info[i]['label'] for i in indices]


    def to_fixed_size_bag(self, bag, names, bag_size: int = 512):

        #duplicate bag instances unitl 

        bag_idxs = torch.randperm(bag.shape[0])[:bag_size]
        bag_samples = bag[bag_idxs]
        name_samples = [names[i] for i in bag_idxs]
        # bag_sample_names = [bag_names[i] for i in bag_idxs]
        # q, r  = divmod(bag_size, bag_samples.shape[0])
        # if q > 0:
        #     bag_samples = torch.cat([bag_samples]*q, 0)

        # self_padded = torch.cat([bag_samples, bag_samples[:r,:, :, :]])

        # zero-pad if we don't have enough samples
        # zero_padded = torch.cat((bag_samples,
        #                         torch.zeros(bag_size-bag_samples.shape[0], bag_samples.shape[1], bag_samples.shape[2], bag_samples.shape[3])))

        return bag_samples, name_samples, min(bag_size, len(bag))

    def data_dropout(self, bag, batch_names, drop_rate):
        bag_size = bag.shape[0]
        bag_idxs = torch.randperm(bag_size)[:int(bag_size*(1-drop_rate))]
        bag_samples = bag[bag_idxs]
        name_samples = [batch_names[i] for i in bag_idxs]

        return bag_samples, name_samples

    def __len__(self):
        # return len(self.files)
        return len(self.data_info)

    def __getitem__(self, index):

        (wsi, batch_coords), label, wsi_name, patient = self.get_data(index)

        label = torch.as_tensor(label)
        label = torch.nn.functional.one_hot(label, num_classes=self.n_classes)

        return wsi, label, (wsi_name, batch_coords, patient)

if __name__ == '__main__':
    
    from pathlib import Path
    import os
    import time
    # from fast_tensor_dl import FastTensorDataLoader
    # from custom_resnet50 import resnet50_baseline
    
    

    home = Path.cwd().parts[1]
    train_csv = f'/{home}/ylan/DeepGraft_project/code/debug_train.csv'
    data_root = f'/{home}/ylan/data/DeepGraft/224_128um_v2'
    # data_root = f'/{home}/ylan/DeepGraft/dataset/hdf5/256_256um_split/'
    # label_path = f'/{home}/ylan/DeepGraft_project/code/split_PAS_bin.json'
    # label_path = f'/{home}/ylan/DeepGraft/training_tables/split_debug.json'
    label_path = f'/{home}/ylan/DeepGraft/training_tables/dg_split_PAS_HE_Jones_norm_rest.json'
    output_dir = f'/{data_root}/debug/augments'
    os.makedirs(output_dir, exist_ok=True)

    n_classes = 2

    dataset = ZarrFeatureBagLoader(data_root, label_path=label_path, mode='train', cache=False, data_cache_size=3000, n_classes=n_classes)

    # print(dataset.get_labels(0))
    a = int(len(dataset)* 0.8)
    b = int(len(dataset) - a)
    train_data, valid_data = random_split(dataset, [a, b])
    # print(dataset.dataset)
    # a = int(len(dataset)* 0.8)
    # b = int(len(dataset) - a)
    # train_ds, val_ds = torch.utils.data.random_split(dataset, [a, b])
    # dl = FastTensorDataLoader(dataset, batch_size=1, shuffle=False)
    dl = DataLoader(train_data, batch_size=1, num_workers=8)#, pin_memory=True , sampler=ImbalancedDatasetSampler(train_data)
    # print(len(dl))
    # dl = DataLoader(dataset, batch_size=1, sampler=ImbalancedDatasetSampler(dataset), num_workers=5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()

    # model_ft = resnet50_baseline(pretrained=True)
    # for param in model_ft.parameters():
    #     param.requires_grad = False
    # model_ft.to(device)
    # model = TransMIL(n_classes=n_classes).to(device)
    
    c = 0
    label_count = [0] *n_classes
    epochs = 1
    print(len(dl))
    # start = time.time()

    count = 0
    for i in range(epochs):
        start = time.time()
        for item in tqdm(dl): 
            # if c >= 10:
            #     break
            bag, label, (name, batch_names, patient) = item
            # print(bag.shape)
            # print(len(batch_names))
            # print(label)
            # print(batch_names)
            bag = bag.float().to(device)
            # print(bag)
            # print(name)
            # bag = bag.float().to(device)
            # print(bag.shape)
            # label = label.to(device)
            # with torch.cuda.amp.autocast():
            #     output = model(bag)
            count += 1
            
        end = time.time()
        print('Bag Time: ', end-start)
        print(count)