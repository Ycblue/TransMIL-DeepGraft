import pandas as pd

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn.functional import one_hot
import torch.nn.functional as F
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
import h5py
import math

# from models import TransMIL



class LocalFeatureBagLoader(data.Dataset):
    def __init__(self, file_path, label_path, mode, n_classes, model='None',cache=False, mixup=False, aug=False, mix_res=False, data_cache_size=5000, max_size=50, max_bag_size=0,device='cuda'):
        super().__init__()

        self.data_info = []
        self.data_cache = {}
        self.slideLabelDict = {}
        self.files = []
        self.labels = []
        self.data_cache_size = data_cache_size
        self.mode = mode
        self.file_path = file_path
        # self.csv_path = csv_path
        self.label_path = label_path
        self.n_classes = n_classes
        # self.max_bag_size = max_bag_size
        self.drop_rate = 0.2
        # self.min_bag_size = 120
        self.empty_slides = []
        self.corrupt_slides = []
        self.cache = cache
        # self.mixup = mixup
        self.aug = False
        # self.file_path_mix = self.file_path.replace('256', '1024')
        self.missing = []
        self.use_1024 = False
        self.max_size = max_size
        self.device = device
        self.dist = []


        # print('Using FeatureBagLoader: ', self.mode)

        home = Path.cwd().parts[1]
        
        self.slide_patient_dict_path = f'/{home}/ylan/data/DeepGraft/training_tables/slide_patient_dict_an_ext.json'
        with open(self.slide_patient_dict_path, 'r') as f:
            self.slide_patient_dict = json.load(f)

        # read labels and slide_path from csv

        with open(self.label_path, 'r') as f:
            json_dict = json.load(f)
            if self.mode == 'fine_tune':
                temp_slide_label_dict = json_dict['train'] + json_dict['test_mixin']
            else: temp_slide_label_dict = json_dict[self.mode]
            # temp_slide_label_dict = json_dict['train']
            # temp_slide_label_dict = json_dict['train'] + json_dict['test_mixin'] # simulate fine tuning
            
            for (x,y) in temp_slide_label_dict:
                
                # test_path = Path(self.file_path)
                # if Path(self.file_path) / 
                # if self.mode != 'test':
                    # x = x.replace('FEATURES_RETCCL_2048', 'FEATURES_RETCCL_2048_HED')
                    # x = x.replace('FEATURES_RETCCL_2048', 'TEST')
                # else:
                    # x = x.replace('FEATURES_RETCCL_2048', 'FEATURES_RESNET50_1024_HED')
                    
                # x = x.replace('FEATURES_RETCCL_2048', 'FEATURES_CTRANSPATH_768')
                # else:
                    # x = x.replace('Aachen_Biopsy_Slides', 'Aachen_Biopsy_Slides_extended')
                x_name = Path(x).stem
                # print(x)
                # print(x_name)
                if x_name in self.slide_patient_dict.keys():
                    x_path_list = [Path(self.file_path)/x]
                    # x_name = x.stem
                    # x_path_list = [Path(self.file_path)/ x for (x,y) in temp_slide_label_dict]
                    # print(x)
                    if self.aug:
                        for i in range(10):
                            aug_path = Path(self.file_path)/f'{x}_aug{i}'
                            if self.use_1024:
                                aug_path = Path(f'{aug_path}-1024')
                            if aug_path.exists():
                                # aug_path = Path(self.file_path)/f'{x}_aug{i}'
                                x_path_list.append(aug_path)
                    else: 
                        aug_path = Path(self.file_path)/f'{x}_aug0'
                        if self.use_1024:
                            aug_path = Path(f'{aug_path}-1024')
                        if aug_path.exists():
                            x_path_list.append(aug_path)
                    # print('x_path_list: ', len(x_path_list))
                    for x_path in x_path_list: 
                        # print(x_path)
                        # print(x_path)
                        # x_path = Path(f'{x_path}.pt')
                        if x_path.exists():
                            label = int(y)
                            wsi_name = x_name
                            patient = self.slide_patient_dict[wsi_name]
                            idx = -1
                            # self.slideLabelDict[x_name] = y
                            self.labels.append(int(y))
                            # self.files.append(x_path)
                            self.data_info.append({'data_path': x_path, 'label': label, 'name': wsi_name, 'patient': patient,'cache_idx': idx})
                        # elif Path(str(x_path) + '.zarr').exists():
                        #     self.slideLabelDict[x] = y
                        #     self.files.append(str(x_path)+'.zarr')
                        # else:
                        #     self.missing.append(x)

        self.feature_bags = []
        
        self.wsi_names = []
        self.coords = []
        self.patients = []
        # if self.cache:
        #     for t in tqdm(self.files):
        #         # zarr_t = str(t) + '.zarr'
        #         batch, (wsi_name, batch_coords, patient) = self.get_data(t)
        #         batch = batch.to(self.device)
        #         # print(label)
        #         # self.labels.append(label)
        #         self.feature_bags.append(batch)
        #         self.wsi_names.append(wsi_name)
        #         self.coords.append(batch_coords)
        #         self.patients.append(patient)
        # else: 
        #     for t in tqdm(self.files):
        #         self.labels = 

    def get_data(self, i):

        fp = self.data_info[i]['data_path']
        if fp not in self.data_cache:
            self._load_data(fp)

        cache_idx = self.data_info[i]['cache_idx']
        label = self.data_info[i]['label']
        wsi_name = self.data_info[i]['name']
        patient = self.data_info[i]['patient']

        return self.data_cache[fp][cache_idx], label, wsi_name, patient
    
    def get_dist(self):
        return self.dist
    
    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):

        # if self.cache:
        #     label = self.labels[index]
        #     bag = self.feature_bags[index]
            
        #     wsi_name = self.wsi_names[index]
        #     batch_coords = self.coords[index]
        #     patient = self.patients[index]
        # else:
        # t = self.files[index]
        # label = self.labels[index]
        (bag, torch_coords), label, wsi_name, patient = self.get_data(index)

        # if self.mode == 'train' or self.mode == 'fine_tune':
        # bag_size = bag.shape[0]

        out_bag = torch.permute(bag, (2,0,1))

        if self.mode == 'train':
            return out_bag, label, (wsi_name, patient)
        elif self.mode == 'val':
            return out_bag, label, (wsi_name, torch_coords, patient)
        else:
            return out_bag, label, (wsi_name, torch_coords, patient)
        # return out_bag, label, (wsi_name, batch_coords, patient)

    # def _add_data_infos(self, file_path, cache, slide_patient_dict):

    #     wsi_name = Path(file_path).stem
    #     if wsi_name in self.slideLabelDict:
    #         # if wsi_name[:2] != 'RU': #skip RU because of container problems in dataset
    #         label = self.slideLabelDict[wsi_name]
    #         patient = slide_patient_dict[wsi_name]
    #         idx = -1
    #         self.data_info.append({'data_path': file_path, 'label': label, 'name': wsi_name, 'patient': patient,'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        batch_names=[] #add function for name_batch read out

        # wsi_name = Path(file_path).stem
        # base_file = file_path.with_suffix('')
        # if wsi_name.split('_')[-1][:3] == 'aug':
        # parts = wsi_name.rsplit('_', 1)
        # if parts[1][:3] == 'aug':
        #     if parts[1].split('-')[0] == '1024':
        #         wsi_name = parts[0]
        #     else: 
        #         wsi_name = '_'.join(parts[:-1])
        # patient = self.slide_patient_dict[wsi_name]
        # print(file_path)
        with h5py.File(file_path, 'r') as hdf5_file:
            np_bag = hdf5_file['features'][:]
            coords = hdf5_file['coords'][:]

        # Order by coordinates!
        torch_bag = torch.from_numpy(np_bag)
        torch_coords = torch.from_numpy(coords)
        #get max coords for assembly    
        x_max = torch.max(torch_coords[:,0])
        y_max = torch.max(torch_coords[:,1])
        x_min = torch.min(torch_coords[:,0])
        y_min = torch.min(torch_coords[:,1])

        self.dist.append((x_max-x_min, y_max-y_min))

        # print(x_min, x_max)
        if x_max-x_min > self.max_size:
            x_start_pos = torch.randint(x_min, x_max-self.max_size, [1])
            x_end_pos = x_start_pos + self.max_size
        else: 
            x_start_pos = x_min 
            x_end_pos = x_max

        if y_max-y_min > self.max_size:
            y_start_pos = torch.randint(y_min, y_max-self.max_size, [1])
            y_end_pos = y_start_pos + self.max_size
        else: 
            y_start_pos = y_min
            y_end_pos = y_max

        slide_3d = torch.zeros([self.max_size, self.max_size, 2048]) #feature vector size


        # Define a size for a 3D feature stack! 
        for c, patch_features in zip(torch_coords, torch_bag):
            x = c[0]
            y = c[1]
            if x > x_start_pos and x < x_end_pos and y > y_start_pos and y < y_end_pos:

                # print(x,x_start_pos, x_end_pos)
                # print(y,y_start_pos, y_end_pos)

                slide_3d[x-x_start_pos, y-y_start_pos, :] = patch_features

        # slide_3d = slide_3d[500, 500, :]
        # limit size of slide to self.max_size


        # if slide_3d.shape[0] > self.max_size:
        #     slide_3d = slide_3d[self.max_size, :, :]
        # if slide_3d.shape[1] > self.max_size:
        #     slide_3d = slide_3d[:, self.max_size, :]

        # padding_x1 = math.floor((self.max_size - slide_3d.shape[0])/2)
        # padding_x2 = self.max_size - padding_x1 - slide_3d.shape[0]
        # padding_y1 = math.floor((self.max_size - slide_3d.shape[1])/2)
        # padding_y2 = self.max_size - padding_y1 - slide_3d.shape[1]
        
        # padding = (0, 0, padding_y1, padding_y2, padding_x1, padding_x2)
        # wsi_bag = F.pad(slide_3d, padding, mode='constant') #pad to max_size 
        # print(slide_3d.shape)
        wsi_bag = slide_3d
        # return wsi_bag, (wsi_name, batch_coords, patient)

        # add data to cache, get id for cache entry
        idx = self._add_to_cache((wsi_bag, torch_coords), file_path)
        file_idx = next(i for i,v in enumerate(self.data_info) if v['data_path'] == file_path)
        self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            # self.data_info = [{'data_path': di['data_path'], 'label': di['label'], 'shape': di['shape'], 'name': di['name'], 'cache_idx': -1} if di['data_path'] == removal_keys[0] else di for di in self.data_info]
            self.data_info = [{'data_path': di['data_path'], 'label': di['label'], 'name': di['name'], 'patient':di['patient'], 'cache_idx': -1} if di['data_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, data_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if data_path not in self.data_cache:
            self.data_cache[data_path] = [data]
        else:
            self.data_cache[data_path].append(data)
        return len(self.data_cache[data_path]) - 1

    def get_name(self, i):
        # name = self.get_data_infos(type)[i]['name']
        name = self.data_info[i]['name']
        return name

    # def get_labels(self, indices):

    #     return [self.data_info[i]['label'] for i in indices]
        # return self.slideLabelDict.values()


    def to_fixed_size_bag(self, bag, names, bag_size: int = 512):

        bag_idxs = torch.randperm(bag.shape[0])[:bag_size]
        bag_samples = bag[bag_idxs]
        name_samples = [names[i] for i in bag_idxs]

        return bag_samples, name_samples, min(bag_size, len(bag))

    def data_dropout(self, bag, batch_names, drop_rate):
        # bag_size = self.max_bag_size
        bag_size = bag.shape[0]
        bag_idxs = torch.randperm(self.max_bag_size)[:int(bag_size*(1-drop_rate))]
        bag_samples = bag[bag_idxs]
        name_samples = [batch_names[i] for i in bag_idxs]

        return bag_samples, name_samples

    def get_mixup_bag(self, bag):

        bag_size = bag.shape[0]

        a = torch.rand([bag_size])
        b = 0.6
        rand_x = torch.randint(0, bag_size, [bag_size,])
        rand_y = torch.randint(0, bag_size, [bag_size,])

        bag_x = bag[rand_x, :]
        bag_y = bag[rand_y, :]


        temp_bag = (bag_x.t()*a).t() + (bag_y.t()*(1.0-a)).t()

        if bag_size < self.max_bag_size:
            diff = self.max_bag_size - bag_size
            bag_idxs = torch.randperm(bag_size)[:diff]
            
            mixup_bag = torch.cat((bag, temp_bag[bag_idxs, :]))
        else:
            random_sample_list = torch.rand(bag_size)
            mixup_bag = [bag[i] if random_sample_list[i] else temp_bag[i] > b for i in range(bag_size)] #make pytorch native?!
            mixup_bag = torch.stack(mixup_bag)

        return mixup_bag

if __name__ == '__main__':
    
#%%
    from pathlib import Path
    import os
    import time
    # from fast_tensor_dl import FastTensorDataLoader
    from custom_resnet50 import resnet50_baseline
    from torchvision import models
    import matplotlib.pyplot as plt
    
    home = Path.cwd().parts[1]
    train_csv = f'/{home}/ylan/DeepGraft_project/code/debug_train.csv'
    data_root = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated'
    # data_root = f'/{home}/ylan/DeepGraft/dataset/hdf5/256_256um_split/'
    # label_path = f'/{home}/ylan/DeepGraft_project/code/split_PAS_bin.json'
    # label_path = f'/{home}/ylan/DeepGraft/training_tables/split_debug.json'
    label_path = f'/{home}/ylan/data/DeepGraft/training_tables/dg_split_PAS_HE_Jones_Grocott_norm_rest_ext.json'
    # label_path = f'/{home}/ylan/data/DeepGraft/training_tables/dg_limit_20_split_PAS_HE_Jones_norm_rest.json'
    # output_dir = f'/{data_root}/debug/augments'
    # os.makedirs(output_dir, exist_ok=True)
    n_classes = 2

    train_dataset = LocalFeatureBagLoader(data_root, label_path=label_path, mode='train', cache=False, n_classes=n_classes)
    print('train_dataset: ', len(train_dataset))

    def simple_collate(data):
        # print(data[0])
        bags = [i[0] for i in data]
        labels = [i[1] for i in data]
        name = [i[2][0] for i in data]
        patient = [i[2][1] for i in data]
        bags = torch.stack(bags)
        labels = torch.Tensor(np.stack(labels, axis=0)).long()
        return bags, labels, (name, patient)


    train_dl = DataLoader(train_dataset, batch_size=5, sampler=ImbalancedDatasetSampler(train_dataset), collate_fn=simple_collate) #

    print('train_dl: ', len(train_dl))

    # train_dataset = FeatureBagLoader(data_root, label_path=label_path, mode='train', cache=False, n_classes=n_classes, model='None', aug=True, mixup=True)
    # test_dataset = FeatureBagLoader(data_root, label_path=label_path, mode='test', cache=False, n_classes=n_classes, model='None', aug=True, mixup=True)
    # test_dl = DataLoader(test_dataset, batch_size=1)
    # print('test_dl: ', len(test_dl))

    # # print(dataset.get_labels(0))
    # # a = int(len(dataset)* 0.8)
    # # b = int(len(dataset) - a)
    # # train_data, valid_data = random_split(dataset, [a, b])

    # val_dataset = FeatureBagLoader(data_root, label_path=label_path, mode='val', cache=False, mixup=False, aug=False, n_classes=n_classes, model='None')
    # valid_dl = DataLoader(val_dataset, batch_size=1)
    # print('valid_dl: ', len(valid_dl))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # scaler = torch.cuda.amp.GradScaler()

    # model_ft = resnet50_baseline()
    # model = models.resnet50(weights='IMAGENET1K_V1')
    # model.conv1 = torch.nn.Sequential(
    #     torch.nn.Conv2d(2048, 1024, kernel_size=(7,7), stride=(2,2)),
    #     torch.nn.BatchNorm2d(1024),
    #     torch.nn.ReLU,
    #     torch.nn.MaxPool2d(kernel_size=3)
    # )
    # model.conv1 = torch.nn.Conv2d(2048, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3))
    # # print(model)
    # model.fc = torch.nn.Sequential(
    #     torch.nn.Linear(model.fc.in_features, n_classes),
    # )

    # for param in model_ft.parameters():
    #     param.requires_grad = False
    # print(list(model_ft.children()))
    # model_ft.fc = model_ft.
    # model_ft.to(device)
    

    # model = TransMIL(n_classes=n_classes).to(device)
    

    # print(dataset.get_labels(np.arange(len(dataset))))

    c = 0
    # label_count = [0] *n_classes
    epochs = 1
    # # print(len(dl))
    # # start = time.time()
    print(device)
    for i in range(epochs):
        start = time.time()
        for item in tqdm(train_dl): 
            # print(item)
            bag, label, (name, patient) = item
            print(bag.shape)
            bag.to(device)
            # pred = model(bag)
            c += 1
        end = time.time()
        print('Bag Time: ', end-start)


    # dist_array = train_dataset.get_dist()
    # x_dist = [x[0] for x in dist_array]
    # y_dist = [x[1] for x in dist_array]

    # h_x = np.histogram(x_dist)
    # h_y = np.histogram(y_dist)

    # print(h_x)
    # print(h_y)


    # _ = plt.hist(h_x, bins='auto')
    # plt.show()

    # _ = plt.hist(h_y, bins='auto')
    # plt.show()

    
