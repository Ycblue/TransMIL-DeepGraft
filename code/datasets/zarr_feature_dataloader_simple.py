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
    def __init__(self, file_path, label_path, mode, n_classes, cache=False, data_cache_size=5000, max_bag_size=1000):
        super().__init__()

        self.data_info = []
        self.data_cache = {}
        self.slideLabelDict = {}
        self.files = []
        self.data_cache_size = data_cache_size
        self.mode = mode
        self.file_path = file_path
        # self.csv_path = csv_path
        self.label_path = label_path
        self.n_classes = n_classes
        self.max_bag_size = max_bag_size
        self.drop_rate = 0.1
        # self.min_bag_size = 120
        self.empty_slides = []
        self.corrupt_slides = []
        self.cache = cache
        
        # read labels and slide_path from csv
        with open(self.label_path, 'r') as f:
            temp_slide_label_dict = json.load(f)[mode]
            # print(len(temp_slide_label_dict))
            for (x, y) in temp_slide_label_dict:
                x = Path(x).stem
                # x_complete_path = Path(self.file_path)/Path(x)
                for cohort in Path(self.file_path).iterdir():
                    # x_complete_path = Path(self.file_path) / cohort / 'FEATURES_RETCCL' / (str(x) + '.zarr')
                    if self.mode == 'test': #set to test if using GAN output
                        x_path_list = [Path(self.file_path) / cohort / 'FEATURES_RETCCL_2048' / (str(x) + '.zarr')]
                    else:
                        x_path_list = [Path(self.file_path) / cohort / 'FEATURES_RETCCL_2048' / (str(x) + '.zarr')]
                        for i in range(5):
                            x_path_list.append(Path(self.file_path) / cohort / 'FEATURES_RETCCL' / (str(x) + f'_aug{i}.zarr'))
                    # print(x_complete_path)
                    for x_path in x_path_list:
                        if x_path.is_dir():
                            # if len(list(x_complete_path.iterdir())) > self.min_bag_size:
                            # # print(x_complete_path)
                            self.slideLabelDict[x] = y
                            self.files.append(x_path)
        
        # print(self.files)
        home = Path.cwd().parts[1]
        self.slide_patient_dict_path = f'/{home}/ylan/DeepGraft/training_tables/slide_patient_dict.json'
        with open(self.slide_patient_dict_path, 'r') as f:
            self.slide_patient_dict = json.load(f)

        self.feature_bags = []
        self.labels = []
        self.wsi_names = []
        self.name_batches = []
        self.patients = []
        if self.cache:
            for t in tqdm(self.files):
                # zarr_t = str(t) + '.zarr'
                batch, label, (wsi_name, name_batch, patient) = self.get_data(t)

                self.labels.append(label)
                self.feature_bags.append(batch)
                self.wsi_names.append(wsi_name)
                self.name_batches.append(name_batch)
                self.patients.append(patient)

    def get_data(self, file_path):
        
        batch_names=[] #add function for name_batch read out

        wsi_name = Path(file_path).stem
        if wsi_name.split('_')[-1][:3] == 'aug':
            wsi_name = '_'.join(wsi_name.split('_')[:-1])
        # if wsi_name in self.slideLabelDict:
        label = self.slideLabelDict[wsi_name]
        label = torch.as_tensor(label)
        label = torch.nn.functional.one_hot(label, num_classes=self.n_classes)
        patient = self.slide_patient_dict[wsi_name]
        z = zarr.open(file_path, 'r')
        np_bag = np.array(z['data'][:])
        # np_bag = np.array(zarr.open(file_path, 'r')).astype(np.uint8)
        wsi_bag = torch.from_numpy(np_bag)
        batch_coords = torch.from_numpy(np.array(z['coords'][:]))

        # print(wsi_bag.shape)
        bag_size = wsi_bag.shape[0]
        
        # random drop 
        
        bag_idxs = torch.randperm(bag_size)[:int(self.max_bag_size*(1-self.drop_rate))]
        wsi_bag = wsi_bag[bag_idxs, :]
        batch_coords = batch_coords[bag_idxs]
        # print(wsi_bag.shape)
        # name_samples = [batch_names[i] for i in bag_idxs]
        return wsi_bag, label, (wsi_name, batch_coords, patient)
    
    def get_labels(self, indices):
        return [self.labels[i] for i in indices]


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
        # bag_size = self.max_bag_size
        # bag_size = bag.shape[0]
        bag_idxs = torch.randperm(self.max_bag_size)[:int(bag_size*(1-drop_rate))]
        bag_samples = bag[bag_idxs]
        name_samples = [batch_names[i] for i in bag_idxs]

        return bag_samples, name_samples

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        if self.cache:
            label = self.labels[index]
            wsi = self.feature_bags[index]
            # label = Variable(Tensor(label))
            # label = torch.as_tensor(label)
            # label = torch.nn.functional.one_hot(label, num_classes=self.n_classes)
            wsi_name = self.wsi_names[index]
            name_batch = self.name_batches[index]
            patient = self.patients[index]

            #random dropout
            #shuffle

            # feats = Variable(Tensor(feats))
            return wsi, label, (wsi_name, name_batch, patient)
        else:
            t = self.files[index]
            batch, label, (wsi_name, name_batch, patient) = self.get_data(t)
            # label = torch.as_tensor(label)
            # label = torch.nn.functional.one_hot(label, num_classes=self.n_classes)
                # self.labels.append(label)
                # self.feature_bags.append(batch)
                # self.wsi_names.append(wsi_name)
                # self.name_batches.append(name_batch)
                # self.patients.append(patient)

            return batch, label, (wsi_name, name_batch, patient)

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
    label_path = f'/{home}/ylan/DeepGraft/training_tables/dg_limit_20_split_PAS_HE_Jones_norm_rest.json'
    output_dir = f'/{data_root}/debug/augments'
    os.makedirs(output_dir, exist_ok=True)

    n_classes = 2

    dataset = ZarrFeatureBagLoader(data_root, label_path=label_path, mode='train', cache=True, n_classes=n_classes)

    # print(dataset.get_labels(0))
    a = int(len(dataset)* 0.8)
    b = int(len(dataset) - a)
    train_data, valid_data = random_split(dataset, [a, b])
    # print(dataset.dataset)
    # a = int(len(dataset)* 0.8)
    # b = int(len(dataset) - a)
    # train_ds, val_ds = torch.utils.data.random_split(dataset, [a, b])
    # dl = FastTensorDataLoader(dataset, batch_size=1, shuffle=False)
    dl = DataLoader(train_data, batch_size=1, num_workers=8, pin_memory=True)
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
    # print(len(dl))
    # start = time.time()
    for i in range(epochs):
        start = time.time()
        for item in tqdm(dl): 

            # if c >= 10:
            #     break
            bag, label, (name, batch_coords, patient) = item
            # print(bag.shape)
            # print(len(batch_names))
            # print(label)
            # print(batch_coords)
            print(name)
            bag = bag.float().to(device)
            # print(bag.shape)
            # label = label.to(device)
            # with torch.cuda.amp.autocast():
            #     output = model(bag)
            # c += 1
        end = time.time()
        print('Bag Time: ', end-start)