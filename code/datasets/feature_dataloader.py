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
import h5py
import math
# from models import TransMIL



class FeatureBagLoader(data.Dataset):
    def __init__(self, file_path, label_path, mode, n_classes, model='None',cache=False, mixup=False, aug=False, mix_res=False, data_cache_size=5000, max_bag_size=1000, slides=None,**kwargs):
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
        self.max_bag_size = max_bag_size
        self.drop_rate = 0.2
        # self.min_bag_size = 120
        self.empty_slides = []
        self.corrupt_slides = []
        self.cache = cache
        # if self.mode == 'test':
            # self.cache = False
        self.mixup = mixup
        self.aug = aug
        # self.file_path_mix = self.file_path.replace('256', '1024')
        self.missing = []
        self.use_1024 = False
        # print(kwargs.keys())
        if 'feature_extractor' in kwargs.keys():
            self.feature_extractor = kwargs['feature_extractor']
        else: self.feature_extractor = 'FEATURES_RETCCL_2048'
            # print('self.feature_extractor: ', self.feature_extractor)
            # self.fe_name = f'FEATURES_{self.feature_extractor.upper()}_{self.in_features}'
        
        # print('Using FeatureBagLoader: ', self.mode)

        home = Path.cwd().parts[1]
        
        # print(self.label_path)
        project = Path(label_path).parts[4]
        
        # print(project)
        self.slide_patient_dict_path = f'/homeStor1/ylan/data/{project}/training_tables/slide_patient_dict_an_ext.json'
        with open(self.slide_patient_dict_path, 'r') as f:
            self.slide_patient_dict = json.load(f)

        # read labels and slide_path from csv

        with open(self.label_path, 'r') as f:
            json_dict = json.load(f)
            if self.mode == 'fine_tune':
                temp_slide_label_dict = json_dict['train'] + json_dict['test_mixin']
            else: temp_slide_label_dict = json_dict[self.mode]
            # temp_slide_label_dict = json_dict['val']
            # temp_slide_label_dict = json_dict['train'] + json_dict['test_mixin'] # simulate fine tuning
            print('len(temp_slide_label_dict): ', len(temp_slide_label_dict))
            for (x,y) in temp_slide_label_dict:
                
                # test_path = Path(self.file_path)
                # if Path(self.file_path) / 
                # if self.mode != 'test':
                    # x = x.replace('FEATURES_RETCCL_2048', 'FEATURES_RETCCL_2048_HED')
                    # x = x.replace('FEATURES_RETCCL_2048', 'TEST')
                # else:
                    # x = x.replace('FEATURES_RETCCL_2048', 'FEATURES_RESNET50_1024_HED')
                #     # x = x.replace('FEATURES_RETCCL_2048', 'FEATURES_HISTOENCODER_384')
                
                if self.feature_extractor:
                    x = x.replace('FEATURES_RETCCL_2048', self.feature_extractor)
                # x = x.replace('FEATURES_RETCCL_2048', 'FEATURES_HISTOENCODER_384')
                # else:
                    # x = x.replace('Aachen_Biopsy_Slides', 'Aachen_Biopsy_Slides_extended')
                x_name = Path(x).stem
                # print(x)
                # print(x_name)
                if x_name in self.slide_patient_dict.keys():

                    
                    if slides:
                        if x_name in slides:
                            x_path_list = [Path(self.file_path)/x]
                            
                            for x_path in x_path_list:
                                if x_path.exists():
                                    
                                    self.slideLabelDict[x_name] = y
                                    self.labels.append(int(y))
                                    self.files.append(x_path)

                    else:
                        x_path_list = [Path(self.file_path)/x]
                        # x_name = x.stem
                        # x_path_list = [Path(self.file_path)/ x for (x,y) in temp_slide_label_dict]
                        # print(x)
                        # if self.aug:
                        #     for i in range(10):
                        #         aug_path = Path(self.file_path)/f'{x}_aug{i}'
                        #         if self.use_1024:
                        #             aug_path = Path(f'{aug_path}-1024')
                        #         if aug_path.exists():
                        #             # aug_path = Path(self.file_path)/f'{x}_aug{i}'
                        #             x_path_list.append(aug_path)
                        # else: 
                        #     aug_path = Path(self.file_path)/f'{x}_aug0'
                        #     if self.use_1024:
                        #         aug_path = Path(f'{aug_path}-1024')
                        #     if aug_path.exists():
                        #         x_path_list.append(aug_path)
                        # print('x_path_list: ', len(x_path_list))

                        for x_path in x_path_list: 
                            # print(x_path)
                            # print(x_path)
                            # x_path = Path(f'{x_path}.pt')
                            if x_path.exists():
                                self.slideLabelDict[x_name] = y
                                self.labels.append(int(y))
                                self.files.append(x_path)
                            elif Path(str(x_path) + '.zarr').exists():
                                self.slideLabelDict[x] = y
                                self.files.append(str(x_path)+'.zarr')
                            else:
                                self.missing.append(x)

                            ### 1024
                            # x_1024_path = Path(str(x_path).replace('256', '1024'))
                            # if x_1024_path.exists():
                            #     self.slideLabelDict[x_name] = y
                            #     self.labels.append(int(y))
                            #     self.files.append(x_1024_path)

                # print(x, y)
                # x_complete_path = Path(self.file_path)/Path(x)
                # for cohort in Path(self.file_path).iterdir():
                #     # x_complete_path = Path(self.file_path) / cohort / 'FEATURES_RETCCL' / (str(x) + '.zarr')
                #     # if self.mode == 'test': #set to test if using GAN output
                #     #     x_path_list = [Path(self.file_path) / cohort / 'FE' / (str(x) + '.zarr')]
                #     # else:
                #     # x_path_list = [Path(self.file_path) / cohort / 'FEATURES' / (str(x))]
                #     x_path_list = [Path(self.file_path) / cohort / 'FEATURES_RETCCL_2048' / (str(x))]
                #     # if not self.mixup:
                #     for i in range(5):
                #         aug_path = Path(self.file_path) / cohort / 'FEATURES_RETCCL_2048' / (str(x) + f'_aug{i}')
                #         if aug_path.exists():
                #             x_path_list.append(aug_path)
                #     # print(x_complete_path)
                #     for x_path in x_path_list:
                #         # print(x_path)
                        
                #         if x_path.exists():
                #             # print(x_path)
                #             # if len(list(x_complete_path.iterdir())) > self.min_bag_size:
                #             # # print(x_complete_path)
                #             self.slideLabelDict[x] = y
                #             self.files.append(x_path)
                #         elif Path(str(x_path) + '.zarr').exists():
                #             self.slideLabelDict[x] = y
                #             self.files.append(str(x_path)+'.zarr')
                #         else:
                #             self.missing.append(x)
        
        # mix in 10 Slides of Test data
            # if 'test_mixin' in json_dict.keys():
            #     test_slide_label_dict = json_dict['test']
            #     for (x, y) in test_slide_label_dict:
            #         x = Path(x).stem
            #         for cohort in Path(self.file_path).iterdir():
            #             x_path_list = [Path(self.file_path) / cohort / 'FEATURES_RETCCL_2048' / (str(x))]
            #             for x_path in x_path_list:
            #                 if x_path.exists():
            #                     self.slideLabelDict[x] = y
            #                     self.files.append(x_path)
            #                     patient = self.slide_patient_dict[x]
            #                 elif Path(str(x_path) + '.zarr').exists():
            #                     self.slideLabelDict[x] = y
            #                     self.files.append(str(x_path)+'.zarr')



        
        # print('files: ', self.files)
        self.feature_bags = []
        
        self.wsi_names = []
        self.coords = []
        self.patients = []
        # if self.cache:
        #     for t in tqdm(self.files):
        #         # zarr_t = str(t) + '.zarr'
        #         batch, (wsi_name, batch_coords, patient) = self.get_data(t)

        #         # print(label)
        #         # self.labels.append(label)
        #         self.feature_bags.append(batch)
        #         self.wsi_names.append(wsi_name)
        #         self.coords.append(batch_coords)
        #         self.patients.append(patient)
        # else: 
        #     for t in tqdm(self.files):
        #         self.labels = 

    def get_data(self, file_path):
        
        batch_names=[] #add function for name_batch read out

        wsi_name = Path(file_path).stem
        # base_file = file_path.with_suffix('')
        # if wsi_name.split('_')[-1][:3] == 'aug':
        # print(wsi_name)
        
        # print(len(parts[1]))
        # if len(parts[1]) > 2:
        if 'aug' in wsi_name:
            parts = wsi_name.rsplit('_', 1)
            if parts[1].split('-')[0] == '1024':
                wsi_name = parts[0]
            else: 
                wsi_name = '_'.join(parts[:-1])
            # wsi_name = '_'.join(wsi_name.split('_')[:-1])
            
            # base_file = Path('_'.join(str(base_file).split('_')[:-1]))
        # if wsi_name in self.slideLabelDict:
        # label = self.slideLabelDict[wsi_name]
        patient = self.slide_patient_dict[wsi_name]
        # print(file_path)
        with h5py.File(file_path, 'r') as hdf5_file:
            np_bag = hdf5_file['features'][:]
            coords = hdf5_file['coords'][:]
            #info@arno-flachskampf.de

        # if Path(file_path).suffix == '.zarr':
        #     z = zarr.open(file_path, 'r')
        #     np_bag = np.array(z['data'][:])
        #     coords = np.array(z['coords'][:])
        # else:
        #     with h5py.File(file_path, 'r') as hdf5_file:
        #         print(hdf5_file.keys())
        #         np_bag = hdf5_file['features'][:]
        #         coords = hdf5_file['coords'][:]

        # np_bag = torch.load(file_path)
        # z = zarr.open(file_path, 'r')
        # np_bag = np.array(z['data'][:])
        # np_bag = np.array(zarr.open(file_path, 'r')).astype(np.uint8)
        # label = torch.as_tensor(label)
        # label = int(label)
        wsi_bag = torch.from_numpy(np_bag)
        batch_coords = torch.from_numpy(coords)

        return wsi_bag, (wsi_name, batch_coords, patient)
    
    # def get_labels(self, indices):
    #     # for i in indices: 
    #     #     print(self.labels[i])
    #     return [self.labels[i] for i in indices]
        

    def get_labels(self):
        return self.labels


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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        if self.cache:
            label = self.labels[index]
            # print()
            bag = self.feature_bags[index]
            
            
            wsi_name = self.wsi_names[index]
            batch_coords = self.coords[index]
            patient = self.patients[index]
            if self.mode == 'train' or self.mode == 'fine_tune':
                bag_size = bag.shape[0]

                bag_idxs = torch.randperm(bag_size)[:self.max_bag_size]
                # bag_idxs = torch.randperm(bag_size)[:int(self.max_bag_size*(1-self.drop_rate))]
                out_bag = bag[bag_idxs, :]
                if self.mixup:
                    out_bag = self.get_mixup_bag(out_bag)
                    # batch_coords = 
                if out_bag.shape[0] < self.max_bag_size:
                    out_bag = torch.cat((out_bag, torch.zeros(self.max_bag_size-out_bag.shape[0], out_bag.shape[1])))

                # shuffle again
                out_bag_idxs = torch.randperm(out_bag.shape[0])
                out_bag = out_bag[out_bag_idxs]


                # batch_coords only useful for test
                batch_coords = batch_coords[bag_idxs]
                return out_bag, label, (wsi_name, patient)
            else: 
                
                # bag_size = bag.shape[0]
                # bag_idxs = torch.randperm(bag_size)[:self.max_bag_size]
                # out_bag = bag[bag_idxs, :]
                np.random.seed(0)
                draw_size = math.ceil(bag.shape[0] * 0.1)
                # draw_size=10
                draw_indices = np.random.choice(bag.shape[0], draw_size)
                # print(draw_indices)
                bag = bag[draw_indices, :]
                # print('bag: ',bag.shape)
                # print('out_bag: ', out_bag.shape)
                
                
                out_bag = bag
        else:
            t = self.files[index]
            label = self.labels[index]
            bag, (wsi_name, batch_coords, patient) = self.get_data(t)

            if self.mode == 'train' or self.mode == 'fine_tune':
                bag_size = bag.shape[0]

                bag_idxs = torch.randperm(bag_size)[:self.max_bag_size]
                # bag_idxs = torch.randperm(bag_size)[:int(self.max_bag_size*(1-self.drop_rate))]
                out_bag = bag[bag_idxs, :]
                if self.mixup:
                    out_bag = self.get_mixup_bag(out_bag)
                    # batch_coords = 
                if out_bag.shape[0] < self.max_bag_size:
                    out_bag = torch.cat((out_bag, torch.zeros(self.max_bag_size-out_bag.shape[0], out_bag.shape[1])))

                # shuffle again
                out_bag_idxs = torch.randperm(out_bag.shape[0])
                out_bag = out_bag[out_bag_idxs]


                # batch_coords only useful for test
                batch_coords = batch_coords[bag_idxs]
                return out_bag, label, (wsi_name, patient)

            else: 
                # bag_size = bag.shape[0]
                # bag_idxs = torch.randperm(bag_size)[:self.max_bag_size]
                # batch_coords = batch_coords[bag_idxs, :]
                # out_bag = bag[bag_idxs, :]
                # print(out_bag.shape)
                # if out_bag.shape[0] < self.max_bag_size:
                    # out_bag = torch.cat((out_bag, torch.zeros(self.max_bag_size-out_bag.shape[0], out_bag.shape[1])))
                    # out_batch_coords = 
                # bag_size = bag.shape[0]
                # bag_idxs = torch.randperm(bag_size)[:self.max_bag_size]
                # out_bag = bag[bag_idxs, :]
                ###
                # random draw experiment: [10%, 20%, 30%, 40%, 50%]
                np.random.seed(0)
                draw_size = math.ceil(bag.shape[0] * 0.1)
                # draw_size=10
                draw_indices = np.random.choice(bag.shape[0], draw_size)
                # print(draw_indices)
                bag = bag[draw_indices, :]
                # print('bag: ',bag.shape)
                # print('out_bag: ', out_bag.shape)

                out_bag = bag

            # print('feature_dataloader: ', out_bag.shape)

        return out_bag, label, (wsi_name, batch_coords, patient)
        # return out_bag, label, (wsi_name, batch_coords, patient)




if __name__ == '__main__':
    
    from pathlib import Path
    import os
    import time
    # from fast_tensor_dl import FastTensorDataLoader
    from custom_resnet50 import resnet50_baseline
    from sklearn.decomposition import PCA
    
    home = Path.cwd().parts[1]
    train_csv = f'/{home}/ylan/DeepGraft_project/code/debug_train.csv'
    ktx_root = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated'
    rcc_root = f'/{home}/ylan/data/RCC/224_256uM_annotated'
    # data_root = f'/{home}/ylan/DeepGraft/dataset/hdf5/256_256um_split/'
    # label_path = f'/{home}/ylan/DeepGraft_project/code/split_PAS_bin.json'
    # label_path = f'/{home}/ylan/DeepGraft/training_tables/split_debug.json'
    ktx_label_path = f'/{home}/ylan/data/DeepGraft/training_tables/dg_split_PAS_HE_Jones_Grocott_norm_rej_rest_ext_2.json'
    rcc_label_path = f'/{home}/ylan/data/RCC/training_tables/rcc_split_HE_big_three_ext.json'
    # output_dir = f'/{data_root}/debug/augments'
    # os.makedirs(output_dir, exist_ok=True)
    n_classes = 3

    ktx_test_dataset = FeatureBagLoader(ktx_root, label_path=ktx_label_path, mode='test', cache=False, mixup=False, aug=False, n_classes=n_classes)
    rcc_test_dataset = FeatureBagLoader(rcc_root, label_path=rcc_label_path, mode='test', cache=False, mixup=False, aug=False, n_classes=n_classes)
    ktx_dl = DataLoader(ktx_test_dataset, batch_size=1) #
    rcc_dl = DataLoader(rcc_test_dataset, batch_size=1) #

    epochs = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bag_size_dict = {'RCC': [], 'KTX': []}

    patient_dict = {}

    for i in range(epochs):
        start = time.time()
        for item in tqdm(ktx_dl): 
            bag, label, (name, bc, patient) = item
            print(bag.shape)
            bag_size_dict['KTX'].append(bag.shape)
        end = time.time()
    for i in range(epochs):
        start = time.time()
        for item in tqdm(rcc_dl): 
            bag, label, (name, bc, patient) = item
            print(bag.shape)
            bag_size_dict['RCC'].append(bag.shape)
        end = time.time()
        
    print(bag_size_dict)
    # print(len(patient_dict.keys()))
    
    # for k in patient_dict.keys():
    #     # print(patient_dict[k])
    #     if len(patient_dict[k]) > 3:
    #         print(patient_dict[k])
        # break
 
    output_path = '/homeStor1/ylan/npj_sus_data/bag_size_dict.json'
    json.dump(bag_size_dict, open(output_path, 'w'))
    # with 
    # pca_tensor = torch.cat(pca_tensor, dim=0)
    # print(pca_tensor.shape)
    # x_train = pca.fit_transform(pca_tensor.squeeze())
    # print(pca.n_components_)
    # print(pca.components_)
    # print(x_train.shape)