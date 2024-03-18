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
# from models import TransMIL



class FeatureBagLoader(data.Dataset):
    def __init__(self, file_path, label_path, mode, n_classes, model='None',cache=False, mixup=False, aug=False, mix_res=False, data_cache_size=5000, max_bag_size=1000, **kwargs):
        super().__init__()

        self.data_info = []
        self.data_cache = {}
        self.slideLabelDict = {}
        self.files = []
        self.labels = []
        self.data_cache_size = data_cache_size
        self.mode = mode
        self.file_path = file_path
        # print(self.file_path)
        self.mixed_file_paths = [self.file_path, self.file_path.replace('256','1024').replace('224', '512')] #, self.file_path.replace('256', '128')
        # self.mixed_file_paths = [self.file_path]
        # print('Paths: ', self.mixed_file_paths)
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
        self.feature_extractor = 'FEATURES_HISTOENCODER_384'
        if 'feature_extractor' in kwargs.keys():
            self.feature_extractor = kwargs['feature_extractor']
            # print('self.feature_extractor: ', self.feature_extractor)
            # self.fe_name = f'FEATURES_{self.feature_extractor.upper()}_{self.in_features}'
        # print(self.feature_extractor)

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
                #     # x = x.replace('FEATURES_RETCCL_2048', 'FEATURES_HISTOENCODER_384')
                
                if self.feature_extractor:
                    x = x.replace('FEATURES_RETCCL_2048', self.feature_extractor)
                # x = x.replace('FEATURES_RETCCL_2048', 'FEATURES_HISTOENCODER_384')
                # else:
                    # x = x.replace('Aachen_Biopsy_Slides', 'Aachen_Biopsy_Slides_extended')
                # print(x)
                x_name = Path(x).stem
                # print(x_name)
                if x_name in self.slide_patient_dict.keys():
                    # x_path_list = [Path(self.file_path)/x]
                    if self.mode == 'train':
                        x_path_list = [Path(f)/x for f in self.mixed_file_paths]
                    else:
                        x_path_list = [Path(self.file_path)/x]

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
                # 
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



        

        self.feature_bags = []
        
        self.wsi_names = []
        self.coords = []
        self.patients = []
        if self.cache:
            for t in tqdm(self.files):
                # zarr_t = str(t) + '.zarr'
                batch, (wsi_name, batch_coords, patient) = self.get_data(t)

                # print(label)
                # self.labels.append(label)
                self.feature_bags.append(batch)
                self.wsi_names.append(wsi_name)
                self.coords.append(batch_coords)
                self.patients.append(patient)
        # else: 
        #     for t in tqdm(self.files):
        #         self.labels = 
        print(len(self.files))

    def get_data(self, file_path):
        
        batch_names=[] #add function for name_batch read out

        wsi_name = Path(file_path).stem
        # base_file = file_path.with_suffix('')
        # if wsi_name.split('_')[-1][:3] == 'aug':
        parts = wsi_name.rsplit('_', 1)
        if parts[1][:3] == 'aug':
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
            # print(label)
            # if label == 2:
            #     label = torch.LongTensor([0,1])

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
    data_root = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated'
    # data_root = f'/{home}/ylan/DeepGraft/dataset/hdf5/256_256um_split/'
    # label_path = f'/{home}/ylan/DeepGraft_project/code/split_PAS_bin.json'
    # label_path = f'/{home}/ylan/DeepGraft/training_tables/split_debug.json'
    label_path = f'/{home}/ylan/data/DeepGraft/training_tables/dg_split_PAS_HE_Jones_Grocott_norm_rest_ext.json'
    # output_dir = f'/{data_root}/debug/augments'
    # os.makedirs(output_dir, exist_ok=True)
    n_classes = 2

    train_dataset = FeatureBagLoader(data_root, label_path=label_path, mode='train', cache=False, mixup=True, aug=False, n_classes=n_classes, max_bag_size=200)
    print('train_dataset: ', len(train_dataset))

    train_dl = DataLoader(train_dataset, batch_size=1) #

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # scaler = torch.cuda.amp.GradScaler()

    # model_ft = resnet50_baseline(pretrained=True)
    # for param in model_ft.parameters():
    #     param.requires_grad = False
    # model_ft.to(device)
    # model = TransMIL(n_classes=n_classes).to(device)
    

    # print(dataset.get_labels(np.arange(len(dataset))))
    pca = PCA(0.95)
    c = 0
    label_count = [0] *n_classes
    epochs = 1
    # print(len(dl))
    # start = time.time()

    pca_tensor = []

    # for i in range(epochs):
    #     start = time.time()
    #     for item in tqdm(train_dl): 
    #         if c >= 1000:
    #             break
    #         # print(item)
    #         bag, label, (name, patient) = item
    #         print(label)
    #         if label == 2:
    #             label = torch.LongTensor([0,1])
    #         y_onehot = torch.nn.functional.one_hot(label, num_classes=2)
    #         print(y_onehot)
    #         y_onehot = y_onehot.sum(dim=0).float()
    #         print(y_onehot)
    #         # print(bag.shape)
            
    #         # print(pca.explained_variance_ratio_)
    #         # print(pca.n_components_)
    #         # train_pca = pca.transform(x_train)
    #         # print(x_train.shape)
    #         # pca_tensor.append(bag.squeeze())
            
    #         c += 1
    #     end = time.time()
    #     print('Bag Time: ', end-start)



    # pca_tensor = torch.cat(pca_tensor, dim=0)
    # print(pca_tensor.shape)
    # x_train = pca.fit_transform(pca_tensor.squeeze())
    # print(pca.n_components_)
    # print(pca.components_)
    # print(x_train.shape)