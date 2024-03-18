# import pandas as pd

import numpy as np
import torch
from torch import Tensor
from torch.utils import data
from torch.utils.data import random_split, DataLoader
from torch.autograd import Variable
from torch.nn.functional import one_hot
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import pandas as pd
from sklearn.utils import shuffle
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
import json
from imgaug import augmenters as iaa
from torchsampler import ImbalancedDatasetSampler
from .utils import myTransforms
from transformers import ViTFeatureExtractor
import torchvision.models as models
import torch.nn as nn
import random


class JPGBagLoader(data_utils.Dataset):
    def __init__(self, file_path, label_path, mode, n_classes, data_cache_size=100, max_bag_size=1000, cache=False, mixup=False, aug=False, model='', **kargs):
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
        self.min_bag_size = 50
        self.empty_slides = []
        self.corrupt_slides = []
        self.cache = False
        self.labels = []
        if model == 'inception':
            self.size = 299
        # elif model == 'vit':
        #     self.size = 384
        else: self.size = 224
        
        home = Path.cwd().parts[1]
        self.slide_patient_dict_path = Path(self.label_path).parent / 'slide_patient_dict_an_ext.json'
        # self.slide_patient_dict_path = f'/{home}/ylan/data/DeepGraft/training_tables/slide_patient_dict_an.json'
        with open(self.slide_patient_dict_path, 'r') as f:
            self.slide_patient_dict = json.load(f)
        
        # read labels and slide_path from csv
        with open(self.label_path, 'r') as f:
            json_dict = json.load(f)
            temp_slide_label_dict = json_dict[self.mode]
            # temp_slide_label_dict = json_dict['train'] #export train metrics
            print(len(temp_slide_label_dict))
            for (x,y) in temp_slide_label_dict:
                x = x.replace('FEATURES_RETCCL_2048', 'BLOCKS')
                x_name = Path(x).stem
                x_path_list = [Path(self.file_path)/x]
                if x_name in self.slide_patient_dict.keys():
                    for x_path in x_path_list:
                        if x_path.exists():
                            # print(len(list(x_path.glob('*'))))

                            self.slideLabelDict[x_name] = y
                            self.labels += [int(y)]*len(list(x_path.glob('*')))
                            # self.labels.append(int(y))
                            for patch in x_path.iterdir():
                                self.files.append((patch, x_name, y))
        random.shuffle(self.files)
        # with open(self.label_path, 'r') as f:
        #     temp_slide_label_dict = json.load(f)[mode]
        #     print(len(temp_slide_label_dict))
        #     for (x, y) in temp_slide_label_dict:
        #         x = Path(x).stem 
        #         # x_complete_path = Path(self.file_path)/Path(x)
        #         for cohort in Path(self.file_path).iterdir():
        #             x_complete_path = Path(self.file_path) / cohort / 'BLOCKS' / Path(x)
        #             if x_complete_path.is_dir():
        #                 if len(list(x_complete_path.iterdir())) > self.min_bag_size:
        #                 # print(x_complete_path)
        #                     self.slideLabelDict[x] = y
        #                     self.files.append(x_complete_path)
        #                 else: self.empty_slides.append(x_complete_path)
        
        



        self.color_transforms = myTransforms.Compose([
            myTransforms.ColorJitter(
                brightness = (0.65, 1.35), 
                contrast = (0.5, 1.5),
                # saturation=(0, 2), 
                # hue=0.3,
                ),
            # myTransforms.RandomChoice([myTransforms.ColorJitter(saturation=(0, 2), hue=0.3),
            #                             myTransforms.HEDJitter(theta=0.05)]),
            myTransforms.HEDJitter(theta=0.005),
            
        ])
        # self.color_transforms = myTransforms.Compose([
        #     myTransforms.Grayscale(num_output_channels=3)
        # ])
        self.train_transforms = myTransforms.Compose([
            myTransforms.RandomChoice([myTransforms.RandomHorizontalFlip(p=0.5),
                                        myTransforms.RandomVerticalFlip(p=0.5),
                                        myTransforms.AutoRandomRotation()]),
        
            myTransforms.RandomGaussBlur(radius=[0.5, 1.5]),
            myTransforms.RandomAffineCV2(alpha=0.1),
            # myTransforms.RandomElastic(alpha=2, sigma=0.06),
        ])

        self.resize_transforms = transforms.Resize((self.size,self.size), transforms.InterpolationMode.BICUBIC)

        # sometimes = lambda aug: iaa.Sometimes(0.5, aug, name="Random1")
        # sometimes2 = lambda aug: iaa.Sometimes(0.2, aug, name="Random2")
        # sometimes3 = lambda aug: iaa.Sometimes(0.9, aug, name="Random3")
        # sometimes4 = lambda aug: iaa.Sometimes(0.9, aug, name="Random4")
        # sometimes5 = lambda aug: iaa.Sometimes(0.9, aug, name="Random5")

        # self.resize_transforms = iaa.Sequential([
        #     iaa.Resize({'height': size, 'width': size}),
        #     # iaa.Resize({'height': 299, 'width': 299}),
        # ], name='resizeAug')
        # # self.resize_transforms = transforms.Resize(size=(299,299))

        # self.train_transforms = iaa.Sequential([
        #     iaa.AddToHueAndSaturation(value=(-30, 30), name="MyHSV"), #13
        #     sometimes2(iaa.GammaContrast(gamma=(0.85, 1.15), name="MyGamma")),
        #     iaa.Fliplr(0.5, name="MyFlipLR"),
        #     iaa.Flipud(0.5, name="MyFlipUD"),
        #     sometimes(iaa.Rot90(k=1, keep_size=True, name="MyRot90")),
        #     # iaa.OneOf([
        #     #     sometimes3(iaa.PiecewiseAffine(scale=(0.015, 0.02), cval=0, name="MyPiece")),
        #     #     sometimes4(iaa.ElasticTransformation(alpha=(100, 200), sigma=20, cval=0, name="MyElastic")),
        #     #     sometimes5(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, rotate=(-45, 45), shear=(-4, 4), cval=0, name="MyAffine"))
        #     # ], name="MyOneOf")

        # ], name="MyAug")
        # if self.model == 'vit':
        #     model_name_or_path = 'models/ckpt/vit-base-patch16-224-in21k/'
        #     self.val_transforms = ViTFeatureExtractor.from_pretrained(model_name_or_path)
        self.val_transforms = transforms.Compose([
            # 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            # RangeNormalization(),
        ])


        

    def get_data(self, query):
        
        patch_path, wsi_name, label = query

        # img = np.asarray(Image.open(patch_path)).astype(np.uint8)
        img = Image.open(patch_path)
        # img=  
        # img = img.resize((self.size, self.size), Image.BICUBIC)

        # img = np.moveaxis(img, 2, 0)
        # print(img.shape)
        # img = torch.from_numpy(img)
        tile_name = Path(patch_path).stem
        # patient = tile_name.rsplit('_', 1)[0]
        patient = self.slide_patient_dict[wsi_name]

        return img, label, (wsi_name, tile_name, patient)
    
    def get_labels(self, indices):
        return [self.labels[i] for i in indices]


    def to_fixed_size_bag(self, bag, bag_size: int = 512):

        #duplicate bag instances unitl 

        bag_idxs = torch.randperm(bag.shape[0])[:bag_size]
        bag_samples = bag[bag_idxs]
        # name_samples = [names[i] for i in bag_idxs]

        # bag_sample_names = [bag_names[i] for i in bag_idxs]
        # q, r  = divmod(bag_size, bag_samples.shape[0])
        # if q > 0:
        #     bag_samples = torch.cat([bag_samples]*q, 0)

        # self_padded = torch.cat([bag_samples, bag_samples[:r,:, :, :]])

        # zero-pad if we don't have enough samples
        zero_padded = torch.cat((bag_samples,
                                torch.zeros(bag_size-bag_samples.shape[0], bag_samples.shape[1], bag_samples.shape[2], bag_samples.shape[3])))

        return zero_padded, min(bag_size, len(bag))

    def data_dropout(self, bag, drop_rate):
        bag_size = bag.shape[0]
        bag_idxs = torch.randperm(bag_size)[:int(bag_size*(1-drop_rate))]
        bag_samples = bag[bag_idxs]
        # name_samples = [batch_names[i] for i in bag_idxs]

        return bag_samples

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

    
        if self.cache:
            label = self.labels[index]
            wsi = self.features[index]
            label = int(label)
            wsi_name = self.wsi_names[index]
            tile_name = self.name_batches[index]
            patient = self.patients[index]
            # feats = Variable(Tensor(feats))
            return wsi, label, (wsi_name, tile_name, patient)
        else:
            t = self.files[index]
            # label = self.labels[index]
            if self.mode=='train':
                # t = self.files[index]
                # label = self.labels[index]
                img, label, (wsi_name, tile_name, patient) = self.get_data(t)
                # save_img(img, f'{tile_name}_original')
                # if self.model == 'vit':
                #     img = self.val_transforms(img, return_tesnors='pt')
                # else:
                img = self.resize_transforms(img)
                # img = self.color_transforms(img)
                img = self.train_transforms(img)

                # save_img(img, f'{tile_name}')

                img = self.val_transforms(img.copy())
                
            else:
                img, label, (wsi_name, tile_name, patient) = self.get_data(t)
                # label = Variable(Tensor(label))
                # seq_img_d = self.train_transforms.to_deterministic()
                # seq_img_resize = self.resize_transforms.to_deterministic()
                # img = img.numpy().astype(np.uint8)
                # if self.model == 'vit':
                #     img = self.val_transforms(img, return_tesnors='pt')
            # else:
                img = self.resize_transforms(img)
                # img = np.moveaxis(img, 0, 2)
                img = self.val_transforms(img)

            return img, label, (wsi_name, tile_name, patient)

def save_img(img, comment):
    home = Path.cwd().parts[1]
    outputPath = f'/{home}/ylan/data/DeepGraft/224_128uM_annotated/debug/augments_2'
    img = img.convert('RGB')
    img.save(f'{outputPath}/{comment}.jpg')


class LazyJPGBagLoader(data_utils.Dataset):
    def __init__(self, file_path, label_path, mode, n_classes, data_cache_size=100, max_bag_size=1000, cache=False, mixup=False, aug=False, model='', **kargs):
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
        self.min_bag_size = 50
        self.empty_slides = []
        self.corrupt_slides = []
        self.cache = cache
        self.labels = []
        if model == 'inception':
            self.size = 299
        # elif model == 'vit':
        #     size = 384
        else: self.size = 224
        self.cached_data = []
        
        home = Path.cwd().parts[1]
        self.slide_patient_dict_path = Path(self.label_path).parent / 'slide_patient_dict_an_ext.json'
        # self.slide_patient_dict_path = f'/{home}/ylan/data/DeepGraft/training_tables/slide_patient_dict_an.json'
        with open(self.slide_patient_dict_path, 'r') as f:
            self.slide_patient_dict = json.load(f)
        
        # read labels and slide_path from csv
        with open(self.label_path, 'r') as f:
            json_dict = json.load(f)
            temp_slide_label_dict = json_dict[self.mode]
            # temp_slide_label_dict = json_dict['train'] #export train metrics
            # print(len(temp_slide_label_dict))
            for (x,y) in temp_slide_label_dict:
                x = x.replace('FEATURES_RETCCL_2048', 'BLOCKS')
                x_name = Path(x).stem
                x_path_list = [Path(self.file_path)/x]
                if x_name in self.slide_patient_dict.keys():
                    for x_path in x_path_list:
                        if x_path.exists():
                            # print(len(list(x_path.glob('*'))))

                            self.slideLabelDict[x_name] = y
                            self.labels += [int(y)]*len(list(x_path.glob('*')))
                            # self.labels.append(int(y))
                            for patch_path in x_path.iterdir():
                                self.files.append((patch_path, x_name, y))
        random.shuffle(self.files)

        self.color_transforms = myTransforms.Compose([
            myTransforms.ColorJitter(
                brightness = (0.65, 1.35), 
                contrast = (0.5, 1.5),
                # saturation=(0, 2), 
                # hue=0.3,
                ),
            # myTransforms.RandomChoice([myTransforms.ColorJitter(saturation=(0, 2), hue=0.3),
            #                             myTransforms.HEDJitter(theta=0.05)]),
            myTransforms.HEDJitter(theta=0.005),
            
        ])
        # self.color_transforms = myTransforms.Compose([
        #     myTransforms.Grayscale(num_output_channels=3)
        # ])
        self.train_transforms = myTransforms.Compose([
            myTransforms.RandomChoice([myTransforms.RandomHorizontalFlip(p=0.5),
                                        myTransforms.RandomVerticalFlip(p=0.5),
                                        myTransforms.AutoRandomRotation()]),
        
            myTransforms.RandomGaussBlur(radius=[0.5, 1.5]),
            myTransforms.RandomAffineCV2(alpha=0.1),
            myTransforms.RandomElastic(alpha=2, sigma=0.06),
        ])

        self.resize_transforms = transforms.Resize((self.size,self.size), transforms.InterpolationMode.BICUBIC)

        self.val_transforms = transforms.Compose([
            # 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            # RangeNormalization(),
        ])

        for f_info in tqdm(self.files):
            self._add_data_infos(f_info, self.cache)

    def _add_data_infos(self, file_info, load_data):
        patch_path, wsi_name, label = file_info

        label = self.slideLabelDict[wsi_name]
        tile_name = Path(patch_path).stem
        patient = self.slide_patient_dict[wsi_name]
        idx = -1
        self.data_info.append({'data_path': patch_path, 'label': label, 'wsi_name': wsi_name, 'tile_name': tile_name, 'patient': patient, 'cache_idx': idx})

    def _add_to_cache(self, data, data_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if data_path not in self.data_cache:
            self.data_cache[data_path] = [data]
        else:
            self.data_cache[data_path].append(data)
        return len(self.data_cache[data_path]) - 1

    def _load_data(self, file_path):

        img = Image.open(file_path)
        img = img.resize((self.size, self.size), Image.BICUBIC)
        tile_name = Path(file_path).stem
        # patient = self.slide_patient_dict[wsi_name]
        idx = self._add_to_cache(img, file_path)
        file_idx = next(i for i,v in enumerate(self.data_info) if v['data_path'] == file_path)
        self.data_info[file_idx + idx]['cache_idx'] = idx
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            # self.data_info = [{'data_path': di['data_path'], 'label': di['label'], 'shape': di['shape'], 'name': di['name'], 'cache_idx': -1} if di['data_path'] == removal_keys[0] else di for di in self.data_info]
            self.data_info = [{'data_path': di['data_path'], 'label': di['label'], 'wsi_name': di['wsi_name'], 'tile_name': di['tile_name'], 'patient': di['patient'], 'cache_idx': -1} if di['data_path'] == removal_keys[0] else di for di in self.data_info]


    def get_labels(self, indices):
        return [self.labels[i] for i in indices]
    
    def get_data(self, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
            i = index
        """
        # fp = self.get_data_infos(type)[i]['data_path']
        fp = self.data_info[i]['data_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        
        # get new cache_idx assigned by _load_data_info
        # cache_idx = self.get_data_infos(type)[i]['cache_idx']
        cache_idx = self.data_info[i]['cache_idx']
        label = self.data_info[i]['label']
        wsi_name = self.data_info[i]['wsi_name']
        tile_name = self.data_info[i]['tile_name']
        patient = self.data_info[i]['patient']
        # print(self.data_cache[fp][cache_idx])
        return self.data_cache[fp][cache_idx], label, (wsi_name, tile_name, patient)

    

    def __len__(self):
        # return len(self.files)
        return len(self.data_info)
    

    def __getitem__(self, index):


        img, label, (wsi_name, tile_name, patient) = self.get_data(index)
    
        if self.mode == 'train':
            
            # img = self.color_transforms(img)
            img = self.train_transforms(img)
            img = self.val_transforms(img.copy())
        else: 
            img = self.val_transforms(img)

        return img, label, (wsi_name, tile_name, patient)



if __name__ == '__main__':
    
    from pathlib import Path
    import os
    import time
    from fast_tensor_dl import FastTensorDataLoader
    from custom_resnet50 import resnet50_baseline
    from utils import myTransforms

    
    

    home = Path.cwd().parts[1]
    # train_csv = f'/{home}/ylan/DeepGraft_project/code/debug_train.csv'
    data_root = f'/raid/ylan/data/DeepGraft/224_256uM_annotated'
    # data_root = f'/{home}/ylan/DeepGraft/dataset/hdf5/256_256um_split/'
    # label_path = f'/{home}/ylan/DeepGraft_project/code/split_PAS_bin.json'
    # label_path = f'/{home}/ylan/DeepGraft/training_tables/split_debug.json'
    label_path = '/homeStor1/ylan/data/DeepGraft/training_tables/dg_split_PAS_HE_Jones_Grocott_norm_rest_ext.json'
    # output_dir = f'/{data_root}/debug/augments'
    # os.makedirs(output_dir, exist_ok=True)

    n_classes = 2

    dataset = LazyJPGBagLoader(data_root, label_path=label_path, mode='train', model='inception', n_classes=n_classes, cache=True, data_cache_size=5000)
    # dataset = JPGBagLoader(data_root, label_path=label_path, mode='train', n_classes=n_classes, cache=False)

    # print(dataset.get_labels(0))
    # a = int(len(dataset)* 0.8)
    # b = int(len(dataset) - a)
    # train_data, valid_data = random_split(dataset, [a, b])
    # print(dataset.dataset)
    # a = int(len(dataset)* 0.8)
    # b = int(len(dataset) - a)
    # train_ds, val_ds = torch.utils.data.random_split(dataset, [a, b])
    # dl = FastTensorDataLoader(dataset, batch_size=1, shuffle=False)
    dl = DataLoader(dataset, batch_size=2000, num_workers=4, pin_memory=True)
    # print(len(dl))
    # dl = DataLoader(dataset, batch_size=1, sampler=ImbalancedDatasetSampler(dataset), num_workers=5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()

    # model_ft = resnet50_baseline(pretrained=True)
    # for param in model_ft.parameters():
    #     param.requires_grad = False
    # model_ft.to(device)

    # model_ft = models.resnet50(weights='IMAGENET1K_V1')

    
    # ct = 0
    # for child in model_ft.children():
    #     ct += 1
    #     if ct < len(list(model_ft.children())) - 3:
    #         for parameter in child.parameters():
    #             parameter.requires_grad=False
    # model_ft.fc = nn.Linear(model_ft.fc.in_features, 2)

    # print(model_ft)

    # print(list(model_ft.children())[:7])
    #     for parameter in child.parameters():
    #         print(parameter.requires_grad)    
    
    c = 0
    # label_count = [0] *n_classes
    # # print(len(dl))
    start = time.time()
    for item in tqdm(dl): 

        if c >= 1000:
            break
        bag, label, (name, batch_names, patient) = item
        # print(bag.shape)
        print(name)
        # print(name)
        # print(batch_names)
        # print(patient)
        # print(len(batch_names))

        # bag = bag.squeeze(0).float().to(device)
        # label = label.to(device)
        # with torch.cuda.amp.autocast():
        #     output = model_ft(bag)
        c += 1
    end = time.time()

    print('Bag Time: ', end-start)