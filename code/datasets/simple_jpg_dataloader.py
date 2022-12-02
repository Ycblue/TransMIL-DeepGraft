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


class JPGBagLoader(data_utils.Dataset):
    def __init__(self, file_path, label_path, mode, n_classes, load_data=False, data_cache_size=100, max_bag_size=1000, cache=False):
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
        self.cache = True
        
        # read labels and slide_path from csv
        with open(self.label_path, 'r') as f:
            temp_slide_label_dict = json.load(f)[mode]
            print(len(temp_slide_label_dict))
            for (x, y) in temp_slide_label_dict:
                x = Path(x).stem 
                # x_complete_path = Path(self.file_path)/Path(x)
                for cohort in Path(self.file_path).iterdir():
                    x_complete_path = Path(self.file_path) / cohort / 'BLOCKS' / Path(x)
                    if x_complete_path.is_dir():
                        if len(list(x_complete_path.iterdir())) > self.min_bag_size:
                        # print(x_complete_path)
                            self.slideLabelDict[x] = y
                            self.files.append(x_complete_path)
                        else: self.empty_slides.append(x_complete_path)
        
        home = Path.cwd().parts[1]
        self.slide_patient_dict_path = f'/{home}/ylan/DeepGraft/training_tables/slide_patient_dict.json'
        with open(self.slide_patient_dict_path, 'r') as f:
            self.slide_patient_dict = json.load(f)

        sometimes = lambda aug: iaa.Sometimes(0.5, aug, name="Random1")
        sometimes2 = lambda aug: iaa.Sometimes(0.2, aug, name="Random2")
        sometimes3 = lambda aug: iaa.Sometimes(0.9, aug, name="Random3")
        sometimes4 = lambda aug: iaa.Sometimes(0.9, aug, name="Random4")
        sometimes5 = lambda aug: iaa.Sometimes(0.9, aug, name="Random5")

        self.train_transforms = iaa.Sequential([
            iaa.AddToHueAndSaturation(value=(-30, 30), name="MyHSV"), #13
            sometimes2(iaa.GammaContrast(gamma=(0.85, 1.15), name="MyGamma")),
            iaa.Fliplr(0.5, name="MyFlipLR"),
            iaa.Flipud(0.5, name="MyFlipUD"),
            sometimes(iaa.Rot90(k=1, keep_size=True, name="MyRot90")),
            # iaa.OneOf([
            #     sometimes3(iaa.PiecewiseAffine(scale=(0.015, 0.02), cval=0, name="MyPiece")),
            #     sometimes4(iaa.ElasticTransformation(alpha=(100, 200), sigma=20, cval=0, name="MyElastic")),
            #     sometimes5(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, rotate=(-45, 45), shear=(-4, 4), cval=0, name="MyAffine"))
            # ], name="MyOneOf")

        ], name="MyAug")
        self.val_transforms = transforms.Compose([
            # 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            # RangeNormalization(),
        ])




        self.features = []
        self.labels = []
        self.wsi_names = []
        self.name_batches = []
        self.patients = []
        if self.cache:
            if mode=='train':
                seq_img_d = self.train_transforms.to_deterministic()
                
                # with tqdm(total=len(self.files)) as pbar:

                for t in tqdm(self.files):
                    batch, label, (wsi_name, name_batch, patient) = self.get_data(t)
                    # print('label: ', label)
                    out_batch = []
                    for img in batch: 
                        img = img.numpy().astype(np.uint8)
                        img = seq_img_d.augment_image(img)
                        img = self.val_transforms(img.copy())
                        out_batch.append(img)
                    # ft = ft.view(-1, 512)
                    
                    out_batch = torch.stack(out_batch)
                    self.labels.append(label)
                    self.features.append(out_batch)
                    self.wsi_names.append(wsi_name)
                    self.name_batches.append(name_batch)
                    self.patients.append(patient)
                        # pbar.update()
            else: 
                # with tqdm(total=len(self.file_path)) as pbar:
                for t in tqdm(self.file_path):
                    batch, label, (wsi_name, name_batch, patient) = self.get_data(t)
                    out_batch = []
                    for img in batch: 
                        img = img.numpy().astype(np.uint8)
                        img = self.val_transforms(img.copy())
                        out_batch.append(img)
                    # ft = ft.view(-1, 512)
                    out_batch = torch.stack(out_batch)
                    self.labels.append(label)
                    self.features.append(out_batch)
                    self.wsi_names.append(wsi_name)
                    self.name_batches.append(name_batch)
                    self.patients.append(patient)
                        # pbar.update()
        # print(self.get_bag_feats(self.train_path))
        # self.r = np.random.RandomState(seed)

        # self.num_in_train = 60000
        # self.num_in_test = 10000

        # if self.train:
        #     self.train_bags_list, self.train_labels_list = self._create_bags()
        # else:
        #     self.test_bags_list, self.test_labels_list = self._create_bags()

    def get_data(self, file_path):
        
        wsi_batch=[]
        name_batch=[]
        
        for tile_path in Path(file_path).iterdir():
            img = np.asarray(Image.open(tile_path)).astype(np.uint8)
            img = torch.from_numpy(img)
            wsi_batch.append(img)
            name_batch.append(tile_path.stem)

        wsi_batch = torch.stack(wsi_batch)

        if wsi_batch.size(0) > self.max_bag_size:
            wsi_batch, name_batch, _ = self.to_fixed_size_bag(wsi_batch, name_batch, self.max_bag_size)


        wsi_batch, name_batch = self.data_dropout(wsi_batch, name_batch, drop_rate=0.1)

        wsi_name = Path(file_path).stem
        try:
            label = self.slideLabelDict[wsi_name]
        except KeyError:
            print(f'{wsi_name} is not included in label file {self.label_path}')

        try:
            patient = self.slide_patient_dict[wsi_name]
        except KeyError:
            print(f'{wsi_name} is not included in label file {self.slide_patient_dict_path}')

        return wsi_batch, label, (wsi_name, name_batch, patient)
    
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
        bag_size = bag.shape[0]
        bag_idxs = torch.randperm(bag_size)[:int(bag_size*(1-drop_rate))]
        bag_samples = bag[bag_idxs]
        name_samples = [batch_names[i] for i in bag_idxs]

        return bag_samples, name_samples

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        if self.cache:
            label = self.labels[index]
            wsi = self.features[index]
            label = int(label)
            wsi_name = self.wsi_names[index]
            name_batch = self.name_batches[index]
            patient = self.patients[index]
            # feats = Variable(Tensor(feats))
            return wsi, label, (wsi_name, name_batch, patient)
        else:
            if self.mode=='train':
                batch, label, (wsi_name, name_batch, patient) = self.get_data(self.files[index])
                # label = Variable(Tensor(label))

                # wsi = Variable(Tensor(wsi_batch))
                out_batch = []
                seq_img_d = self.train_transforms.to_deterministic()
                for img in batch: 
                    img = img.numpy().astype(np.uint8)
                    # img = seq_img_d.augment_image(img)
                    img = self.val_transforms(img.copy())
                    out_batch.append(img)
                out_batch = torch.stack(out_batch)
                # ft = ft.view(-1, 512)
                
            else:
                batch, label, (wsi_name, name_batch, patient) = self.get_data(self.files[index])
                label = Variable(Tensor(label))
                out_batch = []
                seq_img_d = self.train_transforms.to_deterministic()
                for img in batch: 
                    img = img.numpy().astype(np.uint8)
                    img = self.val_transforms(img.copy())
                    out_batch.append(img)
                out_batch = torch.stack(out_batch)

            return out_batch, label, (wsi_name, name_batch, patient)

if __name__ == '__main__':
    
    from pathlib import Path
    import os
    import time
    from fast_tensor_dl import FastTensorDataLoader
    from custom_resnet50 import resnet50_baseline
    
    

    home = Path.cwd().parts[1]
    train_csv = f'/{home}/ylan/DeepGraft_project/code/debug_train.csv'
    data_root = f'/{home}/ylan/data/DeepGraft/224_128um_v2'
    # data_root = f'/{home}/ylan/DeepGraft/dataset/hdf5/256_256um_split/'
    # label_path = f'/{home}/ylan/DeepGraft_project/code/split_PAS_bin.json'
    label_path = f'/{home}/ylan/DeepGraft/training_tables/split_debug.json'
    # label_path = f'/{home}/ylan/DeepGraft/training_tables/dg_limit_20_split_PAS_HE_Jones_norm_rest.json'
    output_dir = f'/{data_root}/debug/augments'
    os.makedirs(output_dir, exist_ok=True)

    n_classes = 2

    dataset = JPGBagLoader(data_root, label_path=label_path, mode='train', load_data=False, n_classes=n_classes)

    # print(dataset.get_labels(0))
    a = int(len(dataset)* 0.8)
    b = int(len(dataset) - a)
    train_data, valid_data = random_split(dataset, [a, b])
    # print(dataset.dataset)
    # a = int(len(dataset)* 0.8)
    # b = int(len(dataset) - a)
    # train_ds, val_ds = torch.utils.data.random_split(dataset, [a, b])
    # dl = FastTensorDataLoader(dataset, batch_size=1, shuffle=False)
    dl = DataLoader(train_data, batch_size=1, num_workers=8, sampler=ImbalancedDatasetSampler(train_data), pin_memory=True)
    # print(len(dl))
    # dl = DataLoader(dataset, batch_size=1, sampler=ImbalancedDatasetSampler(dataset), num_workers=5)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()

    model_ft = resnet50_baseline(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False
    model_ft.to(device)
    
    c = 0
    label_count = [0] *n_classes
    # print(len(dl))
    start = time.time()
    for item in tqdm(dl): 

        # if c >= 10:
        #     break
        bag, label, (name, batch_names, patient) = item
        # print(bag.shape)
        # print(len(batch_names))
        print(label)
        bag = bag.squeeze(0).float().to(device)
        label = label.to(device)
        with torch.cuda.amp.autocast():
            output = model_ft(bag)
        c += 1
    end = time.time()

    print('Bag Time: ', end-start)