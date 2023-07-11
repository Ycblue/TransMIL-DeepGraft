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
import re
from monai.networks.nets import milmodel
import torch.nn as nn

class JPGMILDataloader(data_utils.Dataset):
    def __init__(self, file_path, label_path, mode, n_classes, model=None, data_cache_size=100, max_bag_size=1000, cache=False, mixup=False, aug=False, patients=None, slides=None):
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
        self.slides_to_process = []

        # self.features = []
        # self.labels = []
        # self.wsi_names = []
        # self.name_batches = []
        # self.patients = []
        home = Path.cwd().parts[1]

        self.slide_patient_dict_path = f'/{home}/ylan/data/DeepGraft/training_tables/slide_patient_dict_an_ext.json'
        # self.slide_patient_dict_path = Path(self.label_path).parent / 'slide_patient_dict_an.json'
        with open(self.slide_patient_dict_path, 'r') as f:
            self.slide_patient_dict = json.load(f)
        # if patients: 
        #     self.slides_to_process = [self.slide_patient_dict[p] for p in patients]
        # elif slides: 
        #     self.slides_to_process = slides
        # else 

        
        # read labels and slide_path from csv
        with open(self.label_path, 'r') as f:
            json_dict = json.load(f)
            temp_slide_label_dict = json_dict[self.mode]

            # print(len(temp_slide_label_dict))

            for (x,y) in temp_slide_label_dict:
                
                if self.mode == 'test':
                    x = x.replace('FEATURES_RETCCL_2048', 'TEST')
                    
                else:
                    x = x.replace('FEATURES_RETCCL_2048', 'BLOCKS')
                # print(x)
                x_name = Path(x).stem
                if x_name in self.slide_patient_dict.keys():
                    if patients:
                        if self.slide_patient_dict[x_name] in patients:
                            x_path_list = [Path(self.file_path)/x]
                            for x_path in x_path_list:
                                if x_path.exists():
                                    # print(len(list(x_path.glob('*'))))
                                    self.slideLabelDict[x_name] = y
                                    self.labels += [int(y)]*len(list(x_path.glob('*')))
                                    # self.labels.append(int(y))
                                    self.files.append(x_path)
                            
                    elif slides: 
                        if x_name in slides:
                            x_path_list = [Path(self.file_path)/x]
                            for x_path in x_path_list:
                                if x_path.exists():
                                    self.slideLabelDict[x_name] = y
                                    self.labels += [int(y)]*len(list(x_path.glob('*')))
                                    self.files.append(x_path)
                    else:
                        x_path_list = [Path(self.file_path)/x]
                        for x_path in x_path_list:
                            if x_path.exists():
                                # print(len(list(x_path.glob('*'))))
                                self.slideLabelDict[x_name] = y
                                self.labels += [int(y)]*len(list(x_path.glob('*')))
                                # self.labels.append(int(y))
                                self.files.append(x_path)
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
        
        

        # def get_transforms_2():
        
        self.color_transforms = myTransforms.Compose([
            myTransforms.ColorJitter(
                brightness = (0.65, 1.35), 
                contrast = (0.5, 1.5),
                ),
            myTransforms.HEDJitter(theta=0.005),
            
        ])
        self.train_transforms = myTransforms.Compose([
            myTransforms.RandomChoice([myTransforms.RandomHorizontalFlip(p=0.5),
                                        myTransforms.RandomVerticalFlip(p=0.5),
                                        myTransforms.AutoRandomRotation()]),
        
            myTransforms.RandomGaussBlur(radius=[0.5, 1.5]),
            myTransforms.RandomAffineCV2(alpha=0.1),
            myTransforms.RandomElastic(alpha=2, sigma=0.06),
        ])

        self.val_transforms = transforms.Compose([
            # 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])




        
        # if self.cache:
        #     if mode=='train':
        #         seq_img_d = self.train_transforms.to_deterministic()
                
        #         # with tqdm(total=len(self.files)) as pbar:

        #         for t in tqdm(self.files):
        #             batch, label, (wsi_name, name_batch, patient) = self.get_data(t)
        #             # print('label: ', label)
        #             out_batch = []
        #             for img in batch: 
        #                 img = img.numpy().astype(np.uint8)
        #                 img = seq_img_d.augment_image(img)
        #                 img = self.val_transforms(img.copy())
        #                 out_batch.append(img)
        #             # ft = ft.view(-1, 512)
                    
        #             out_batch = torch.stack(out_batch)
        #             self.labels.append(label)
        #             self.features.append(out_batch)
        #             self.wsi_names.append(wsi_name)
        #             self.name_batches.append(name_batch)
        #             self.patients.append(patient)
        #                 # pbar.update()
        #     else: 
        #         # with tqdm(total=len(self.file_path)) as pbar:
        #         for t in tqdm(self.file_path):
        #             batch, label, (wsi_name, name_batch, patient) = self.get_data(t)
        #             out_batch = []
        #             for img in batch: 
        #                 img = img.numpy().astype(np.uint8)
        #                 img = self.val_transforms(img.copy())
        #                 out_batch.append(img)
        #             # ft = ft.view(-1, 512)
        #             out_batch = torch.stack(out_batch)
        #             self.labels.append(label)
        #             self.features.append(out_batch)
        #             self.wsi_names.append(wsi_name)
        #             self.name_batches.append(name_batch)
        #             self.patients.append(patient)
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
        coords_batch=[]
        
        for tile_path in Path(file_path).iterdir():
            img = Image.open(tile_path)
            # if self.mode == 'train':
        
            #     # img = self.color_transforms(img)
            #     img = self.train_transforms(img)
            img = self.val_transforms(img)
            # img = np.asarray(Image.open(tile_path)).astype(np.uint8)
            # img = np.moveaxis(img, 2, 0)
            # print(img.shape)
            # img = torch.from_numpy(img)
            wsi_batch.append(img)
            pos = re.findall(r'\((.*?)\)', tile_path.stem)
            x, y = pos[-1].split('-')
            coords = [int(x),int(y)]

            # name_batch.append(tile_path.stem)

            coords_batch.append(coords)
        
        coords_batch = np.stack(coords_batch)
        # print(coords_batch.shape)
        coords_batch = torch.from_numpy(coords_batch)
        wsi_batch = torch.stack(wsi_batch)

        # if wsi_batch.size(0) > self.max_bag_size:
        

        wsi_name = Path(file_path).stem
        # try:
        # label = self.slideLabelDict[wsi_name]
        # except KeyError:
        #     print(f'{wsi_name} is not included in label file {self.label_path}')

        

        # try:
        patient = self.slide_patient_dict[wsi_name]
        # except KeyError:
        #     print(f'{wsi_name} is not included in label file {self.slide_patient_dict_path}')
        

        return wsi_batch, (wsi_name, coords_batch, patient)
    
    def get_labels(self, indices):
        return [self.labels[i] for i in indices]


    def to_fixed_size_bag(self, bag, names, bag_size: int = 250):

        #duplicate bag instances unitl 

        bag_idxs = torch.randperm(bag.shape[0])[:bag_size]
        bag_samples = bag[bag_idxs]
        name_samples = [names[i] for i in bag_idxs]
        zero_padded = torch.cat((bag_samples,
                                torch.zeros(bag_size-bag_samples.shape[0], bag_samples.shape[1], bag_samples.shape[2], bag_samples.shape[3])))
        return zero_padded, name_samples, min(bag_size, len(bag))

    def data_dropout(self, bag, drop_rate):
        bag_size = bag.shape[0]
        bag_idxs = torch.randperm(bag_size)[:int(bag_size*(1-drop_rate))]
        bag_samples = bag[bag_idxs]
        # name_samples = [batch_names[i] for i in bag_idxs]

        return bag_samples

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        # if self.cache:
        #     label = self.labels[index]
        #     wsi = self.features[index]
        #     label = int(label)
        #     wsi_name = self.wsi_names[index]
        #     name_batch = self.name_batches[index]
        #     patient = self.patients[index]
        #     return wsi, label, (wsi_name, name_batch, patient)
        # else:
        t = self.files[index]
        # label = self.labels[index]
        if self.mode=='train' or self.mode=='val':

            batch, (wsi_name, batch_coords, patient) = self.get_data(t)
            label = self.labels[index]
            out_batch, batch_coords, _ = self.to_fixed_size_bag(batch, batch_coords, self.max_bag_size)

            # print('_getitem_: ', out_batch.shape)
            # bag_idxs = 
            # batch = self.data_dropout(batch, drop_rate=0.1)
            # print(batch.shape)
            # # label = Variable(Tensor(label))

            # # wsi = Variable(Tensor(wsi_batch))
            # out_batch = []

            # # seq_img_d = self.train_transforms.to_deterministic()
            # # seq_img_resize = self.resize_transforms.to_deterministic()
            # for img in batch: 
            #     # img = img.numpy().astype(np.uint8)
            #     # print(img.shape)
            #     img = self.resize_transforms(img)
            #     # print(img)
            #     # print(img.shape)
            #     # img = torch.moveaxis(img, 0, 2) # with HEDJitter wants [W,H,3], ColorJitter wants [3,W,H]
            #     # print(img.shape)
            #     img = self.color_transforms(img)
            #     print(img.shape)
            #     img = self.train_transforms(img)
                
            #     # img = seq_img_d.augment_image(img)
            #     img = self.val_transforms(img.copy())
            #     out_batch.append(img)
            # out_batch = torch.stack(out_batch)
            
            # ft = ft.view(-1, 512)
        else:
            batch, (wsi_name, batch_coords, patient) = self.get_data(t)
            label = self.labels[index]
            out_batch = batch

        # return out_batch, label, (wsi_name, patient)
        return out_batch, label, (wsi_name, batch_coords, patient)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        out = input.view(*self.shape)
        return out


if __name__ == '__main__':
    
    from pathlib import Path
    import os
    import time
    from fast_tensor_dl import FastTensorDataLoader
    from custom_resnet50 import resnet50_baseline
    
    

    home = Path.cwd().parts[1]
    # train_csv = f'/{home}/ylan/DeepGraft_project/code/debug_train.csv'
    data_root = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated'
    # data_root = f'/{home}/ylan/DeepGraft/dataset/hdf5/256_256um_split/'
    # label_path = f'/{home}/ylan/DeepGraft_project/code/split_PAS_bin.json'
    # label_path = f'/{home}/ylan/DeepGraft/training_tables/split_debug.json'
    label_path = f'/{home}/ylan/data/DeepGraft/training_tables/dg_split_PAS_HE_Jones_norm_rest_val_1.json'
    # output_dir = f'/{data_root}/debug/augments'
    # os.makedirs(output_dir, exist_ok=True)

    n_classes = 2

    dataset = JPGMILDataloader(data_root, label_path=label_path, mode='train', n_classes=n_classes, cache=False)
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
    dl = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)
    # print(len(dl))
    # dl = DataLoader(dataset, batch_size=1, sampler=ImbalancedDatasetSampler(dataset), num_workers=5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()

    # model_ft = resnet50_baseline(pretrained=True)
    # for param in model_ft.parameters():
    #     param.requires_grad = False
    # model_ft.to(device)
    model_ft = nn.Sequential(
        nn.Conv2d(3, 20, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(20, 50, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        View((-1, 53*53*50)),
        nn.Linear(53*53*50, 1024),
        nn.ReLU(),
    )
    model_ft.to(device)

    # model = milmodel.MILModel(num_classes=2, pretrained=True, mil_mode='att_trans').to(device)
    
    # c = 0
    # label_count = [0] *n_classes
    # # print(len(dl))
    start = time.time()
    for item in tqdm(dl): 

        # if c >= 10:
        #     break
        
        bag, label, (name, batch_names, patient) = item
        bag = bag.to(device)
        # print(bag.shape)
        # print(name)
        # print(batch_names)
        # print(patient)
        # print(len(batch_names))

        # print(label.shape)
        # bag = bag.squeeze(0).float().to(device)
        # label = label.to(device)
        with torch.cuda.amp.autocast():
            output = model_ft(bag.squeeze())
            # output = model_ft(bag)
        print(output.shape)
        # c += 1
    end = time.time()

    print('Bag Time: ', end-start)