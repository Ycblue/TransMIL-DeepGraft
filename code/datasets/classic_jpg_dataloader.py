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


class JPGBagLoader(data_utils.Dataset):
    def __init__(self, file_path, label_path, mode, n_classes, data_cache_size=100, max_bag_size=1000, cache=False, mixup=False, aug=False, model='inception'):
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
            size = 299
        elif model == 'vit':
            size = 384
        else: size = 224

        
        # read labels and slide_path from csv
        with open(self.label_path, 'r') as f:
            json_dict = json.load(f)
            temp_slide_label_dict = json_dict[self.mode]
            # print(len(temp_slide_label_dict))
            for (x,y) in temp_slide_label_dict:
                x = x.replace('FEATURES_RETCCL_2048', 'BLOCKS')
                # print(x)
                x_name = Path(x).stem
                x_path_list = [Path(self.file_path)/x]
                for x_path in x_path_list:
                    if x_path.exists():
                        # print(len(list(x_path.glob('*'))))

                        self.slideLabelDict[x_name] = y
                        self.labels += [int(y)]*len(list(x_path.glob('*')))
                        # self.labels.append(int(y))
                        for patch in x_path.iterdir():
                            self.files.append((patch, x_name, y))

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
        
        home = Path.cwd().parts[1]
        self.slide_patient_dict_path = Path(self.label_path).parent / 'slide_patient_dict_an.json'
        # self.slide_patient_dict_path = f'/{home}/ylan/data/DeepGraft/training_tables/slide_patient_dict_an.json'
        with open(self.slide_patient_dict_path, 'r') as f:
            self.slide_patient_dict = json.load(f)


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
        self.color_transforms = myTransforms.Compose([
            myTransforms.Grayscale(num_output_channels=3)
        ])
        self.train_transforms = myTransforms.Compose([
            myTransforms.RandomChoice([myTransforms.RandomHorizontalFlip(p=0.5),
                                        myTransforms.RandomVerticalFlip(p=0.5),
                                        myTransforms.AutoRandomRotation()]),
        
            myTransforms.RandomGaussBlur(radius=[0.5, 1.5]),
            myTransforms.RandomAffineCV2(alpha=0.1),
            myTransforms.RandomElastic(alpha=2, sigma=0.06),
        ])

        self.resize_transforms = transforms.Resize((299,299), transforms.InterpolationMode.BICUBIC)

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
        # img = np.moveaxis(img, 2, 0)
        # print(img.shape)
        # img = torch.from_numpy(img)
        tile_name = Path(patch_path).stem
        # patient = tile_name.rsplit('_', 1)[0]
        patient = self.slide_patient_dict[wsi_name]

        # for tile_path in Path(file_path).iterdir():
        #     img = np.asarray(Image.open(tile_path)).astype(np.uint8)
        #     img = np.moveaxis(img, 2, 0)
        #     # print(img.shape)
        #     img = torch.from_numpy(img)
        #     wsi_batch.append(img)
        #     name_batch.append(tile_path.stem)

        # wsi_batch = torch.stack(wsi_batch)
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
                save_img(img, f'{tile_name}_original')
                img = self.resize_transforms(img)
                img = self.color_transforms(img)
                img = self.train_transforms(img)

                # save_img(img, f'{tile_name}')

                img = self.val_transforms(img.copy())

                
                # ft = ft.view(-1, 512)
                
            else:
                img, label, (wsi_name, tile_name, patient) = self.get_data(t)
                # label = Variable(Tensor(label))
                # seq_img_d = self.train_transforms.to_deterministic()
                # seq_img_resize = self.resize_transforms.to_deterministic()
                # img = img.numpy().astype(np.uint8)
                img = self.resize_transforms(img)
                # img = np.moveaxis(img, 0, 2)
                img = self.val_transforms(img)

            return img, label, (wsi_name, tile_name, patient)

def save_img(img, comment):
    home = Path.cwd().parts[1]
    outputPath = f'/{home}/ylan/data/DeepGraft/224_128uM_annotated/debug/augments_2'
    img = img.convert('RGB')
    img.save(f'{outputPath}/{comment}.jpg')

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
    label_path = f'/{home}/ylan/data/DeepGraft/training_tables/dg_limit_5_split_PAS_HE_Jones_norm_rest_test.json'
    # output_dir = f'/{data_root}/debug/augments'
    # os.makedirs(output_dir, exist_ok=True)

    n_classes = 2

    dataset = JPGBagLoader(data_root, label_path=label_path, mode='train', n_classes=n_classes, cache=False)
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
    dl = DataLoader(dataset, batch_size=5, num_workers=8, pin_memory=True)
    # print(len(dl))
    # dl = DataLoader(dataset, batch_size=1, sampler=ImbalancedDatasetSampler(dataset), num_workers=5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

        if c >= 1000:
            break
        bag, label, (name, batch_names, patient) = item
        print(bag.shape)
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