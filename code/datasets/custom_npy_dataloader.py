'''
ToDo: remove bag_size
'''


from custom_resnet50 import resnet50_baseline
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import cv2
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from imgaug import augmenters as iaa
import imgaug as ia
from torchsampler import ImbalancedDatasetSampler



class RangeNormalization(object):
    def __call__(self, sample):
        img = sample
        return (img / 255.0 - 0.5) / 0.5

class NPYMILDataloader(data.Dataset):
    
    def __init__(self, file_path, label_path, mode, n_classes, load_data=False, data_cache_size=500, max_bag_size=1000):
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
        self.min_bag_size = 120
        self.empty_slides = []
        self.corrupt_slides = []
        # self.label_file = label_path
        recursive = True
        exclude_cohorts = ["debug", "test"]
        # read labels and slide_path from csv
        with open(self.label_path, 'r') as f:
            temp_slide_label_dict = json.load(f)[mode]
            print(len(temp_slide_label_dict))
            for (x, y) in temp_slide_label_dict:
                x = Path(x).stem 
                # x_complete_path = Path(self.file_path)/Path(x)
                for cohort in Path(self.file_path).iterdir():
                    if cohort.stem not in exclude_cohorts:
                        x_complete_path = cohort / 'BLOCKS'
                        for slide_dir in Path(x_complete_path).iterdir():
                            if slide_dir.suffix == '.npy':
                                
                                if slide_dir.stem == x:
                            # if len(list(x_complete_path.iterdir())) > self.min_bag_size:
                            # print(x_complete_path)
                                    self.slideLabelDict[x] = y
                                    self.files.append(slide_dir)
                            # else: self.empty_slides.append(x_complete_path)
                    
        print(len(self.files))

        print(f'Slides with bag size under {self.min_bag_size}: ', len(self.empty_slides))
        # print(self.empty_slides)
        # print(len(self.files))
        # print(len(self.corrupt_slides))
        # print(self.corrupt_slides)
        home = Path.cwd().parts[1]
        slide_patient_dict_path = f'/{home}/ylan/DeepGraft/training_tables/slide_patient_dict.json'
        with open(slide_patient_dict_path, 'r') as f:
            slide_patient_dict = json.load(f)

        for slide_dir in tqdm(self.files):
            self._add_data_infos(str(slide_dir.resolve()), load_data, slide_patient_dict)


        self.resize_transforms = A.Compose([
            A.SmallestMaxSize(max_size=256)
        ])
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
        self.albu_transforms = A.Compose([
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, always_apply=False, p=0.5),
            A.ColorJitter(always_apply=False, p=0.5),
            A.RandomGamma(gamma_limit=(80,120)),
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            # A.OneOf([
            #     A.ElasticTransform(alpha=150, sigma=20, alpha_affine=50),
            #     A.Affine(
            #         scale={'x': (0.95, 1.05), 'y': (0.95, 1.05)},
            #         rotate=(-45, 45),
            #         shear=(-4, 4),
            #         cval=8,
            #         )
            # ]),
            A.Normalize(),
            ToTensorV2(),
        ])
        # self.train_transforms = A.Compose([
        #     A.HueSaturationValue(hue_shift_limit=13, sat_shift_limit=2, val_shift_limit=0, always_apply=True, p=1.0),
        #     # A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=0, val_shift_limit=0, always_apply=False, p=0.5),
        #     # A.RandomGamma(),
        #     # A.HorizontalFlip(),
        #     # A.VerticalFlip(),
        #     # A.RandomRotate90(),
        #     # A.OneOf([
        #     #     A.ElasticTransform(alpha=150, sigma=20, alpha_affine=50),
        #     #     A.Affine(
        #     #         scale={'x': (0.95, 1.05), 'y': (0.95, 1.05)},
        #     #         rotate=(-45, 45),
        #     #         shear=(-4, 4),
        #     #         cval=8,
        #     #         )
        #     # ]),
        #     A.Normalize(),
        #     ToTensorV2(),
        # ])
        self.val_transforms = transforms.Compose([
            # A.Normalize(),
            # ToTensorV2(),
            RangeNormalization(),
            transforms.ToTensor(),

        ])
        self.img_transforms = transforms.Compose([    
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            # histoTransforms.AutoRandomRotation(),
            transforms.Lambda(lambda a: np.array(a)),
        ]) 
        self.hsv_transforms = transforms.Compose([
            RandomHueSaturationValue(hue_shift_limit=(-13,13), sat_shift_limit=(-13,13), val_shift_limit=(-13,13)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        # get data
        (batch, batch_names), label, name, patient = self.get_data(index)
        out_batch = []
        seq_img_d = self.train_transforms.to_deterministic()
        
        if self.mode == 'train':
            # print(img)
            # print(.shape)
            for img in batch: # expects numpy 
                # img = img.numpy().astype(np.uint8)
                # img = self.albu_transforms(image=img)
                # print(img)
                # print(img.shape)
                # print(img)
                img = img.astype(np.uint8)

                img = seq_img_d.augment_image(img)
                img = self.val_transforms(img)
                out_batch.append(img)
                # img = self.albu_transforms(image=img)
                # out_batch.append(img['image'])

        else:
            for img in batch:
                img = img.numpy().astype(np.uint8)
                img = self.val_transforms(img)
                out_batch.append(img)

        # if len(out_batch) == 0:
        #     # print(name)
        #     out_batch = torch.randn(self.bag_size,3,256,256)
        # else: 
        # print(len(out_batch))
        out_batch = torch.stack(out_batch)
        # print(out_batch.shape)
        # out_batch = out_batch[torch.randperm(out_batch.shape[0])] #shuffle tiles within batch
        # print(out_batch.shape)
        # if out_batch.shape != torch.Size([self.bag_size, 256, 256, 3]) and out_batch.shape != torch.Size([self.bag_size, 3,256,256]):
        #     print(name)
        #     print(out_batch.shape)
        # out_batch = torch.permute(out_batch, (0, 2,1,3))
        label = torch.as_tensor(label)
        label = torch.nn.functional.one_hot(label, num_classes=self.n_classes)
        # print(out_batch)
        return out_batch, label, (name, batch_names, patient) #, name_batch

    def __len__(self):
        return len(self.data_info)
    
    def _add_data_infos(self, file_path, load_data, slide_patient_dict):

        
        wsi_name = Path(file_path).stem
        if wsi_name in self.slideLabelDict:
            # if wsi_name[:2] != 'RU': #skip RU because of container problems in dataset
            label = self.slideLabelDict[wsi_name]
            patient = slide_patient_dict[wsi_name]
            idx = -1
            self.data_info.append({'data_path': file_path, 'label': label, 'name': wsi_name, 'patient': patient,'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        wsi_batch = []
        name_batch = []
        # print(wsi_batch)
        # for tile_path in Path(file_path).iterdir():
        #     print(tile_path)
        wsi_batch = np.load(file_path)
        

        if wsi_batch.shape[0] > self.max_bag_size:
            wsi_batch, name_batch, _ = to_fixed_size_bag(wsi_batch, name_batch, self.max_bag_size)
        idx = self._add_to_cache((wsi_batch,name_batch), file_path)
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

    # def get_data_infos(self, type):
    #     """Get data infos belonging to a certain type of data.
    #     """
    #     data_info_type = [di for di in self.data_info if di['type'] == type]
    #     return data_info_type

    def get_name(self, i):
        # name = self.get_data_infos(type)[i]['name']
        name = self.data_info[i]['name']
        return name

    def get_labels(self, indices):

        return [self.data_info[i]['label'] for i in indices]
        # return self.slideLabelDict.values()

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
        name = self.data_info[i]['name']
        patient = self.data_info[i]['patient']
        # print(self.data_cache[fp][cache_idx])
        return self.data_cache[fp][cache_idx], label, name, patient



class RandomHueSaturationValue(object):

    def __init__(self, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255), val_shift_limit=(-255, 255), p=0.5):
        
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.p = p

    def __call__(self, sample):
    
        img = sample #,lbl
    
        if np.random.random() < self.p:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #takes np.float32
            h, s, v = cv2.split(img)
            hue_shift = np.random.randint(self.hue_shift_limit[0], self.hue_shift_limit[1] + 1)
            hue_shift = np.uint8(hue_shift)
            h += hue_shift
            sat_shift = np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
            v = cv2.add(v, val_shift)
            img = cv2.merge((h, s, v))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img #, lbl

def to_fixed_size_bag(bag, names, bag_size: int = 512):

    #duplicate bag instances unitl 

    # get up to bag_size elements
    rng = np.random.default_rng()
    # if bag.shape[0] > bag_size:
    rng.shuffle(bag)
    bag_samples = bag[:bag_size, :, :, :]
        # bag_samples = bag[bag_idxs]
    # else: 
    #     original_size = bag.shape[0]
    #     q, r = divmod(bag_size - bag.shape[0], bag.shape[0])
    #     # print(bag_size)
    #     # print(bag.shape[0])
    #     # print(q, r)

    #     if q > 0:
    #         bag = np.concatenate([bag for _ in range(q)])
    #     if r > 0: 
    #         temp = bag 
    #         rng.shuffle(temp)
    #         temp = temp[:r, :, :, :]
    #         bag = np.concatenate((bag, temp))
    #     bag_samples = bag
    name_batch = [''] * bag_size
    # name_samples = [names[i] for i in bag_idxs]
    # bag_sample_names = [bag_names[i] for i in bag_idxs]
    # q, r  = divmod(bag_size, bag_samples.shape[0])
    # if q > 0:
    #     bag_samples = torch.cat([bag_samples]*q, 0)
    
    # self_padded = torch.cat([bag_samples, bag_samples[:r,:, :, :]])

    # zero-pad if we don't have enough samples
    # zero_padded = torch.cat((bag_samples,
    #                         torch.zeros(bag_size-bag_samples.shape[0], bag_samples.shape[1], bag_samples.shape[2], bag_samples.shape[3])))

    return bag_samples, name_batch, min(bag_size, len(bag))


class RandomHueSaturationValue(object):

    def __init__(self, hue_shift_limit=(-180, 180), sat_shift_limit=(-255, 255), val_shift_limit=(-255, 255), p=0.5):
        
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.p = p

    def __call__(self, sample):
    
        img = sample #,lbl
    
        if np.random.random() < self.p:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #takes np.float32
            h, s, v = cv2.split(img)
            hue_shift = np.random.randint(self.hue_shift_limit[0], self.hue_shift_limit[1] + 1)
            hue_shift = np.uint8(hue_shift)
            h += hue_shift
            sat_shift = np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
            v = cv2.add(v, val_shift)
            img = cv2.merge((h, s, v))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img #, lbl



if __name__ == '__main__':
    from pathlib import Path
    import os
    import time
    from fast_tensor_dl import FastTensorDataLoader
    
    

    home = Path.cwd().parts[1]
    train_csv = f'/{home}/ylan/DeepGraft_project/code/debug_train.csv'
    data_root = f'/{home}/ylan/data/DeepGraft/224_128um_v2'
    # data_root = f'/{home}/ylan/DeepGraft/dataset/hdf5/256_256um_split/'
    # label_path = f'/{home}/ylan/DeepGraft_project/code/split_PAS_bin.json'
    label_path = f'/{home}/ylan/DeepGraft/training_tables/dg_limit_20_split_PAS_HE_Jones_norm_rest.json'
    output_dir = f'/{data_root}/debug/augments'
    os.makedirs(output_dir, exist_ok=True)

    n_classes = 2

    dataset = NPYMILDataloader(data_root, label_path=label_path, mode='train', load_data=False, n_classes=n_classes)
    a = int(len(dataset)* 0.8)
    b = int(len(dataset) - a)
    train_data, valid_data = random_split(dataset, [a, b])
    # print(dataset.dataset)
    # a = int(len(dataset)* 0.8)
    # b = int(len(dataset) - a)
    # train_ds, val_ds = torch.utils.data.random_split(dataset, [a, b])
    # dl = FastTensorDataLoader(dataset, batch_size=1, shuffle=False)
    dl = DataLoader(train_data, batch_size=1, num_workers=16, sampler=ImbalancedDatasetSampler(train_data), pin_memory=True)
    # print(len(dl))
    # dl = DataLoader(dataset, batch_size=1, sampler=ImbalancedDatasetSampler(dataset), num_workers=5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()

    model_ft = resnet50_baseline(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False
    model_ft.to(device)
    

    
    # data = DataLoader(dataset, batch_size=1)

    # print(len(dataset))
    # # x = 0
    #/home/ylan/DeepGraft/dataset/hdf5/256_256um_split/RU0248_PASD_jke_PASD_20200201_195900_BIG.hdf5
    c = 0
    label_count = [0] *n_classes
    # print(len(dl))
    start = time.time()
    for item in tqdm(dl): 

        
        # if c >= 10:
        #     break
        bag, label, (name, _, patient) = item
        bag = bag.squeeze(0).float().to(device)
        label = label.to(device)
        with torch.cuda.amp.autocast():
            output = model_ft(bag)
        c += 1
    end = time.time()

    print('Bag Time: ', end-start)
        # print(label)
        # print(name)
        # print(patient)
    #     label_count[torch.argmax(label)] += 1
        # print(name)
        # if name == 'RU0248_PASD_jke_PASD_20200201_195900_BIG':
        
            # print(bag)
            # print(label)
        
    #     # # print(bag.shape)
    #     # if bag.shape[1] == 1:
    #     #     print(name)
    #     #     print(bag.shape)
        # print(bag.shape)
        
    #     # out_dir = Path(output_path) / name
    #     # os.makedirs(out_dir, exist_ok=True)

    #     # # print(item[2])
    #     # # print(len(item))
    #     # # print(item[1])
    #     # # print(data.shape)
    #     # # data = data.squeeze()
    #     # bag = item[0]
    #     bag = bag.squeeze()
    #     original = original.squeeze()
        # output_path = Path(output_dir) / name
        # output_path.mkdir(exist_ok=True)
        # for i in range(bag.shape[0]):
        #     img = bag[i, :, :, :]
        #     img = img.squeeze()
            
        #     img = ((img-img.min())/(img.max() - img.min())) * 255
        #     # print(img)
        #     # print(img)
        #     img = img.numpy().astype(np.uint8).transpose(1,2,0)

            
        #     img = Image.fromarray(img)
        #     img = img.convert('RGB')
        #     img.save(f'{output_path}/{i}.png')

        # c += 1
            
    #         o_img = original[i,:,:,:]
    #         o_img = o_img.squeeze()
    #         print(o_img.shape)
    #         o_img = ((o_img-o_img.min())/(o_img.max()-o_img.min()))*255
    #         o_img = o_img.numpy().astype(np.uint8).transpose(1,2,0)
    #         o_img = Image.fromarray(o_img)
    #         o_img = o_img.convert('RGB')
    #         o_img.save(f'{output_path}/{i}_original.png')
        
    #     break
        # else: break
        # print(data.shape)
        # print(label)
    # a = [torch.Tensor((3,256,256))]*3
    # b = torch.stack(a)
    # print(b)
    # c = to_fixed_size_bag(b, 512)
    # print(c)