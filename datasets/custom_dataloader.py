import h5py
# import helpers
import numpy as np
from pathlib import Path
import torch
# from torch._C import long
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
# from histoTransforms import RandomHueSaturationValue
import torchvision.transforms as transforms
import torch.nn.functional as F
import csv
from PIL import Image
import cv2
import pandas as pd
import json

class HDF5MILDataloader(data.Dataset):
    """Represents an abstract HDF5 dataset. For single H5 container! 
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        mode: 'train' or 'test'
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).

    """
    def __init__(self, file_path, label_path, mode, n_classes, load_data=False, data_cache_size=20):
        super().__init__()

        self.data_info = []
        self.data_cache = {}
        self.slideLabelDict = {}
        self.data_cache_size = data_cache_size
        self.mode = mode
        self.file_path = file_path
        # self.csv_path = csv_path
        self.label_path = label_path
        self.n_classes = n_classes
        self.bag_size = 120
        # self.label_file = label_path
        recursive = True

        # read labels and slide_path from csv

        # df = pd.read_csv(self.csv_path)
        # labels = df.LABEL
        # slides = df.FILENAME
        with open(self.label_path, 'r') as f:
            self.slideLabelDict = json.load(f)[mode]

        self.slideLabelDict = {Path(x).stem : y for (x,y) in self.slideLabelDict}

            
        # if Path(slides[0]).suffix:
        #     slides = list(map(lambda x: Path(x).stem, slides))

        # print(labels)
        # print(slides)
        # self.slideLabelDict = dict(zip(slides, labels))
        # print(self.slideLabelDict)

        #check if files in slideLabelDict, only take files that are available.

        files_in_path = list(Path(self.file_path).rglob('*.hdf5'))
        files_in_path = [x.stem for x in files_in_path]
        # print(len(files_in_path))
        # print(files_in_path)
        # print(list(self.slideLabelDict.keys()))
        # for x in list(self.slideLabelDict.keys()):
        #     if x in files_in_path:
        #         path = Path(self.file_path) / (x + '.hdf5')
        #         print(path)

        self.files = [Path(self.file_path)/ (x + '.hdf5') for x in list(self.slideLabelDict.keys()) if x in files_in_path]

        print(len(self.files))
        # self.files = list(map(lambda x: Path(self.file_path) / (Path(x).stem + '.hdf5'), list(self.slideLabelDict.keys())))

        for h5dataset_fp in tqdm(self.files):
            # print(h5dataset_fp)
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)

        # print(self.data_info)
        self.resize_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
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

        # self._add_data_infos(load_data)


    def __getitem__(self, index):
        # get data
        batch, label, name = self.get_data(index)
        out_batch = []
        
        if self.mode == 'train':
            # print(img)
            # print(img.shape)
            for img in batch: 
                img = self.img_transforms(img)
                img = self.hsv_transforms(img)
                out_batch.append(img)

        else:
            for img in batch:
                img = transforms.functional.to_tensor(img)
                out_batch.append(img)
        if len(out_batch) == 0:
            # print(name)
            out_batch = torch.randn(100,3,256,256)
        else: out_batch = torch.stack(out_batch)
        out_batch = out_batch[torch.randperm(out_batch.shape[0])] #shuffle tiles within batch

        label = torch.as_tensor(label)
        label = torch.nn.functional.one_hot(label, num_classes=self.n_classes)
        return out_batch, label, name

    def __len__(self):
        return len(self.data_info)
    
    def _add_data_infos(self, file_path, load_data):
        wsi_name = Path(file_path).stem
        if wsi_name in self.slideLabelDict:
            label = self.slideLabelDict[wsi_name]
            wsi_batch = []
            # with h5py.File(file_path, 'r') as h5_file:
            #     numKeys = len(h5_file.keys())
            #     sample = list(h5_file.keys())[0]
            #     shape = (numKeys,) + h5_file[sample][:].shape
                # for tile in h5_file.keys():
                #     img = h5_file[tile][:]
                    
                    # print(img)
                    # if type == 'images':
                    #     t = 'data'
                    # else: 
                    #     t = 'label'
            idx = -1
            # if load_data: 
            #     for tile in h5_file.keys():
            #         img = h5_file[tile][:]
            #         img = img.astype(np.uint8)
            #         img = self.resize_transforms(img)
            #         wsi_batch.append(img)
            #     idx = self._add_to_cache(wsi_batch, file_path)
                #     wsi_batch.append(img)
            # self.data_info.append({'data_path': file_path, 'label': label, 'shape': shape, 'name': wsi_name, 'cache_idx': idx})
            self.data_info.append({'data_path': file_path, 'label': label, 'name': wsi_name, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path, 'r') as h5_file:
            wsi_batch = []
            for tile in h5_file.keys():
                img = h5_file[tile][:]
                img = img.astype(np.uint8)
                img = self.resize_transforms(img)
                wsi_batch.append(img)
            idx = self._add_to_cache(wsi_batch, file_path)
            file_idx = next(i for i,v in enumerate(self.data_info) if v['data_path'] == file_path)
            self.data_info[file_idx + idx]['cache_idx'] = idx

            # for type in ['images', 'labels']:
            #     for key in tqdm(h5_file[f'{self.mode}/{type}'].keys()):
            #         img = h5_file[data_path][:]
            #         idx = self._add_to_cache(img, data_path)
            #         file_idx = next(i for i,v in enumerate(self.data_info) if v['data_path'] == data_path)
            #         self.data_info[file_idx + idx]['cache_idx'] = idx
            # for gname, group in h5_file.items():
            #     for dname, ds in group.items():
            #         # add data to the data cache and retrieve
            #         # the cache index
            #         idx = self._add_to_cache(ds.value, file_path)

            #         # find the beginning index of the hdf5 file we are looking for
            #         file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

            #         # the data info should have the same index since we loaded it in the same way
            #         self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            # self.data_info = [{'data_path': di['data_path'], 'label': di['label'], 'shape': di['shape'], 'name': di['name'], 'cache_idx': -1} if di['data_path'] == removal_keys[0] else di for di in self.data_info]
            self.data_info = [{'data_path': di['data_path'], 'label': di['label'], 'name': di['name'], 'cache_idx': -1} if di['data_path'] == removal_keys[0] else di for di in self.data_info]

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
        # print(self.data_cache[fp][cache_idx])
        return self.data_cache[fp][cache_idx], label, name


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

    home = Path.cwd().parts[1]
    train_csv = f'/{home}/ylan/DeepGraft_project/code/debug_train.csv'
    data_root = f'/{home}/ylan/DeepGraft/dataset/hdf5/256_256um_split/'
    # label_path = f'/{home}/ylan/DeepGraft_project/code/split_PAS_bin.json'
    label_path = f'/{home}/ylan/DeepGraft/training_tables/split_Aachen_PAS_all.json'
    output_path = f'/{home}/ylan/DeepGraft/dataset/check/256_256um_split/'
    # os.makedirs(output_path, exist_ok=True)


    dataset = HDF5MILDataloader(data_root, label_path=label_path, mode='train', load_data=False, n_classes=6)
    data = DataLoader(dataset, batch_size=1)

    # print(len(dataset))
    x = 0
    c = 0
    for item in data: 
        if c >=10:
            break
        bag, label, name = item
        print(bag)
        # # print(bag.shape)
        # if bag.shape[1] == 1:
        #     print(name)
        #     print(bag.shape)
        # print(bag.shape)
        # print(name)
        # out_dir = Path(output_path) / name
        # os.makedirs(out_dir, exist_ok=True)

        # # print(item[2])
        # # print(len(item))
        # # print(item[1])
        # # print(data.shape)
        # # data = data.squeeze()
        # bag = item[0]
        # bag = bag.squeeze()
        # for i in range(bag.shape[0]):
        #     img = bag[i, :, :, :]
        #     img = img.squeeze()
        #     img = img*255
        #     img = img.numpy().astype(np.uint8).transpose(1,2,0)
            
        #     img = Image.fromarray(img)
        #     img = img.convert('RGB')
        #     img.save(f'{out_dir}/{i}.png')
        c += 1
        # else: break
        # print(data.shape)
        # print(label)