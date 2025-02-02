import inspect # 查看python 类的参数和模块、函数代码
import importlib # In order to dynamically import the library
from typing import Optional
import pytorch_lightning as pl
# from pytorch_lightning.loops.base import Loop
# from pytorch_lightning.loops.fit_loop import FitLoop

from torch.utils.data import random_split, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.dataset import Dataset, Subset
from torchvision.datasets import MNIST
from torchvision import transforms
# from .camel_dataloader import FeatureBagLoader
from .custom_dataloader import HDF5MILDataloader
from .jpg_dataloader import JPGMILDataloader
from .classic_jpg_dataloader import JPGBagLoader, LazyJPGBagLoader
from .zarr_feature_dataloader_simple import ZarrFeatureBagLoader
# from .feature_dataloader_mixed import FeatureBagLoader
from .feature_dataloader import FeatureBagLoader
from .local_feature_dataloader import LocalFeatureBagLoader
from pathlib import Path
# from transformers import AutoFeatureExtractor
from torchsampler import ImbalancedDatasetSampler

from abc import ABC, abstractclassmethod, abstractmethod
from sklearn.model_selection import KFold
import numpy as np
import torch


# class DataInterface(pl.LightningDataModule):

#     def __init__(self, train_batch_size=64, train_num_workers=8, test_batch_size=1, test_num_workers=1,dataset_name=None, **kwargs):
#         """[summary]

#         Args:
#             batch_size (int, optional): [description]. Defaults to 64.
#             num_workers (int, optional): [description]. Defaults to 8.
#             dataset_name (str, optional): [description]. Defaults to ''.
#         """        
#         super().__init__()

#         self.train_batch_size = train_batch_size
#         self.train_num_workers = train_num_workers
#         self.test_batch_size = test_batch_size
#         self.test_num_workers = test_num_workers
#         self.dataset_name = dataset_name
#         self.kwargs = kwargs
#         self.load_data_module()
#         home = Path.cwd().parts[1]
#         self.data_root = f'/{home}/ylan/RCC_project/rcc_classification/datasets/Camelyon16/Camelyon16.csv'

 

#     def prepare_data(self):
#         # 1. how to download
#         # MNIST(self.data_dir, train=True, download=True)
#         # MNIST(self.data_dir, train=False, download=True)
#         ...

#     def setup(self, stage=None):
#         # 2. how to split, argument
#         """  
#         - count number of classes

#         - build vocabulary

#         - perform train/val/test splits

#         - apply transforms (defined explicitly in your datamodule or assigned in init)
#         """
#         # Assign train/val datasets for use in dataloaders
#         if stage == 'fit' or stage is None:
#             dataset = FeatureBagLoader(data_root = self.data_root,
#                                                 train=True)
#             a = int(len(dataset)* 0.8)
#             b = int(len(dataset) - a)
#             # print(a)
#             # print(b)
#             self.train_dataset, self.val_dataset = random_split(dataset, [a, b]) # returns data.Subset

#             # self.train_dataset = self.instancialize(state='train')
#             # self.val_dataset = self.instancialize(state='val')
 

#         # Assign test dataset for use in dataloader(s)
#         if stage == 'test' or stage is None:
#             # self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
#             self.test_dataset = FeatureBagLoader(data_root = self.data_root,
#                                                 train=False)
#             # self.test_dataset = self.instancialize(state='test')


#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)

#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=self.test_num_workers, shuffle=False)


#     def load_data_module(self):
#         camel_name =  ''.join([i.capitalize() for i in (self.dataset_name).split('_')])
#         try:
#             self.data_module = getattr(importlib.import_module(
#                 f'datasets.{self.dataset_name}'), camel_name)
#         except:
#             raise ValueError(
#                 'Invalid Dataset File Name or Invalid Class Name!')
    
#     def instancialize(self, **other_args):
#         """ Instancialize a model using the corresponding parameters
#             from self.hparams dictionary. You can also input any args
#             to overwrite the corresponding value in self.kwargs.
#         """
#         class_args = inspect.getargspec(self.data_module.__init__).args[1:]
#         inkeys = self.kwargs.keys()
#         args1 = {}
#         for arg in class_args:
#             if arg in inkeys:
#                 args1[arg] = self.kwargs[arg]
#         args1.update(other_args)
#         return self.data_module(**args1)

class MILDataModule(pl.LightningDataModule):

    def __init__(self, data_root: str, label_path: str, model_name: str, batch_size: int=1, num_workers: int=8, n_classes=2, cache: bool=True, use_features=False, train_classic=False, mixup=False, aug=False, fine_tune=False, bag_size=500, *args, **kwargs):
        super().__init__()
        self.data_root = data_root
        self.label_path = label_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = 224
        self.n_classes = n_classes
        self.target_number = 9
        self.mean_bag_length = 10
        self.var_bag_length = 2
        self.num_bags_train = 200
        self.num_bags_test = 50
        self.seed = 1
        self.mixup = mixup
        self.aug = aug
        self.train_classic = train_classic
        self.fine_tune = fine_tune
        self.max_bag_size = bag_size
        self.model_name = model_name
        self.use_features = use_features
        self.in_features = kwargs['in_features']

        if use_features:
            self.feature_extractor = kwargs['feature_extractor']
        # if self.feature_
        # elif self.feature_extractor == 'histoencoder':

            self.fe_name = f'FEATURES_{self.feature_extractor.upper()}_{self.in_features}'
        else:
            self.fe_name = None


        self.class_weight = []
        self.cache = cache
        self.fe_transform = None
        # print('use_features: ', use_features)
        if self.train_classic: 
            # self.base_dataloader = LazyJPGBagLoader
            self.base_dataloader = JPGBagLoader
        elif not use_features: 
            self.base_dataloader = JPGMILDataloader
        else: 
            self.base_dataloader = FeatureBagLoader
            # self.cache = True
        if model_name == 'resnet50' or model_name == 'CTMIL':
            self.base_dataloader = LocalFeatureBagLoader

        print(f'Using {self.base_dataloader} dataloader')

    def setup(self, stage: Optional[str] = None) -> None:
        home = Path.cwd().parts[1]
        # print('batch size: ', self.batch_size)
        # print('valid_data')
        
        ####
        # mode='test for valid_data ONLY FOR PROTOTYPING!!!
        ####
        self.valid_data = self.base_dataloader(self.data_root, label_path=self.label_path, mode='val', n_classes=self.n_classes, cache=self.cache, model=self.model_name, feature_extractor=self.fe_name) #, max_bag_size=self.max_bag_size
        if stage in (None, 'fit'):
            # print('self.fine_tune', self.fine_tune)
            if self.fine_tune:
                self.train_data = self.base_dataloader(self.data_root, label_path=self.label_path, mode='fine_tune', n_classes=self.n_classes, cache=self.cache, mixup=self.mixup, aug=self.aug, model=self.model_name, feature_extractor=self.fe_name, max_bag_size=self.max_bag_size) #, max_bag_size=self.max_bag_size
            else:
                self.train_data = self.base_dataloader(self.data_root, label_path=self.label_path, mode='train', n_classes=self.n_classes, cache=self.cache, mixup=self.mixup, aug=self.aug, model=self.model_name, feature_extractor=self.fe_name) #, max_bag_size=self.max_bag_size
            # self.valid_data = self.base_dataloader(self.data_root, label_path=self.label_path, mode='val', n_classes=self.n_classes, cache=self.cache, model=self.model_name)

            # dataset = JPGMILDataloader(self.data_root, label_path=self.label_path, mode='train', n_classes=self.n_classes)
            # print(self.base_dataloader)
            print('Train Data: ', len(self.train_data))
            print('Val Data: ', len(self.valid_data))
            # a = int(len(dataset)* 0.8)
            # b = int(len(dataset) - a)
            # self.train_data, self.valid_data = random_split(dataset, [a, b])

            # self.weights = self.get_weights(dataset)



        if stage in (None, 'test'):
            
            self.test_data = self.base_dataloader(self.data_root, label_path=self.label_path, mode='test', n_classes=self.n_classes, cache=False, model=self.model_name, mixup=False, aug=False, feature_extractor=self.fe_name) #, max_bag_size=self.max_bag_size
            # self.test_data = self.base_dataloader(self.data_root, label_path=self.label_path, mode='test', n_classes=self.n_classes, cache=False, model=self.model_name, mixup=False, aug=False, feature_extractor=self.fe_name) #, max_bag_size=self.max_bag_size

        return super().setup(stage=stage)

        

    def train_dataloader(self) -> DataLoader:
        # return DataLoader(self.train_data,  batch_size = self.batch_size, num_workers=self.num_workers) #batch_transforms=self.transform, pseudo_batch_dim=True, 
        # return DataLoader(self.train_data,  batch_size = self.batch_size, sampler = WeightedRandomSampler(self.weights, len(self.weights)), num_workers=self.num_workers) #batch_transforms=self.transform, pseudo_batch_dim=True, 
        if self.train_classic or not self.use_features:
            return DataLoader(self.train_data, batch_size = self.batch_size, num_workers=self.num_workers) #batch_transforms=self.transform, pseudo_batch_dim=True, 
        else:
            return DataLoader(self.train_data,  batch_size = self.batch_size, sampler=ImbalancedDatasetSampler(self.train_data), num_workers=self.num_workers, collate_fn=self.simple_collate) #batch_transforms=self.transform, pseudo_batch_dim=True, 
            # return DataLoader(self.train_data,  batch_size = self.batch_size, num_workers=self.num_workers) #batch_transforms=self.transform, pseudo_batch_dim=True, 
        #sampler=ImbalancedDatasetSampler(self.train_data)
    def val_dataloader(self) -> DataLoader:
        if self.train_classic or not self.use_features:

            return DataLoader(self.valid_data, batch_size = self.batch_size, num_workers=self.num_workers)
        else:
            return DataLoader(self.valid_data, batch_size = 1, sampler=ImbalancedDatasetSampler(self.valid_data), num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        if self.train_classic:
            return DataLoader(self.test_data, batch_size = self.batch_size, num_workers=self.num_workers)
        else: return DataLoader(self.test_data, batch_size = 1, num_workers=self.num_workers)

    def simple_collate(self, data):
        # print(data[0])
        bags = [i[0] for i in data]
        labels = [i[1] for i in data]
        name = [i[2][0] for i in data]
        patient = [i[2][1] for i in data]
        bags = torch.stack(bags)
        labels = torch.Tensor(np.stack(labels, axis=0)).long()
        return bags, labels, (name, patient)

    def custom_collate_fn(self, batch):
        # out_batch = [i for i in batch]
        # for i in range(len(batch)):
        # x = torch.stack(list(batch))

        out_batch = [i[0] for i in batch]
        labels = [i[1] for i in batch]
        wsi_name = [i[2][0] for i in batch]
        batch_coords = [i[2][1] for i in batch]
        patient = [i[2][2] for i in batch]

        # print(x.shape)
        return out_batch, labels, (wsi_name, batch_coords, patient)
            

    def get_weights(self, dataset):

        label_count = [0]*self.n_classes
        labels = dataset.get_labels(np.arange(len(dataset)))
        for i in labels:
            label_count[i] += 1
        weights_per_class = [0.] * self.n_classes
        for i in range(self.n_classes):
            weights_per_class[i] = float(len(labels) / float(label_count[i]))
        weights_per_class = [i / sum(weights_per_class) for i in weights_per_class]
        weights = [0.] * len(labels)
        for i in range(len(labels)):
            weights[i] = weights_per_class[labels[i]]

        return torch.DoubleTensor(weights)
    
    def get_mixup_bag(self, bag):

        bag_size = bag.shape[0]

        a = torch.rand([bag_size])
        b = 0.6
        rand_x = torch.randint(0, bag_size, [bag_size,])
        rand_y = torch.randint(0, bag_size, [bag_size,])

        bag_x = bag[rand_x, :]
        bag_y = bag[rand_y, :]

        temp_bag = (bag_x.t()*a).t() + (bag_y.t()*(1.0-a)).t()
        # print('temp_bag: ', temp_bag.shape)

        if bag_size < self.max_bag_size:
            diff = self.max_bag_size - bag_size
            bag_idxs = torch.randperm(bag_size)[:diff]
            
            # print('bag: ', bag.shape)
            # print('bag_idxs: ', bag_idxs.shape)
            mixup_bag = torch.cat((bag, temp_bag[bag_idxs, :]))
            # print('mixup_bag: ', mixup_bag.shape)
        else:
            random_sample_list = torch.rand(bag_size)
            mixup_bag = [bag[i] if random_sample_list[i] else temp_bag[i] > b for i in range(bag_size)] #make pytorch native?!
            mixup_bag = torch.stack(mixup_bag)
            # print('else')
            # print(mixup_bag.shape)

        return mixup_bag

class DataModule(pl.LightningDataModule):

    def __init__(self, data_root: str, label_path: str, batch_size: int=1, num_workers: int=8, n_classes=2, cache: bool=True, backbone=None, *args, **kwargs):
        super().__init__()
        self.data_root = data_root
        self.label_path = label_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = 384
        self.n_classes = n_classes
        self.target_number = 9
        self.mean_bag_length = 10
        self.var_bag_length = 2
        self.num_bags_train = 200
        self.num_bags_test = 50
        self.seed = 1

        self.backbone = backbone
        self.cache = True
        self.fe_transform = None


    def setup(self, stage: Optional[str] = None) -> None:
        home = Path.cwd().parts[1]
        
        if stage in (None, 'fit'):
            dataset = HDF5MILDataloader(self.data_root, label_path=self.label_path, mode='train', n_classes=self.n_classes, backbone=self.backbone)
            a = int(len(dataset)* 0.8)
            b = int(len(dataset) - a)
            self.train_data, self.valid_data = random_split(dataset, [a, b])

        if stage in (None, 'test'):
            self.test_data = HDF5MILDataloader(self.data_root, label_path=self.label_path, mode='test', n_classes=self.n_classes, backbone=self.backbone)


        return super().setup(stage=stage)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data,  self.batch_size, sampler=ImbalancedDatasetSampler(self.train_data),shuffle=False,) #batch_transforms=self.transform, pseudo_batch_dim=True, 
        #sampler=ImbalancedDatasetSampler(self.train_data),
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_data, batch_size = self.batch_size)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size = self.batch_size) #, num_workers=self.num_workers


class BaseKFoldDataModule(pl.LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass

class CrossVal_MILDataModule(BaseKFoldDataModule):

    def __init__(self, data_root: str, label_path: str, batch_size: int=1, num_workers: int=8, n_classes=2, cache: bool=True, backbone=None, *args, **kwargs):
        super().__init__()
        self.data_root = data_root
        self.label_path = label_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = 384
        self.n_classes = n_classes
        self.target_number = 9
        self.mean_bag_length = 10
        self.var_bag_length = 2
        self.num_bags_train = 200
        self.num_bags_test = 50
        self.seed = 1

        self.backbone = backbone
        self.cache = True
        self.fe_transform = None

        # train_dataset: Optional[Dataset] = None
        # test_dataset: Optional[Dataset] = None
        # train_fold: Optional[Dataset] = None
        # val_fold: Optional[Dataset] = None
        self.train_data : Optional[Dataset] = None
        self.test_data : Optional[Dataset] = None
        self.train_fold : Optional[Dataset] = None
        self.val_fold : Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        home = Path.cwd().parts[1]

        # if stage in (None, 'fit'):
        dataset = JPGMILDataloader(self.data_root, label_path=self.label_path, mode='train', n_classes=self.n_classes)
        # a = int(len(dataset)* 0.8)
        # b = int(len(dataset) - a)
        # self.train_data, self.val_data = random_split(dataset, [a, b])
        self.train_data = dataset

        # if stage in (None, 'test'):,
        self.test_data = JPGMILDataloader(self.data_root, label_path=self.label_path, mode='test', n_classes=self.n_classes)

        # return super().setup(stage=stage)

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_data)))]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_data, train_indices)
        self.val_fold = Subset(self.train_data, val_indices)


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold,  self.batch_size, sampler=ImbalancedDatasetSampler(self.train_fold), num_workers=self.num_workers) #batch_transforms=self.transform, pseudo_batch_dim=True, 
        # return DataLoader(self.train_fold,  self.batch_size, num_workers=self.num_workers, shuffle=True) #batch_transforms=self.transform, pseudo_batch_dim=True, 
        #sampler=ImbalancedDatasetSampler(self.train_data)
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold, batch_size = self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size = self.batch_size, num_workers=self.num_workers)



# if __name__ == '__main__':
