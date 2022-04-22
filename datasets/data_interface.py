import inspect # 查看python 类的参数和模块、函数代码
import importlib # In order to dynamically import the library
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from .camel_dataloader import FeatureBagLoader
from .custom_dataloader import HDF5MILDataloader
from pathlib import Path

class DataInterface(pl.LightningDataModule):

    def __init__(self, train_batch_size=64, train_num_workers=8, test_batch_size=1, test_num_workers=1,dataset_name=None, **kwargs):
        """[summary]

        Args:
            batch_size (int, optional): [description]. Defaults to 64.
            num_workers (int, optional): [description]. Defaults to 8.
            dataset_name (str, optional): [description]. Defaults to ''.
        """        
        super().__init__()

        self.train_batch_size = train_batch_size
        self.train_num_workers = train_num_workers
        self.test_batch_size = test_batch_size
        self.test_num_workers = test_num_workers
        self.dataset_name = dataset_name
        self.kwargs = kwargs
        self.load_data_module()
        home = Path.cwd().parts[1]
        self.data_root = f'/{home}/ylan/RCC_project/rcc_classification/datasets/Camelyon16/Camelyon16.csv'

 

    def prepare_data(self):
        # 1. how to download
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        ...

    def setup(self, stage=None):
        # 2. how to split, argument
        """  
        - count number of classes

        - build vocabulary

        - perform train/val/test splits

        - apply transforms (defined explicitly in your datamodule or assigned in init)
        """
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            dataset = FeatureBagLoader(data_root = self.data_root,
                                                train=True)
            a = int(len(dataset)* 0.8)
            b = int(len(dataset) - a)
            print(a)
            print(b)
            self.train_dataset, self.val_dataset = random_split(dataset, [a, b])
            # self.train_dataset = self.instancialize(state='train')
            # self.val_dataset = self.instancialize(state='val')
 

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            # self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            self.test_dataset = FeatureBagLoader(data_root = self.data_root,
                                                train=False)
            # self.test_dataset = self.instancialize(state='test')


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=self.train_num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=self.test_num_workers, shuffle=False)


    def load_data_module(self):
        camel_name =  ''.join([i.capitalize() for i in (self.dataset_name).split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                f'datasets.{self.dataset_name}'), camel_name)
        except:
            raise ValueError(
                'Invalid Dataset File Name or Invalid Class Name!')
    
    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)

class MILDataModule(pl.LightningDataModule):

    def __init__(self, data_root: str, label_path: str, batch_size: int=1, num_workers: int=8, n_classes=2, cache: bool=True, *args, **kwargs):
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

        self.cache = True


    def setup(self, stage: Optional[str] = None) -> None:
        # if self.n_classes == 2:
        #     if stage in (None, 'fit'):
        #         dataset = HDF5Dataset(self.data_root, mode='train', n_classes=self.n_classes)
        #         a = int(len(dataset)* 0.8)
        #         b = int(len(dataset) - a)
        #         self.train_data, self.valid_data = random_split(dataset, [a, b])

        #     if stage in (None, 'test'):
        #         self.test_data = HDF5Dataset(self.data_root, mode='test', n_classes=self.n_classes)
        # else:
        home = Path.cwd().parts[1]
        # self.label_path = f'{home}/ylan/DeepGraft_project/code/split_debug.json'
        # train_csv = f'/{home}/ylan/DeepGraft_project/code/debug_train_small.csv'
        # test_csv = f'/{home}/ylan/DeepGraft_project/code/debug_test_small.csv'
        

        if stage in (None, 'fit'):
            dataset = HDF5MILDataloader(self.data_root, label_path=self.label_path, mode='train', n_classes=self.n_classes)
            # print(len(dataset))
            a = int(len(dataset)* 0.8)
            b = int(len(dataset) - a)
            self.train_data, self.valid_data = random_split(dataset, [a, b])

        if stage in (None, 'test'):
            self.test_data = HDF5MILDataloader(self.data_root, label_path=self.label_path, mode='test', n_classes=self.n_classes)

        return super().setup(stage=stage)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, self.batch_size, num_workers=self.num_workers, shuffle=True) #batch_transforms=self.transform, pseudo_batch_dim=True, 
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid_data, batch_size = self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size = self.batch_size, num_workers=self.num_workers)