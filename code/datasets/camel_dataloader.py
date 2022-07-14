import pandas as pd

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn.functional import one_hot
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import pandas as pd
from sklearn.utils import shuffle
from pathlib import Path
from tqdm import tqdm


class FeatureBagLoader(data_utils.Dataset):
    def __init__(self, data_root,train=True, cache=True):

        bags_path = pd.read_csv(data_root)

        self.train_path = bags_path.iloc[0:int(len(bags_path)*0.8), :]
        self.test_path = bags_path.iloc[int(len(bags_path)*0.8):, :]
        # self.train_path = shuffle(train_path).reset_index(drop=True)
        # self.test_path = shuffle(test_path).reset_index(drop=True)

        home = Path.cwd().parts[1]
        self.origin_path = Path(f'/{home}/ylan/RCC_project/rcc_classification/')
        # self.target_number = target_number
        # self.mean_bag_length = mean_bag_length
        # self.var_bag_length = var_bag_length
        # self.num_bag = num_bag
        self.cache = cache
        self.train = train
        self.n_classes = 2

        self.features = []
        self.labels = []
        if self.cache:
            if train:
                with tqdm(total=len(self.train_path)) as pbar:
                    for t in tqdm(self.train_path.iloc()):
                        ft, lbl = self.get_bag_feats(t)
                        # ft = ft.view(-1, 512)
                        
                        self.labels.append(lbl)
                        self.features.append(ft)
                        pbar.update()
            else: 
                with tqdm(total=len(self.test_path)) as pbar:
                    for t in tqdm(self.test_path.iloc()):
                        ft, lbl = self.get_bag_feats(t)
                        # lbl = Variable(Tensor(lbl))
                        # ft = Variable(Tensor(ft)).view(-1, 512)
                        self.labels.append(lbl)
                        self.features.append(ft)
                        pbar.update()
        # print(self.get_bag_feats(self.train_path))
        # self.r = np.random.RandomState(seed)

        # self.num_in_train = 60000
        # self.num_in_test = 10000

        # if self.train:
        #     self.train_bags_list, self.train_labels_list = self._create_bags()
        # else:
        #     self.test_bags_list, self.test_labels_list = self._create_bags()

    def get_bag_feats(self, csv_file_df):
        # if args.dataset == 'TCGA-lung-default':
        #     feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
        # else:
        
        feats_csv_path = self.origin_path / csv_file_df.iloc[0]
        df = pd.read_csv(feats_csv_path)
        # feats = shuffle(df).reset_index(drop=True)
        # feats = feats.to_numpy()
        feats = df.to_numpy()
        label = np.zeros(self.n_classes)
        if self.n_classes==2:
            label[1] = csv_file_df.iloc[1]
        else:
            if int(csv_file_df.iloc[1])<=(len(label)-1):
                label[int(csv_file_df.iloc[1])] = 1
        
        return feats, label

    def __len__(self):
        if self.train:
            return len(self.train_path)
        else:
            return len(self.test_path)

    def __getitem__(self, index):

        if self.cache:
            label = self.labels[index]
            feats = self.features[index]
            label = Variable(Tensor(label))
            feats = Variable(Tensor(feats)).view(-1, 512)
            return feats, label
        else:
            if self.train:
                feats, label = self.get_bag_feats(self.train_path.iloc[index])
                label = Variable(Tensor(label))
                feats = Variable(Tensor(feats)).view(-1, 512)
            else:
                feats, label = self.get_bag_feats(self.test_path.iloc[index])
                label = Variable(Tensor(label))
                feats = Variable(Tensor(feats)).view(-1, 512)

            return feats, label

if __name__ == '__main__':
    import os
    cwd = os.getcwd()
    home = cwd.split('/')[1]
    data_root = f'/{home}/ylan/RCC_project/rcc_classification/datasets/Camelyon16/Camelyon16.csv'
    dataset = FeatureBagLoader(data_root, cache=False)
    for i in dataset: 
        # print(i[1])
        # print(i)
        
        features, label = i
        print(label)
        # print(features.shape)
        # print(label[0].long())