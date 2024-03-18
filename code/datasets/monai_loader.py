import numpy as np
import collections.abc
import torch
import torch.distributed as dist
import torch.nn as nn

from monai.config import KeysCollection
from monai.data import Dataset, load_decathlon_datalist, PersistentDataset
from monai.data.wsi_reader import WSIReader, CuCIMWSIReader
# from monai.data.image_reader import CuCIMWSIReader
from monai.networks.nets import milmodel
from monai.transforms import (
    Compose,
    GridPatchd,
    LoadImaged,
    LoadImage,
    MapTransform,
    RandFlipd,
    RandGridPatchd,
    RandRotate90d,
    ScaleIntensityRanged,
    SplitDimd,
    ToTensord,
)
from sklearn.metrics import cohen_kappa_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import  SummaryWriter
import json
from pathlib import Path
import time

class LabelEncodeIntegerGraded(MapTransform):
    """
    Convert an integer label to encoded array representation of length num_classes,
    with 1 filled in up to label index, and 0 otherwise. For example for num_classes=5,
    embedding of 2 -> (1,1,0,0,0)
    Args:
        num_classes: the number of classes to convert to encoded format.
        keys: keys of the corresponding items to be transformed. Defaults to ``'label'``.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        num_classes: int,
        keys: KeysCollection = "label",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.num_classes = num_classes

    def __call__(self, data):

        d = dict(data)
        for key in self.keys:
            label = int(d[key])

            lz = np.zeros(self.num_classes, dtype=np.float32)
            lz[:label] = 1.0
            # alternative oneliner lz=(np.arange(self.num_classes)<int(label)).astype(np.float32) #same oneliner
            d[key] = lz

        return d

def list_data_collate(batch: collections.abc.Sequence):
    # print(f"{i} = {item['image'].shape=} >> {item['image'].keys=}")
    for i, item in enumerate(batch):
        data = item[0]
        data["image"] = torch.stack([ix["image"] for ix in item], dim=0)
        # data["patch_location"] = torch.stack([ix["patch_location"] for ix in item], dim=0)
        batch[i] = data
    return default_collate(batch)





if __name__ == '__main__':

    num_classes = 2
    batch_size=1
    tile_size = 224
    tile_count = 1000
    home = Path.cwd().parts[1]
    data_root = f'/{home}/datasets/DeepGraft/'
    # labels = [0]
    # data_root = f'/{home}/public/DeepGraft/Aachen_Biopsy_Slides_Extended'
    data = {"training": [{
        "image": 'Aachen_KiBiDatabase_KiBiAcRCIQ360_01_018_PAS.svs', 
        "label": 0
        }, {
        "image": 'Aachen_KiBiDatabase_KiBiAcRCIQ360_01_018_PAS.svs', 
        "label": 0
        }, {
        "image": 'Aachen_KiBiDatabase_KiBiAcRCIQ360_01_018_PAS.svs', 
        "label": 0
        }],
        "validation": [{
        "image": 'Aachen_KiBiDatabase_KiBiAcRCIQ360_01_018_PAS.svs', 
        "label": 0
        }]
    }
    with open('monai_test.json', 'w') as jf:
        json.dump(data, jf)
    json_data_path = f'/homeStor1/ylan/data/DeepGraft/training_tables/dg_decathlon_PAS_HE_Jones_norm_rest.json'

    training_list = load_decathlon_datalist(
        data_list_file_path=json_data_path,
        data_list_key="training",
        base_dir=data_root,
    )
    

    train_transform = Compose(
        [
            LoadImaged(keys=["image"], reader=WSIReader, backend="cucim", dtype=np.uint8, level=1, image_only=True, num_workers=8),
            LabelEncodeIntegerGraded(keys=["label"], num_classes=num_classes),
            RandGridPatchd(
                keys=["image"],
                patch_size=(tile_size, tile_size),
                threshold=0.999 * 3 * 255 * tile_size * tile_size,
                num_patches=None,
                sort_fn="min",
                pad_mode=None,
                constant_values=255,
            ),
            SplitDimd(keys=["image"], dim=0, keepdim=False, list_output=True),
            RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
            RandRotate90d(keys=["image"], prob=0.5),
            ScaleIntensityRanged(keys=["image"], a_min=np.float32(255), a_max=np.float32(0)),
            ToTensord(keys=["image", "label"]),
        ]
    )
    # training_list = data['training']
    # print(training_list)
    # dataset_train = Dataset(data=training_list)
    dataset_train = Dataset(data=training_list, transform=train_transform)
    # persistent_dataset = PersistentDataset(data=training_list, transform=train_transform, cache_dir='/home/ylan/workspace/test')
    

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        sampler=None,
        collate_fn=list_data_collate,
    )

    # print(len(train_loader))
    # start = time.time()
    count = 0

    # train_transform = LoadImage(reader=WSIReader, backend='openslide', level=3)
    # filename = '/home/ylan/DeepGraft/DEEPGRAFT_RU/T19-01474_I1_HE 10_959004.ndpi'
    # X = train_transform(filename)
    # print(X)
    # img, meta = reader.read(data='/home/ylan/DeepGraft/DEEPGRAFT_RU/T19-01474_I1_HE 10_959004.ndpi')

    # print(meta)

    for idx, batch_data in enumerate(train_loader):
        # print(batch_data)
        if count > 10: 
            break
        data, target = batch_data["image"], batch_data["label"]
        print(target)
        count += 1
    end = time.time()
    print('Time: ', end-start)

    # image_reader = WSIReader(backend='cucim')
    # for i in training_list:
    #     # print(i)
    #     wsi = image_reader.read(i['image'])
    #     img_data, meta_data = image_reader.get_data(wsi)
    #     print(meta_data)