import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import zarr
from numcodecs import Blosc
import torch
import torch.nn as nn
import ResNet as ResNet 
import torchvision.transforms as transforms
import torch.nn.functional as F
import re
from imgaug import augmenters as iaa
import argparse

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--augment', default=False, action='store_true')
    parser.add_argument('--cohort', default='RU', type=str)
    
    args = parser.parse_args()
    return args

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_coords(batch_names): #ToDO: Change function for precise coords
    coords = []
    
    for tile_name in batch_names: 
        # print(tile_name)
        pos = re.findall(r'\((.*?)\)', tile_name)
        # pos = pos
        x, y = pos[-1].replace('-', '_').split('_')
        coords.append((int(x),int(y)))
    return coords

def iaa_augment(img):

    sometimes = lambda aug: iaa.Sometimes(0.5, aug, name="Random1")
    sometimes2 = lambda aug: iaa.Sometimes(0.2, aug, name="Random2")
    sometimes3 = lambda aug: iaa.Sometimes(0.9, aug, name="Random3")
    sometimes4 = lambda aug: iaa.Sometimes(0.9, aug, name="Random4")
    sometimes5 = lambda aug: iaa.Sometimes(0.9, aug, name="Random5")

    transforms = iaa.Sequential([
        iaa.AddToHueAndSaturation(value=(-30, 30), name="MyHSV"), #13
        sometimes2(iaa.GammaContrast(gamma=(0.85, 1.15), name="MyGamma")),
        iaa.Fliplr(0.5, name="MyFlipLR"),
        iaa.Flipud(0.5, name="MyFlipUD"),
        sometimes(iaa.Rot90(k=1, keep_size=True, name="MyRot90")),
        iaa.OneOf([
            sometimes3(iaa.PiecewiseAffine(scale=(0.015, 0.02), cval=0, name="MyPiece")),
            sometimes4(iaa.ElasticTransformation(alpha=(100, 200), sigma=20, cval=0, name="MyElastic")),
            sometimes5(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, rotate=(-45, 45), shear=(-4, 4), cval=0, name="MyAffine"))
        ], name="MyOneOf")
    ])
    seq_img_d = transforms.to_deterministic()
    img = seq_img_d.augment_image(img)

    return img


if __name__ == '__main__':

    torch.set_num_threads(8)
    torch.manual_seed(2022)

    args = make_parse()
    
    augment=args.augment
    cohorts = [args.cohort]
    print('Augment Data: ', augment)
    print('Cohort: ', cohorts)

    home = Path.cwd().parts[1]
    
    data_root = Path(f'/{home}/ylan/data/DeepGraft/224_128uM_annotated')
    # output_path = Path(f'/{home}/ylan/wsi_tools/debug/zarr')
    # cohorts = ['RU', 'RA'] #, 
    # cohorts = ['Aachen_Biopsy_Slides'] #, 
    # cohorts = ['Aachen_Biopsy_Slides', 'DEEPGRAFT_RU', 'DEEPGRAFT_RA', 'Leuven'] #, 
    compressor = Blosc(cname='blosclz', clevel=3)

    val_transforms = transforms.Compose([
            # 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            # RangeNormalization(),
        ])


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()
    n_classes = 2
    # out_features = 1024
    model_ft = ResNet.resnet50(num_classes=1024, mlp=False, two_branch=False, normlinear=True)
    
    model_ft.fc = nn.Identity()
    # print(model_ft)
    # model_ft.fc = nn.Linear(2048, out_features)
    home = Path.cwd().parts[1]
    model_ft.load_state_dict(torch.load(f'/{home}/ylan/workspace/TransMIL-DeepGraft/code/models/ckpt/retccl_best_ckpt.pth'), strict=True)
    # for param in model_ft.parameters():
    #     param.requires_grad = False
    # for m in model_ft.modules():
    #     if isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
    #         m.eval()
    #         m.weight.requires_grad = False
    #         m.bias.requires_grad = False
    # model_ft.fc = nn.Linear(2048, out_features)
    model_ft.eval()
    model_ft.to(device)

    batch_size = 100


    for f in data_root.iterdir():
        
        if f.stem in cohorts:
            # fe_path = Path(test_output_path) / 'FEATURES_RETCCL'
            fe_path = f / 'FEATURES_RETCCL_2048'

            fe_path.mkdir(parents=True, exist_ok=True)
            
            # num_files = len(list((f / 'BLOCKS').iterdir()))
            slide_list = []
            for slide in (f / 'BLOCKS').iterdir():
                if Path(slide).is_dir(): 
                    if slide.suffix != '.zarr':
                        slide_list.append(slide)
            if augment:
                tqdm_len = len(slide_list)*5
            else: tqdm_len = len(slide_list)

            
            with tqdm(total=tqdm_len) as pbar:
                for slide in slide_list:

                    


                    # print('slide: ', slide)

                    # run every slide 5 times for augments
                    if not augment:
                        output_path = fe_path / Path(str(slide.stem) + '.zarr')
                        # if output_path.is_dir():
                        #     pbar.update(1)
                        #     print(output_path, ' skipped.')
                        #     continue
                            # else:
                        output_array = []
                        output_batch_names = []
                        for tile_path_batch in chunker(list(slide.iterdir()), batch_size):
                            batch_array = []
                            batch_names = []
                            for t in tile_path_batch:
                                # for n in range(5):
                                img = np.asarray(Image.open(str(t))).astype(np.uint8) #.astype(np.uint8)
                                img = val_transforms(img.copy()).to(device)
                                batch_array.append(img)

                                tile_name = t.stem
                                batch_names.append(tile_name)
                            if len(batch_array) == 0:
                                continue
                            else:
                                batch_array = torch.stack(batch_array) 
                                with torch.cuda.amp.autocast():
                                    model_output = model_ft(batch_array).detach()
                                output_array.append(model_output)
                                output_batch_names += batch_names 
                        if len(output_array) == 0:
                            pbar.update(1)
                            continue
                        else:
                            output_array = torch.cat(output_array, dim=0).cpu().numpy()
                            output_batch_coords = get_coords(output_batch_names)
                            zarr.save_group(output_path, data=output_array, coords=output_batch_coords)

                            # test eval mode!
                            # z_test = zarr.open(output_path, 'r')
                            # # print(z_test.tree())
                            
                            # if np.all(output_array== z_test['data'][:]):
                            #     print('data same')
                            # else: print(slide)
                            # if np.all(z['tile_names'][:] == z_test['tile_names'][:]):
                            #     print('tile_names true')
                            #     print(output_path ' ')
                            # print(np.all(z[:] == z_test[:]))
                            pbar.update(1)
                    else:
                        for n in range(5):
                            # if n != 5:
                            output_path = fe_path / Path(str(slide.stem) + f'_aug{n}.zarr')
                            if output_path.is_dir():
                                pbar.update(1)
                                # print(output_path, ' skipped.')
                                continue
                            # else:
                            output_array = []
                            output_batch_names = []
                            for tile_path_batch in chunker(list(slide.iterdir()), batch_size):
                                batch_array = []
                                batch_names = []
                                for t in tile_path_batch:
                                    # for n in range(5):
                                    img = np.asarray(Image.open(str(t))).astype(np.uint8) #.astype(np.uint8)
                                    img = iaa_augment(img)
                                    img = val_transforms(img.copy()).to(device)
                                    batch_array.append(img)

                                    tile_name = t.stem
                                    batch_names.append(tile_name)
                                if len(batch_array) == 0:
                                    continue
                                else:
                                    batch_array = torch.stack(batch_array) 
                                    with torch.cuda.amp.autocast():
                                        model_output = model_ft(batch_array).detach()
                                    output_array.append(model_output)
                                    output_batch_names += batch_names 
                            if len(output_array) == 0:
                                pbar.update(1)
                                continue
                            else:
                                output_array = torch.cat(output_array, dim=0).cpu().numpy()
                                output_batch_coords = get_coords(output_batch_names)
                                zarr.save_group(output_path, data=output_array, coords=output_batch_coords)

                                pbar.update(1)
            

                  