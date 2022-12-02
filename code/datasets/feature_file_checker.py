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

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_coords(batch_names): #ToDO: Change function for precise coords
    coords = []
    
    for tile_name in batch_names: 
        # print(tile_name)
        pos = re.findall(r'\((.*?)\)', tile_name)
        x, y = pos[-1].split('_')
        coords.append((int(x),int(y)))
    return coords



if __name__ == '__main__':


    home = Path.cwd().parts[1]
    
    data_root = Path(f'/{home}/ylan/data/DeepGraft/224_128um_v2')
    # output_path = Path(f'/{home}/ylan/wsi_tools/debug/zarr')
    # cohorts = ['Leuven'] #, 
    cohorts = ['DEEPGRAFT_RU'] #, 
    # cohorts = ['Aachen_Biopsy_Slides'] #, 
    # cohorts = ['DEEPGRAFT_RU', 'DEEPGRAFT_RA', 'Leuven'] #, 
    # cohorts = ['Aachen_Biopsy_Slides', 'DEEPGRAFT_RU', 'DEEPGRAFT_RA'] #, 
    # cohorts = ['Aachen_Biopsy_Slides', 'DEEPGRAFT_RU', 'DEEPGRAFT_RA', 'Leuven'] #, 
    
    for f in data_root.iterdir():
        
        if f.stem in cohorts:
            print(f)
            fe_path = f / 'FEATURES_RETCCL'
            fe_path.mkdir(parents=True, exist_ok=True)
            slide_list = []
            counter = 0
            for slide in (f / 'BLOCKS').iterdir():
                if Path(slide).is_dir(): 
                    if slide.suffix != '.zarr':
                        slide_list.append(slide)

            print(len(slide_list))

            with tqdm(total=len(slide_list)) as pbar:
                for slide in slide_list:
                    output_path = fe_path / Path(str(slide.stem) + '.zarr')
                    # print('slide: ', slide)

                    # run every slide 5 times for augments
                    for n in range(6):

                        if n != 5:
                            output_path = fe_path / Path(str(slide.stem) + f'_aug{n}.zarr')
                        else: 
                            output_path = fe_path / Path(str(slide.stem) + '.zarr')
                        if output_path.is_dir():
                                # print(output_path, ' skipped.')
                            pbar.update(1)
                            continue
                            
                        else: 
                            counter += 1
                            print(output_path)
                            pbar.update(1)
            print(counter)

                  