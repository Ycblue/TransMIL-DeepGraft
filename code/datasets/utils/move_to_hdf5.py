from pathlib import Path
import os
import torch
import h5py
import numpy as np
from tqdm import tqdm

def Save_hdf5(output_dir, asset_dict, mode='a'):
    
	file = h5py.File(output_dir, mode)

	for key, val in asset_dict.items():
		data_shape = val.shape
        
		if key not in file:
			data_type = val.dtype
			chunk_shape = (1, ) + data_shape[1:]
			maxshape = (None, ) + data_shape[1:]
			dset = file.create_dataset(key, shape = data_shape, maxshape = maxshape, chunks = chunk_shape, dtype = data_type)
			dset[:] = val
		else:
			dset = file[key]
			dset.resize(len(dset) + data_shape[0], axis = 0)
			dset[-data_shape[0]:] = val  

	file.close()
	return output_dir

home = Path.cwd().parts[1]
data_root = Path(f'/{home}/ylan/data/DeepGraft/224_256uM_annotated')

cohorts = ['DEEPGRAFT_RU', 'DEEPGRAFT_RA']
# cohorts = ['debug']

for c in cohorts:
    pt_files = [i for i in (data_root/c/'FEATURES_RETCCL_2048_HED').iterdir() if i.suffix == '.pt' and 'aug' in str(i)]
    # print(len(pt_files))

    for pt in tqdm(pt_files):
        
        hdf5_path = pt.with_suffix('')
        if not hdf5_path.exists():
            coord_hdf5_path = Path('_'.join(str(hdf5_path).split('_')[:-1]))
            with h5py.File(coord_hdf5_path, 'r') as hdf5_file:
                    # np_bag = hdf5_file['features'][:]
                    coords = hdf5_file['coords'][:]
            torch_bag = torch.load(pt)

            asset_dict = {'features': torch_bag.numpy(), 'coords': coords}
            Save_hdf5(hdf5_path, asset_dict, mode='w') 

            with h5py.File(hdf5_path, 'r') as hdf5_file:
                np_bag = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]
