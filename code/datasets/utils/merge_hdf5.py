from pathlib import Path
import os
import torch
import h5py
import numpy as np
from tqdm import tqdm

'''
Merge HDF5 files for different uM resolutions. 
Used for mixed resolution MIL training.
'''


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
# data_root = Path(f'/{home}/ylan/data/DeepGraft/224_256uM_annotated')
res1 = 256
res2 = 1024


cohorts = ['DEEPGRAFT_RU', 'DEEPGRAFT_RA']
cohorts = ['DEEPGRAFT_RA']
res1_root = Path(f'/{home}/ylan/data/DeepGraft/224_{res1}uM_annotated')
res2_root = Path(f'/{home}/ylan/data/DeepGraft/224_{res2}uM_annotated')
out_path = Path(f'/{home}/ylan/data/DeepGraft/224_{res1}_{res2}_mixed_annotated')
# cohorts = ['debug']

for c in cohorts:
    
    res1_files = [i for i in (res1_root/c/'FEATURES_RETCCL_2048_HED').iterdir() if i.suffix != '.pt']
    c_out_path = out_path / c 
    c_out_path.mkdir(parents=True, exist_ok=True)
    # print(len(pt_files))

    for r1 in tqdm(res1_files):
        
        file_name = r1.stem
        # print(file_name)
        r2 = res2_root / c /  'FEATURES_RETCCL_2048_HED' / f'{file_name}-1024'
        merged_out = c_out_path / file_name

        with h5py.File(r1, 'r') as hdf5_file:
            r1_np_bag = hdf5_file['features'][:]
            r1_coords = hdf5_file['coords'][:]
        with h5py.File(r2, 'r') as hdf5_file:
            r2_np_bag = hdf5_file['features'][:]
            r2_coords = hdf5_file['coords'][:]
        
        # print(r1_np_bag.shape)
        # print(r1_coords.shape)
        # print(r2_np_bag.shape)
        # print(r2_coords.shape)

        # print(np.concatenate((r1_np_bag, r2_np_bag), axis=0))
        merged_np_bag = np.concatenate((r1_np_bag, r2_np_bag), axis=0)
        merged_coords = np.concatenate((r1_coords, r2_coords), axis=0)
        # hdf5_path = pt.with_suffix('')
        if not merged_out.exists():
        #     coord_hdf5_path = Path('_'.join(str(hdf5_path).split('_')[:-1]))
        #     with h5py.File(coord_hdf5_path, 'r') as hdf5_file:
        #             # np_bag = hdf5_file['features'][:]
        #             coords = hdf5_file['coords'][:]
        #     torch_bag = torch.load(pt)

            asset_dict = {'features': merged_np_bag, 'coords': merged_coords}
            Save_hdf5(merged_out, asset_dict, mode='w') 

            with h5py.File(merged_out, 'r') as hdf5_file:
                np_bag = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]
