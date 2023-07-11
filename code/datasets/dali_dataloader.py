# import nvidia.dali as dali
# from nvidia.dali import pipeline_def
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
# import nvidia.dali.fn.readers.file as file
from nvidia.dali.fn.decoders import image, image_random_crop

import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy, DALIGenericIterator

from pathlib import Path
import json
from tqdm import tqdm
from random import shuffle
import numpy as np
from PIL import Image
import torch
import os

sequence_length = 800

# Path to MNIST dataset
# data_path = os.path.join(os.environ['DALI_EXTRA_PATH'], 'db/MNIST/training/')
# file_path = f'/home/ylan/data/DeepGraft/224_128um_v2'

class ExternalInputIterator(object):
    def __init__(self, file_path, label_path, mode, n_classes, device_id, num_gpus, batch_size = 1, max_bag_size=sequence_length):

        self.file_path = file_path
        self.label_path = label_path
        self.n_classes = n_classes
        self.mode = mode
        self.max_bag_size = max_bag_size
        self.min_bag_size = 120
        self.batch_size = batch_size

        self.data_info = []
        self.data_cache = {}
        self.files = []
        self.slideLabelDict = {}
        self.empty_slides = []

        home = Path.cwd().parts[1]
        slide_patient_dict_path = f'/homeStor1/ylan/data/DeepGraft/training_tables/slide_patient_dict.json'
        with open(slide_patient_dict_path, 'r') as f:
            self.slidePatientDict = json.load(f)

        with open(self.label_path, 'r') as f:
            temp_slide_label_dict = json.load(f)[mode]
            print(len(temp_slide_label_dict))
            for (x, y) in temp_slide_label_dict:
                x = Path(x).stem 
                # x_complete_path = Path(self.file_path)/Path(x)
                for cohort in Path(self.file_path).iterdir():
                    x_complete_path = Path(self.file_path) / cohort / 'BLOCKS' / Path(x)
                    if x_complete_path.is_dir():
                        if len(list(x_complete_path.iterdir())) > self.min_bag_size:
                        # print(x_complete_path)
                            self.slideLabelDict[x] = y
                            patient = self.slidePatientDict[x_complete_path.stem]
                            self.files.append((x_complete_path, y, patient))
                        else: self.empty_slides.append(x_complete_path)
        

        # for slide_dir in tqdm(self.files):
        #     # self._add_data_infos(str(slide_dir.resolve()), load_data, slide_patient_dict)
        #     wsi_name = Path(slide_dir).stem
        #     if wsi_name in self.slideLabelDict:
        #         label = self.slideLabelDict[wsi_name]
        #         patient = self.slidePatientDict[wsi_name]
        #         idx = -1
        #         self.data_info.append({'data_path': file_path, 'label': label, 'name': wsi_name, 'patient': patient,'cache_idx': idx})

        self.dataset_len = len(self.files)

        self.files = self.files[self.dataset_len * device_id//num_gpus:self.dataset_len*(device_id+1)//num_gpus]

        self.n = len(self.files)

        # test_data_root = os.environ['DALI_EXTRA_PATH']
        # jpeg_file = os.path.join(test_data_root, 'db', 'single', 'jpeg', '510', 'ship-1083562_640.jpg')

    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self
    
    def __next__(self):
        batch = [] 
        labels = []

        if self.i >=self.n:
            self.__iter__()
            raise StopIteration
        
        # for _ in range(self.batch_size):
        wsi_path, label, patient = self.files[self.i]
        wsi_batch = []
        for tile_path in Path(wsi_path).iterdir():
            np_img = np.fromfile(tile_path, dtype=np.uint8)

            batch.append(np_img)

        # test_data_root = os.environ['DALI_EXTRA_PATH']
        # jpeg_file = os.path.join(test_data_root, 'db', 'single', 'jpeg', '510', 'ship-1083562_640.jpg')
        # wsi_batch = [np.fromfile(jpeg_file, dtype=np.uint8) for _ in range(sequence_length)]

            # np_img = np.asarray(Image.open(tile_path)).astype(np.uint8)
            # print(np_img.shape)
            # wsi_batch.append(np_img)
            
            # print(np_img)

        
        
        # wsi_batch = np.stack(wsi_batch, axis=0) 
        # # print(wsi_batch.shape)
        # print(wsi_batch)
        # print(len(wsi_batch))
        # if len(wsi_batch) > self.max_bag_size:
        wsi_batch, _ = self.to_fixed_size_bag(batch, self.max_bag_size)
        # batch.append(wsi_batch)
        batch = wsi_batch
        batch.append(torch.tensor([int(label)], dtype=torch.uint8))    
        self.i += 1
        # for i in range(len(batch)):
        #     print(batch[i].shape)
        #     print(labels[i])
        # print(batch)
        return batch
        # return (batch, labels)       
    
    def __len__(self):
        return self.dataset_len

    def to_fixed_size_bag(self, bag, bag_size):
        
        current_size = len(bag)
        # print(bag)
        
        if current_size < bag_size:
            zero_padded = [np.empty(1000, dtype=np.uint8)] * (bag_size - current_size)
            bag_samples = bag + zero_padded
            # while current_size < bag_size: 
                # bag.append(np.empty(1, dtype=np.uint8)) 

            # zero_padded = np.empty(5000) 
        
        else:

            bag_samples = list(np.random.permutation(bag)[:bag_size])

        # bag_samples = list(np.array(bag, dtype=object)[bag_idxs])

        print(len(bag_samples))

        return bag_samples, min(bag_size, len(bag))
    next = __next__

def ExternalSourcePipeline(batch_size, num_threads, device_id, external_data):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        *jpegs, label = fn.external_source(source=external_data, num_outputs=sequence_length+1, dtype=types.UINT8, batch=False)

        images = fn.decoders.image(jpegs, device="mixed")
        images = fn.resize(images, resize_x=224, resize_y=224)
        
        bag = fn.stack(*images)
        # bag = fn.reshape(bag, layout='')
        # output = fn.cast(bag, dtype=types.UINT8)

        # output.append(images)
        # print(output)
        pipe.set_outputs(bag, label)
    return pipe
# @pipeline_def
# def get_pipeline(file_root:str, size:int=224, validation_size: Optional[int]=256, random_shuffle: bool=False, training:bool=True, decoder_device:str= 'mixed', device:str='gpu'):

#     images, labels = file(file_root=file_root, random_shuffle=random_shuffle, name='Reader')

#     if training: 
#         images = image_random_crop(images,
#                                     random_area=[0.08, 1.0],
#                                     random_aspect_ratio = [0.75, 1.3],
#                                     device=decoder_device,
#         )
#         images = fn.resize(images, 
#                         size=size, 
#                         device=device)
#         mirror = fn.random.coin_flip(
# 			# probability refers to the probability of getting a 1
# 			probability=0.5,
# 		)
#     else: 
#         images = image(images, device=decoder_device)
#         images = resize(images, size=validation_size, mode='not_smaller', device=device)
#         mirror = False

#     images = fn.crop_mirror_normalize(images, 
#                                         crop=(size,size),
#                                         mirror=mirror,
#                                         mean=[0.485 * 255,0.456 * 255,0.406 * 255],
# 		                                std=[0.229 * 255,0.224 * 255,0.225 * 255],
# 		                                device=device,
#         )
#     if device == 'gpu':
#         labels = labels.gpu()

#     return images, labels

# training_pipeline = get_pipeline(batch_size=1, num_threads=8, device_id=0, file_root=f'/home/ylan/data/DeepGraft/224_128um_v2', random_shuffle=True, training=True, size=224)
# validation_pipeline = get_pipeline(batch_size=1, num_threads=8, device_id=0, file_root=f'/home/ylan/data/DeepGraft/224_128um_v2', random_shuffle=True, training=False, size=224)

# training_pipeline.build()
# validation_pipeline.build()

# training_dataloader = DALIClassificationIterator(pipelines=training_pipeline, reader_name='Reader', last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
# validation_pipeline = DALIClassificationIterator(pipelines=validation_pipeline, reader_name='Reader', last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
@pipeline_def(num_threads=4, device_id=self.trainer.local.rank)
def get_dali_pipeline(images_dir):
    images, _ = fn.readers.file(file_root=images_dir, random_shuffle=True, name="Reader")
    # decode data on the GPU
    images = fn.decoders.image_random_crop(images, device="mixed", output_type=types.RGB)
    # the rest of processing happens on the GPU as well
    images = fn.resize(images, resize_x=256, resize_y=256)
    images = fn.crop_mirror_normalize(
        images,
        crop_h=224,
        crop_w=224,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=fn.random.coin_flip(),
    )
    return images

if __name__ == '__main__':

    home = Path.cwd().parts[1]
    file_path = f'/{home}/ylan/data/DeepGraft/224_256uM_annotated'
    label_path = f'/{home}/ylan/data/DeepGraft/training_tables/dg_limit_20_split_PAS_HE_Jones_norm_rest.json'

    train_dataloader = DALIGenericIterator([get_dali_pipeline(batch_size=16)], ['data'])
    # eii = ExternalInputIterator(file_path, label_path, mode="train", n_classes=2, device_id=0, num_gpus=1)

    # pipe = ExternalSourcePipeline(batch_size=1, num_threads=2, device_id = 0,
    #                         external_data = eii)
    # pii = DALIClassificationIterator(pipe, last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)

    # for e in range(3):
    #     for i, data in enumerate(pii):
    #         # print(data)
    #         print("epoch: {}, iter {}, real batch size: {}".format(e, i, len(data[0]["data"])))
    #     pii.reset()

            
