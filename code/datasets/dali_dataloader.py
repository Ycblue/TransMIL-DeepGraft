from nvidia.dali import pipeline_def
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import json
import numpy as np
import cupy as cp
import torch
import imageio

batch_size = 10
home = Path.cwd().parts[1]
# image_filename = f"/{home}/ylan/data/DeepGraft/224_128um/Aachen_Biopsy_Slides/BLOCKS/"

class ExternalInputIterator(object):
    def __init__(self, batch_size):
        self.file_path = f"/{home}/ylan/data/DeepGraft/224_128um/"
        # self.label_file = f'/{home}/ylan/DeepGraft/training_tables/split_PAS_tcmr_viral.json'
        self.label_path = f'/{home}/ylan/DeepGraft/training_tables/split_Aachen_PAS_tcmr_viral.json'
        self.batch_size = batch_size
        
        mode = 'test'
        # with open(self.images_dir + "file_list.txt", 'r') as f:
        #     self.files = [line.rstrip() for line in f if line != '']
        self.slideLabelDict = {}
        self.files = []
        self.empty_slides = []
        with open(self.label_path, 'r') as f:
            temp_slide_label_dict = json.load(f)[mode]
            for (x, y) in temp_slide_label_dict:
                x = Path(x).stem 

                # x_complete_path = Path(self.file_path)/Path(x)
                for cohort in Path(self.file_path).iterdir():
                    x_complete_path = Path(self.file_path) / cohort / 'BLOCKS' / Path(x)
                    if x_complete_path.is_dir():
                        if len(list(x_complete_path.iterdir())) > 50:
                        # print(x_complete_path)
                            # self.slideLabelDict[x] = y
                            self.files.append((x_complete_path, y))
                        else: self.empty_slides.append(x_complete_path)
        
        # shuffle(self.files)

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []
        file_names = []

        for _ in range(self.batch_size):
            wsi_batch = []
            wsi_filename, label = self.files[self.i]
            # jpeg_filename, label = self.files[self.i].split(' ')
            name = Path(wsi_filename).stem
            file_names.append(name)
            for img_path in Path(wsi_filename).iterdir():
                # f = open(img, 'rb')

                f = imageio.imread(img_path)
                img = cp.asarray(f)
                wsi_batch.append(img.astype(cp.uint8))
            wsi_batch = cp.stack(wsi_batch)
            batch.append(wsi_batch)
            labels.append(cp.array([label], dtype = cp.uint8))
            
            self.i = (self.i + 1) % self.n
        # print(batch)
        # print(labels)
        return (batch, labels)


eii = ExternalInputIterator(batch_size=10)

@pipeline_def()
def hsv_pipeline(device, hue, saturation, value):
    # files, labels = fn.readers.file(file_root=image_filename)
    files, labels = fn.external_source(source=eii, num_outputs=2, dtype=types.UINT8)
    images = fn.decoders.image(files, device = 'cpu' if device == 'cpu' else 'mixed')
    converted = fn.hsv(images, hue=hue, saturation=saturation, value=value)
    return images, converted

def display(outputs, idx, columns=2, captions=None, cpu=True):
    rows = int(math.ceil(len(outputs) / columns))
    fig = plt.figure()
    fig.set_size_inches(16, 6 * rows)
    gs = gridspec.GridSpec(rows, columns)
    row = 0
    col = 0
    for i, out in enumerate(outputs):
        plt.subplot(gs[i])
        plt.axis("off")
        if captions is not None:
            plt.title(captions[i])
        plt.imshow(out.at(idx) if cpu else out.as_cpu().at(idx))

# pipe_cpu = hsv_pipeline(device='cpu', hue=120, saturation=1, value=0.4, batch_size=batch_size, num_threads=1, device_id=0)
# pipe_cpu.build()
# cpu_output = pipe_cpu.run()

# display(cpu_output, 3, captions=["Original", "Hue=120, Saturation=1, Value=0.4"])

# pipe_gpu = hsv_pipeline(device='gpu', hue=120, saturation=2, value=1, batch_size=batch_size, num_threads=1, device_id=0)
# pipe_gpu.build()
# gpu_output = pipe_gpu.run()

# display(gpu_output, 0, cpu=False, captions=["Original", "Hue=120, Saturation=2, Value=1"])


pipe_gpu = Pipeline(batch_size=batch_size, num_threads=2, device_id=0)
with pipe_gpu:
    images, labels = fn.external_source(source=eii, num_outputs=2, device="gpu", dtype=types.UINT8)
    enhance = fn.brightness_contrast(images, contrast=2)
    pipe_gpu.set_outputs(enhance, labels)

pipe_gpu.build()
pipe_out_gpu = pipe_gpu.run()
batch_gpu = pipe_out_gpu[0].as_cpu()
labels_gpu = pipe_out_gpu[1].as_cpu()
# file_names = pipe_out_gpu[2].as_cpu()

output_path = f"/{home}/ylan/data/DeepGraft/224_128um/debug/augments/"
output_path.mkdir(exist_ok=True)


img = batch_gpu.at(2)
print(img.shape)
print(labels_gpu.at(2))
plt.axis('off')
plt.imsave(f'{output_path}/0.jpg', img[0, :, :, :])