import os
import numpy as np
from PIL import Image

from vaik_segmentation_trt_inference.trt_model import TrtModel

input_saved_model_dir_path = os.path.expanduser('~/Desktop/model.trt')
classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
image = np.asarray(
    Image.open(os.path.expanduser('~/.vaik-mnist-segmentation-dataset/valid/valid_000000000_raw.png')).convert('RGB'))

# fp16
model = TrtModel(input_saved_model_dir_path, classes)
output, raw_pred = model.inference([image])