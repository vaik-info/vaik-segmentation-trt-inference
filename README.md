# vaik-segmentation-trt-inference

Inference by segmentation Tensor RT model


## Install

``` shell
pip install git+https://github.com/vaik-info/vaik-segmentation-trt-inference
```

## Usage

### Example

```python
import os
import numpy as np
from PIL import Image

from vaik_segmentation_trt_inference.trt_model import TrtModel

input_saved_model_dir_path = os.path.expanduser('~/Desktop/model.trt')
classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
image = np.asarray(
    Image.open(os.path.expanduser('~/.vaik-mnist-segmentation-dataset/valid/valid_000000000_raw.png')).convert('RGB'))

model = TrtModel(input_saved_model_dir_path, classes)
output, raw_pred = model.inference([image])
```

#### Output

- output

```text
[{'labels': array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]])}]
```

- raw_pred
```
[[[[ 1.42327416e+00 -1.07997835e+00 -5.48077583e-01 ... -7.84463704e-01
    -1.98164642e-01  3.20103407e-01]
   [ 1.42327416e+00 -1.07997835e+00 -5.48077583e-01 ... -7.84463704e-01
    -1.98164642e-01  3.20103407e-01]
   [ 1.35650969e+00 -1.04923391e+00 -4.85398561e-01 ... -7.64718473e-01
    -1.93478346e-01  3.09667230e-01]
   ...
   [ 1.14318311e+00 -9.81163085e-01  9.88265425e-02 ... -6.23952448e-01
    -1.15229025e-01 -9.39857513e-02]
   [ 1.16411352e+00 -1.03283930e+00  1.19960904e-01 ... -6.61158621e-01
    -9.09615234e-02 -1.38090789e-01]
   [ 1.16411352e+00 -1.03283930e+00  1.19960904e-01 ... -6.61158621e-01
    -9.09615234e-02 -1.38090789e-01]]

  [[ 1.42327416e+00 -1.07997835e+00 -5.48077583e-01 ... -7.84463704e-01
    -1.98164642e-01  3.20103407e-01]
   [ 1.42327416e+00 -1.07997835e+00 -5.48077583e-01 ... -7.84463704e-01
    -1.98164642e-01  3.20103407e-01]
   [ 1.35650969e+00 -1.04923391e+00 -4.85398561e-01 ... -7.64718473e-01
    -1.93478346e-01  3.09667230e-01]
   ...
   [ 1.14318311e+00 -9.81163085e-01  9.88265425e-02 ... -6.23952448e-01
    -1.15229025e-01 -9.39857513e-02]
   [ 1.16411352e+00 -1.03283930e+00  1.19960904e-01 ... -6.61158621e-01
    -9.09615234e-02 -1.38090789e-01]
   [ 1.16411352e+00 -1.03283930e+00  1.19960904e-01 ... -6.61158621e-01
    -9.09615234e-02 -1.38090789e-01]]

  [[ 1.40541399e+00 -1.04951870e+00 -5.57594895e-01 ... -7.44143546e-01
    -1.95452228e-01  3.17815602e-01]
   [ 1.40541399e+00 -1.04951870e+00 -5.57594895e-01 ... -7.44143546e-01
    -1.95452228e-01  3.17815602e-01]
   [ 1.34074366e+00 -1.01774752e+00 -4.94556099e-01 ... -7.25607157e-01
    -1.88863322e-01  3.05405170e-01]
   ...
   [ 1.12554991e+00 -9.64698136e-01  1.06743097e-01 ... -6.04096115e-01
    -1.13034025e-01 -8.57100785e-02]
   [ 1.14699209e+00 -1.01284969e+00  1.29422382e-01 ... -6.39558613e-01
    -8.87686089e-02 -1.26785845e-01]
   [ 1.14699209e+00 -1.01284969e+00  1.29422382e-01 ... -6.39558613e-01
    -8.87686089e-02 -1.26785845e-01]]

  ...

  [[ 1.58987486e+00 -1.31078374e+00 -4.44715381e-01 ... -7.83953071e-01
    -1.00075424e-01 -2.25890893e-03]
   [ 1.58987486e+00 -1.31078374e+00 -4.44715381e-01 ... -7.83953071e-01
    -1.00075424e-01 -2.25890893e-03]
   [ 1.53708267e+00 -1.28331733e+00 -3.87172729e-01 ... -8.06681931e-01
    -9.52222794e-02  5.73694706e-06]
   ...
   [ 1.30912733e+00 -9.70674694e-01 -8.90501589e-02 ... -1.09043491e+00
    -3.09141040e-01  3.89077850e-02]
   [ 1.32300663e+00 -9.64882731e-01 -9.51028466e-02 ... -1.11300480e+00
    -3.06090146e-01  1.45827802e-02]
   [ 1.32300663e+00 -9.64882731e-01 -9.51028466e-02 ... -1.11300480e+00
    -3.06090146e-01  1.45827802e-02]]

  [[ 1.62034464e+00 -1.33028090e+00 -4.00998175e-01 ... -8.16540956e-01
    -9.74346995e-02 -3.64237502e-02]
   [ 1.62034464e+00 -1.33028090e+00 -4.00998175e-01 ... -8.16540956e-01
    -9.74346995e-02 -3.64237502e-02]
   [ 1.56976604e+00 -1.30398726e+00 -3.46854150e-01 ... -8.42637777e-01
    -9.46996808e-02 -3.38588879e-02]
   ...
   [ 1.35981417e+00 -9.92664456e-01 -9.33551043e-02 ... -1.14417052e+00
    -3.35991263e-01  3.36253569e-02]
   [ 1.37550437e+00 -9.85970557e-01 -1.03232868e-01 ... -1.16642368e+00
    -3.34604144e-01  1.19127491e-02]
   [ 1.37550437e+00 -9.85970557e-01 -1.03232868e-01 ... -1.16642368e+00
    -3.34604144e-01  1.19127491e-02]]

  [[ 1.62034464e+00 -1.33028090e+00 -4.00998175e-01 ... -8.16540956e-01
    -9.74346995e-02 -3.64237502e-02]
   [ 1.62034464e+00 -1.33028090e+00 -4.00998175e-01 ... -8.16540956e-01
    -9.74346995e-02 -3.64237502e-02]
   [ 1.56976604e+00 -1.30398726e+00 -3.46854150e-01 ... -8.42637777e-01
    -9.46996808e-02 -3.38588879e-02]
   ...
   [ 1.35981417e+00 -9.92664456e-01 -9.33551043e-02 ... -1.14417052e+00
    -3.35991263e-01  3.36253569e-02]
   [ 1.37550437e+00 -9.85970557e-01 -1.03232868e-01 ... -1.16642368e+00
    -3.34604144e-01  1.19127491e-02]
   [ 1.37550437e+00 -9.85970557e-01 -1.03232868e-01 ... -1.16642368e+00
    -3.34604144e-01  1.19127491e-02]]]]
```