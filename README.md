# lpips-tf2
* This is an improved version of [lpips-tf2.x](https://github.com/moono/lpips-tf2.x), with added support for non-square images, proper batching, and model checkpoint caching to avoid slow reloading.
* This is tensorflow 2.x conversion of official repo [LPIPS metric][Offical repo] (pytorch)
* Similar to [lpips-tensorflow][TF repo] except,
  * In this repo, network architecture is explicitly implemented rather than converting with ONNX.

## Limitation
* Currently only `model='net-lin', net='vgg'` is implemented; seems to only run on GPU.

## Example usage
* Create a symlink to `lpips_tensorflow.py` in your project.
* input image should be [0.0 ~ 255.0], float32, NHWC format

```python
import os
import numpy as np
import tensorflow as tf
from PIL import Image

from lpips_tensorflow import learned_perceptual_metric_model


def load_image(fn):
    image = Image.open(fn)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = tf.constant(image, dtype=tf.dtypes.float32)
    image = image[..., :3]  # Remove alpha channel
    return image    # [1, H, W, 3]

# official pytorch model metric value
# ex_ref.png <-> ex_p0.png: 0.569
# ex_ref.png <-> ex_p1.png: 0.422
image_fn1 = './imgs/ex_ref.png'
image_fn2 = './imgs/ex_p0.png'
image_fn3 = './imgs/ex_p1.png'

# images should be RGB normalized to [0.0, 255.0]
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
image3 = load_image(image_fn3)


image_hw = tuple((image1.shape)[-3:-1])
# First load the ckpt given the image (height, width); will be slow the first time.
lpips = learned_perceptual_metric_model(image_hw)

batch_ref = tf.concat([image1, image1], axis=0)
batch_inp = tf.concat([image2, image3], axis=0)
# Now compute LPIPS on two batches.
metric = lpips([batch_ref, batch_inp])

print(f'ref shape: {batch_ref.shape}')
print(f'inp shape: {batch_inp.shape}')
print(f'lpips metric shape: {metric.shape}')
print(f'ref <-> p0: {metric[0]:.3f}')
print(f'ref <-> p1: {metric[1]:.3f}')
```

### To reproduce same checkpoint files...
* Clone official repo [LPIPS metric][Offical repo]
* Place `./example_export_script/convert_to_tensorflow.py` and `./models/lpips_tensorflow.py` on root directory
* Run `convert_to_tensorflow.py`

[Offical repo]: https://github.com/richzhang/PerceptualSimilarity
[TF repo]: https://github.com/alexlee-gk/lpips-tensorflow
