import os
import numpy as np
import tensorflow as tf
from PIL import Image

from models.lpips_tensorflow import learned_perceptual_metric_model


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
# image_fn1 = './imgs/ex_ref.png'
# image_fn2 = './imgs/ex_p0.png'
# image_fn3 = './imgs/ex_p1.png'

# image_fn1 = './imgs/kitten.png'
# image_fn2 = './imgs/kitten.png'
# image_fn3 = './imgs/puppy.png'

image_fn1 = '/extra/ucibdl0/shared/data/kodak/kodim01.png'
image_fn2 = '/extra/ucibdl0/shared/data/kodak/kodim01.png'
image_fn3 = '/extra/ucibdl0/shared/data/kodak/kodim03.png'

# images should be RGB normalized to [0.0, 255.0]
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
image3 = load_image(image_fn3)

from time import perf_counter
prev = perf_counter()
for i in range(10):
    cur = perf_counter()
    print(f'Running {i}, took {cur-prev} s')
    prev = cur

    image_hw = tuple((image1.shape)[-3:-1])
    model_dir = './models'
    lpips = learned_perceptual_metric_model(image_hw)

    batch_ref = tf.concat([image1, image1], axis=0)
    batch_inp = tf.concat([image2, image3], axis=0)
    metric = lpips([batch_ref, batch_inp])
print(f'ref shape: {batch_ref.shape}')
print(f'inp shape: {batch_inp.shape}')
print(f'lpips metric shape: {metric.shape}')
print(f'ref <-> p0: {metric[0]:.3f}')
print(f'ref <-> p1: {metric[1]:.3f}')
