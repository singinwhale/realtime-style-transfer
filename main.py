# -*- coding: utf-8 -*-


## Setup

### Import and configure modules

import os
import tensorflow as tf

# Load compressed models from tensorflow_hub
from models.styleLoss import StyleLossModelVGG, StyleLossModelEfficientNet

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

from renderers.image import tensor_to_image

content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg',
                                       'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('kandinsky5.jpg',
                                     'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

from dataloaders.imageloader import load_img_and_resize

content_image = load_img_and_resize(content_path)
style_image = load_img_and_resize(style_path)

from renderers.matplotlib import imshow

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')


"""Choose intermediate layers from the network to represent the style and content of the image:

"""

# extractor = StyleLossModelVGG()
extractor = StyleLossModelEfficientNet()
num_style_layers = extractor.num_style_layers

results = extractor(tf.constant(content_image))

print('Styles:')
for name, output in sorted(results['style'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())
    print()

print("Contents:")
for name, output in sorted(results['content'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())

"""## Run gradient descent

With this style and content extractor, you can now implement the style transfer algorithm. 
Do this by calculating the mean square error for your image's output relative to each target, 
then take the weighted sum of these losses.

Set your style and content target values:
"""

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

"""Define a `tf.Variable` to contain the image to optimize. To make this quick, initialize it with the content image (the `tf.Variable` must be the same shape as the content image):"""

image = tf.Variable(content_image)

"""Since this is a float image, define a function to keep the pixel values between 0 and 1:"""


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


"""Create an optimizer. The paper recommends LBFGS, but Adam works okay, too:"""

opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

"""To optimize this, use a weighted combination of the two losses to get the total loss:"""

style_weight = 1e4
content_weight = 1e-2


def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    per_layer_style_losses = [tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in
                              style_outputs.keys()]

    style_loss = tf.add_n(per_layer_style_losses)
    style_loss *= style_weight / num_style_layers

    per_layer_content_losses = [tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in
                                content_outputs.keys()]

    content_loss = tf.add_n(per_layer_content_losses)
    content_loss *= content_weight / num_style_layers
    loss = style_loss + content_loss
    return loss


"""Use `tf.GradientTape` to update the image."""


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


"""## Re-run the optimization

Choose a weight for the `total_variation_loss`:
"""

total_variation_weight = 30

"""Now include it in the `train_step` function:"""


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))


"""And run the optimization:"""
filename_template = 'logs/test/stylized-image_{:03}.png'
import time

start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(image)
        print(".", end='', flush=True)
    #display.clear_output(wait=True)
    #display.display(tensor_to_image(image))
    tensor_to_image(image).save(filename_template.format(step))
    print("Train step: {}".format(step))

end = time.time()
print("Total time: {:.1f}".format(end - start))

"""Finally, save the result:"""


file_name = filename_template.format(999)
tensor_to_image(image).save(file_name)
