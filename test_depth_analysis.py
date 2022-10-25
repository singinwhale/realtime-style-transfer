import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_hub as hub
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from realtime_style_transfer.dataloaders import *

project_dir = Path(__file__).parent

test_screenshot_path = project_dir / "test" / "test_screenshots"

# images = common.image_dataset_from_filepaths([test_screenshot_path], (480, 960, 3)).batch(1)
images = hdrScreenshots.get_unreal_hdr_screenshot_dataset(test_screenshot_path, [('FinalImage', 3), ('SceneDepth', 1)], (480, 960, 4)).batch(1)

input = tf.keras.Input(shape=(480, 960, 3))

resizing_layer = tf.keras.layers.Resizing(384, 384)
resized = resizing_layer(input)

midas_model = hub.KerasLayer("https://tfhub.dev/intel/midas/v2/2", tags=['serve'],
                             signature='serving_default',
                             input_shape=(3, 384, 384), output_shape=(384, 384))

output = midas_model(tf.transpose(resized, [0, 3, 1, 2]))
model = tf.keras.Model(input, output)


def normalize_depth(d):
    minimum = tf.reduce_min(d)
    maximum = tfp.stats.percentile(d, 98)
    d = tf.where(d < maximum, d, tf.zeros_like(d))
    maximum = tf.reduce_max(d)
    return (d - minimum) / (maximum - minimum)
    t = tfp.stats.percentile(d, 50)
    s = tf.reduce_mean(tf.abs(d - t))
    return (d - t) / s


channels = next(iter(images))
image = channels[..., 0:3]
depth_map = model(image)
fig, (plt1, plt2) = plt.subplots(1, 2, sharey=True, sharex=True)
depth_map_image = normalize_depth(np.squeeze(depth_map.numpy()))
plt1.imshow(depth_map_image)
ground_truth_depth = channels[..., 3]

# ground_truth_depth = tf.where(ground_truth_depth > 0, ground_truth_depth, tf.ones_like(ground_truth_depth))

ground_truth_depth_resized = np.squeeze(resizing_layer(tf.expand_dims(ground_truth_depth, -1)))
ground_truth_depth_normalized = np.squeeze(1 - normalize_depth(ground_truth_depth_resized))
plt2.imshow(ground_truth_depth_normalized)
fig.colorbar(mappable=plt1.pcolor(depth_map_image), ax=plt1)
fig.colorbar(mappable=plt2.pcolor(ground_truth_depth_normalized), ax=plt2)
plt.show()
