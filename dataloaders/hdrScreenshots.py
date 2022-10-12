from . import common
import numpy as np
from pathlib import Path
import pyroexr
import logging
import sys

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def load_unreal_hdr_screenshot(base_png_filepath: Path, expected_channels):
    channel_list = list()
    for channel_name, num_channels in expected_channels:
        channel_path = base_png_filepath.parent / f"{base_png_filepath.stem}_{channel_name}.exr"
        exr_data = pyroexr.load(str(channel_path))
        if num_channels == 3:
            image_tensor = np.stack([exr_data.channel('R'), exr_data.channel('G'), exr_data.channel('B')], axis=-1)
        elif num_channels == 1:
            image_tensor = np.expand_dims(exr_data.channel('R'), axis=-1)
        else:
            image_tensor = np.stack([channel for _, channel in exr_data.channels().items()])
        channel_list.append(image_tensor)

    all_channels = np.concatenate(channel_list, axis=-1)
    log.debug(all_channels.shape)
    return all_channels, base_png_filepath.stem


def get_unreal_hdr_screenshot_dataset(content_image_dir, expected_channels, shape, **kwargs):
    import tensorflow as tf

    screenshot_pngs = list(content_image_dir.glob('*.png'))
    num_samples = len(screenshot_pngs)
    if "seed" in kwargs:
        import random
        rng = random.Random(kwargs['seed'])
        rng.shuffle(screenshot_pngs)

    def load_hdr_screenshots_as_tensor():
        for screenshot_png in screenshot_pngs:
            channels, name = load_unreal_hdr_screenshot(screenshot_png, expected_channels)
            preprocessed_image: tf.Tensor = common.preprocess_numpy_image(channels, shape)
            yield preprocessed_image

    dataset = tf.data.Dataset.from_generator(load_hdr_screenshots_as_tensor,
                                             output_signature=tf.TensorSpec(shape, tf.float32),
                                             name="UnrealHdrScreenshots")
    dataset.num_samples = num_samples
    return dataset
