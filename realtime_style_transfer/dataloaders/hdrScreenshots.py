import PIL.Image

from . import common
import numpy as np
from pathlib import Path
from .common import _load_image_from_file, _image_to_tensor
import pyroexr
import logging

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
    return all_channels, base_png_filepath


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
            try:
                channels, screenshot_path = load_unreal_hdr_screenshot(screenshot_png, expected_channels)
                preprocessed_image: tf.Tensor = common.preprocess_numpy_image(channels, shape)
                if 'output_shape' in kwargs:
                    output_shape = kwargs['output_shape']
                    ground_truth_image = _load_image_from_file(screenshot_path, output_shape[-3:])
                    ground_truth_tensor = _image_to_tensor(ground_truth_image, output_shape)
                    yield preprocessed_image, ground_truth_tensor
                else:
                    yield preprocessed_image
            except Exception as e:
                log.warning(f"Skipping f{screenshot_png} due to an error: {e}")

    if 'output_shape' in kwargs:
        output_signature = (tf.TensorSpec(shape, tf.float32, name="content_data"),
                            tf.TensorSpec(kwargs['output_shape'], tf.float32, name="truth_data"))
    else:
        output_signature = tf.TensorSpec(shape, tf.float32, name="content_data")

    dataset = tf.data.Dataset.from_generator(load_hdr_screenshots_as_tensor,
                                             output_signature=output_signature,
                                             name="UnrealHdrScreenshots")
    dataset.num_samples = num_samples
    return dataset
