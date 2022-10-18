import logging
import random
from pathlib import Path
import math
import PIL.Image
import typing
import os
import tensorflow as tf
import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
content_target_dir = Path(__file__).parent.parent.parent.absolute() / "data" / "screenshots"
style_target_dir = Path(__file__).parent.parent.parent.absolute() / "data" / "wikiart"
style_image_dir = style_target_dir / 'images'
content_image_dir = content_target_dir / 'images'
content_hdr_image_dir = content_target_dir / 'hdr_images'
style_debug_image_dir = style_target_dir / 'debug_images'
content_debug_image_dir = content_target_dir / 'debug_images'
content_hdr_debug_image_dir = content_target_dir / 'debug_hdr_images'


def _preprocess_pillow_image(image: PIL.Image, shape):
    aspect_ratio_image = image.size[0] / image.size[1]
    aspect_ratio_target = shape[0] / shape[1]
    should_scale_to_target_y = aspect_ratio_image > aspect_ratio_target

    new_size = (math.ceil(shape[1] * aspect_ratio_image), shape[1]) if should_scale_to_target_y else (
        shape[0], math.ceil(shape[0] / aspect_ratio_image))
    image = image.resize(new_size)

    width, height = image.size

    left = (width - shape[0]) / 2
    top = (height - shape[1]) / 2
    right = (width + shape[0]) / 2
    bottom = (height + shape[1]) / 2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    return image


def preprocess_numpy_image(image: np.ndarray, shape) -> tf.Tensor:
    input_image = image
    aspect_ratio_image = image.shape[0] / image.shape[1]
    aspect_ratio_target = shape[0] / shape[1]
    should_scale_to_target_y = aspect_ratio_image > aspect_ratio_target

    new_size = (math.ceil(shape[1] * aspect_ratio_image), shape[1]) if should_scale_to_target_y else (
        shape[0], math.ceil(shape[0] / aspect_ratio_image))
    image = tf.image.resize(image, new_size)

    # Crop the center of the image
    final_image = tf.image.resize_with_crop_or_pad(image, shape[0], shape[1])
    log.debug(f"{input_image.shape}| {image.shape} -> {shape}: {final_image.shape}")
    return final_image


def _load_image_from_file(filepath, shape):
    assert len(shape) == 3, "this function does not take care of special shapes"
    image = tf.keras.utils.load_img(path=filepath, interpolation="lanczos",
                                    color_mode='grayscale' if shape[2] == 1 else 'rgb' if shape[2] == 3 else 'rgba')
    image = _preprocess_pillow_image(image, (shape[1], shape[0], shape[2]))
    return image


def _load_images_from_directory(image_dir: Path, shape, **kwargs) -> typing.Generator[PIL.Image.Image, None, None]:
    import os
    log.debug(f"Searching for images in {image_dir}")
    rng = None
    if 'seed' in kwargs:
        rng = random.Random(kwargs['seed'])

    for root, dirnames, filenames in os.walk(image_dir):
        log.debug(f"Found {len(filenames)} files in {root}")

        if rng:
            rng.shuffle(filenames)

        for filename in filenames:
            filepath = Path(root) / Path(filename)
            if filepath.suffix not in PIL.Image.EXTENSION:
                log.debug(f"Ignoring {filepath} because it has an invalid suffix")
                continue

            image = _load_image_from_file(filepath, shape[-3:])
            yield image


def _image_to_tensor(image, shape) -> tf.Tensor:
    tensor: np.ndarray = tf.keras.utils.img_to_array(image, 'channels_last', dtype="float32")
    tensor = tensor / 255.0
    tensor: tf.Tensor = tf.convert_to_tensor(tensor)
    tensor = tf.reshape(tensor, shape)
    return tensor


def image_dataset_from_directory(image_dir: Path, shape, **kwargs):
    def generate_image_tensors():
        for image in _load_images_from_directory(image_dir, shape, **kwargs):
            tensor = _image_to_tensor(image, shape)
            yield tensor

    dataset = tf.data.Dataset.from_generator(generate_image_tensors,
                                             output_signature=tf.TensorSpec(shape))
    return dataset


def image_dataset_from_filepaths(filepaths, shape) -> tf.data.Dataset:
    def generate_image_tensors():
        for imagepath in filepaths:
            try:
                image = _load_image_from_file(imagepath, shape[-3:])
                tensor = _image_to_tensor(image, shape)
                yield tensor
            except Exception as e:
                log.warning(f"Could not read image {imagepath}: {e}")

    dataset = tf.data.Dataset.from_generator(generate_image_tensors,
                                             output_signature=tf.TensorSpec(shape))
    dataset.num_samples = len(filepaths)
    return dataset


def pair_up_content_and_style_datasets(content_dataset, style_dataset, shapes) -> tf.data.Dataset:
    def _pair_up_dataset():
        for i, (content, style) in enumerate(zip(content_dataset, style_dataset)):
            datapoint = {'content': content, 'style_weights': tf.zeros(shapes['style_weights']), 'style': style}
            yield datapoint

    paired_dataset = tf.data.Dataset.from_generator(_pair_up_dataset, output_signature={
        'content': tf.TensorSpec(shape=shapes['content'], dtype=tf.dtypes.float32, name="content_data"),
        'style_weights': tf.TensorSpec(shape=shapes['style_weights'], dtype=tf.dtypes.float32,
                                       name="style_weights_data"),
        'style': tf.TensorSpec(shape=shapes['style'], dtype=tf.dtypes.float32, name="style_data")
    })
    paired_dataset.num_samples = min(content_dataset.num_samples, style_dataset.num_samples)
    return paired_dataset


def load_training_and_validation_dataset_from_directory(image_dir, shape, **kwargs):
    def _create_content_and_style_dataset(subset):
        if 'channels' in kwargs:
            from .hdrScreenshots import get_unreal_hdr_screenshot_dataset
            dataset = get_unreal_hdr_screenshot_dataset(image_dir / subset, kwargs['channels'], shape, **kwargs)
        else:
            dataset: tf.data.Dataset = image_dataset_from_directory(image_dir / subset, shape, **kwargs)
        return dataset

    training_dataset = _create_content_and_style_dataset('training')
    validation_dataset = _create_content_and_style_dataset('validation')
    return training_dataset, validation_dataset


def load_content_and_style_dataset_from_paths(content_image_directory, style_image_directory, shapes, **kwargs) -> \
        (tf.data.Dataset, tf.data.Dataset):
    def _create_content_and_style_dataset(subset):

        if 'channels' in kwargs:
            from .hdrScreenshots import get_unreal_hdr_screenshot_dataset
            content_dataset = get_unreal_hdr_screenshot_dataset(content_image_directory / subset,
                                                                kwargs['channels'],
                                                                shapes['content'], **kwargs)
        else:
            content_dataset: tf.data.Dataset = image_dataset_from_directory(content_image_directory / subset,
                                                                            shapes['content'],
                                                                            **kwargs)
        style_dataset: tf.data.Dataset = image_dataset_from_directory(style_image_directory / subset, shapes['style'],
                                                                      **kwargs)
        return pair_up_content_and_style_datasets(content_dataset, style_dataset, shapes)

    training_dataset = _create_content_and_style_dataset('training')
    validation_dataset = _create_content_and_style_dataset('validation')
    return training_dataset, validation_dataset


def get_single_sample_from_dataset(dataset: tf.data.Dataset):
    for datapoint in dataset.unbatch().batch(1):
        return datapoint
    return None