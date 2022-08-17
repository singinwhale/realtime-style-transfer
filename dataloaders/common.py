import logging
import random
from pathlib import Path
import math
import PIL.Image
import typing
import os
import tensorflow as tf
import numpy as np

log = logging.getLogger()


def _preprocess_image(image, shape):
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


def _load_image_from_file(filepath, shape):
    image = tf.keras.utils.load_img(path=filepath, interpolation="lanczos",
                                    color_mode='rgb' if shape[2] == 3 else 'rgba')
    image = _preprocess_image(image, (shape[1], shape[0], shape[2]))
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

            image = _load_image_from_file(filepath, shape)
            yield image


def _image_to_tensor(image) -> tf.Tensor:
    tensor: np.ndarray = tf.keras.utils.img_to_array(image, 'channels_last', dtype="float32")
    tensor = tensor / 255.0
    tensor: tf.Tensor = tf.convert_to_tensor(tensor)
    return tensor


def image_dataset_from_directory(image_dir: Path, shape, **kwargs):
    def generate_image_tensors():
        for image in _load_images_from_directory(image_dir, shape, **kwargs):
            yield _image_to_tensor(image)

    dataset = tf.data.Dataset.from_generator(generate_image_tensors,
                                             output_signature=tf.TensorSpec((shape[0], shape[1], 3)))
    return dataset


def image_dataset_from_filepaths(filepaths, shape):
    def generate_image_tensors():
        for imagepath in filepaths:
            try:
                image = _load_image_from_file(imagepath, shape)
                yield _image_to_tensor(image)
            except Exception as e:
                log.warning(f"Could not read image {imagepath}: {e}")

    dataset = tf.data.Dataset.from_generator(generate_image_tensors,
                                             output_signature=tf.TensorSpec((shape[0], shape[1], 3)))
    return dataset


def pair_up_content_and_style_datasets(content_dataset, style_dataset, shapes):
    def _pair_up_dataset():
        for i, (content, style) in enumerate(zip(content_dataset, style_dataset)):
            datapoint = {'content': content, 'style': style}
            yield datapoint

    paired_dataset = tf.data.Dataset.from_generator(_pair_up_dataset, output_signature={
        'content': tf.TensorSpec(shape=shapes['content'], dtype=tf.dtypes.float32, name=None),
        'style': tf.TensorSpec(shape=shapes['style'], dtype=tf.dtypes.float32, name=None)
    })
    return paired_dataset


def load_training_and_validation_dataset_from_directory(image_dir, shape, **kwargs):
    def _create_content_and_style_dataset(subset):
        dataset: tf.data.Dataset = image_dataset_from_directory(image_dir / subset, shape, **kwargs)
        return dataset

    training_dataset = _create_content_and_style_dataset('training')
    validation_dataset = _create_content_and_style_dataset('validation')
    return training_dataset, validation_dataset


def load_content_and_style_dataset_from_paths(content_image_dir, style_image_dir, shapes, **kwargs) -> \
        (tf.data.Dataset, tf.data.Dataset):
    def _create_content_and_style_dataset(subset):
        content_dataset: tf.data.Dataset = image_dataset_from_directory(content_image_dir / subset, shapes['content'], **kwargs)
        style_dataset: tf.data.Dataset = image_dataset_from_directory(style_image_dir / subset, shapes['style'], **kwargs)
        return pair_up_content_and_style_datasets(content_dataset, style_dataset, shapes)

    training_dataset = _create_content_and_style_dataset('training')
    validation_dataset = _create_content_and_style_dataset('validation')
    return training_dataset, validation_dataset


def get_single_sample_from_dataset(dataset: tf.data.Dataset):
    for datapoint in dataset.unbatch().batch(1):
        return datapoint
    return None
