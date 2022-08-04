import logging
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


def _load_images_from_directory(image_dir: Path, shape) -> typing.Generator[PIL.Image.Image, None, None]:
    import os
    log.debug(f"Searching for images in {image_dir}")
    for root, dirnames, filenames in os.walk(image_dir):
        log.debug(f"Found {len(filenames)} files in {root}")
        for filename in filenames:
            filepath = Path(root) / Path(filename)
            if filepath.suffix not in PIL.Image.EXTENSION:
                log.debug(f"Ignoring {filepath} because it has an invalid suffix")
                continue

            image = tf.keras.utils.load_img(path=filepath, interpolation="lanczos",
                                            color_mode='rgb' if shape[2] == 3 else 'rgba')
            image = _preprocess_image(image, (shape[1], shape[0], shape[2]))
            yield image


def image_dataset_from_directory(image_dir: Path, shape):
    def generate_image_tensors():
        for image in _load_images_from_directory(image_dir, shape):
            tensor: np.ndarray = tf.keras.utils.img_to_array(image, 'channels_last', dtype="float32")
            tensor = tensor / 255.0
            tensor: tf.Tensor = tf.convert_to_tensor(tensor)
            yield tensor

    dataset = tf.data.Dataset.from_generator(generate_image_tensors,
                                             output_signature=tf.TensorSpec((shape[0], shape[1], 3)))
    return dataset


def load_content_and_style_dataset_from_paths(content_image_dir, style_image_dir, shapes) -> \
        (tf.data.Dataset, tf.data.Dataset):
    def _create_content_and_style_dataset(subset):
        # todo use different content images
        content_dataset: tf.data.Dataset = image_dataset_from_directory(content_image_dir / subset, shapes['content'])
        style_dataset: tf.data.Dataset = image_dataset_from_directory(style_image_dir / subset, shapes['style'])
        content_dataset = content_dataset.shuffle(buffer_size=100, seed=217289)
        style_dataset = style_dataset.shuffle(buffer_size=100, seed=8828)

        def _pair_up_dataset():
            for i, (content, style) in enumerate(zip(content_dataset, style_dataset)):
                datapoint = {'content': content, 'style': style}
                yield datapoint

        paired_dataset = tf.data.Dataset.from_generator(_pair_up_dataset, output_signature={
            'content': tf.TensorSpec(shape=shapes['content'], dtype=tf.dtypes.float32, name=None),
            'style': tf.TensorSpec(shape=shapes['style'], dtype=tf.dtypes.float32, name=None)
        })
        return paired_dataset

    training_dataset = _create_content_and_style_dataset('training')
    validation_dataset = _create_content_and_style_dataset('validation')
    return training_dataset, validation_dataset


def get_single_sample_from_dataset(dataset: tf.data.Dataset):
    for datapoint in dataset.shuffle(buffer_size=100, seed=3780).unbatch().batch(1):
        return datapoint
    return None
