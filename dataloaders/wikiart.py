import shutil
import typing
import unittest
from pathlib import Path

import PIL.Image
import numpy as np

import tqdm.asyncio
import logging

log = logging.getLogger(__name__)

target_dir = Path(__file__).parent.parent.absolute() / "data" / "wikiart"
image_dir = target_dir / 'images'
debug_image_dir = target_dir / 'debug_images'
manifest_filepath = target_dir / 'wikiart_scraped.csv'


def test_manifest_exists():
    return manifest_filepath.exists()


def test_images_exist(thorough=False):
    if not thorough:
        return (image_dir / "a6ab05c7e9f6e8810d3567c699f620b07600ae19.jpg").exists()
    import time
    start = time.time_ns()
    filecount = len(list(image_dir.iterdir()))
    log.info(f"took {(time.time_ns() - start) * 1e-6}ms to count {filecount} images")
    return filecount == 124110


def test_complete():
    return test_manifest_exists() and test_images_exist()


def download_manifest(force=False):
    if test_manifest_exists() and not force:
        return

    import kaggle

    kaggle.api.dataset_download_file(dataset='antoinegruson/-wikiart-all-images-120k-link',
                                     file_name='wikiart_scraped.csv',
                                     path=target_dir)

    import zipfile

    dataset_manifest_filename = str(manifest_filepath) + ".zip"

    print(f"Extracting {dataset_manifest_filename} to {target_dir} ...")
    archive = zipfile.ZipFile(mode="r", file=dataset_manifest_filename)
    archive.extractall(path=target_dir)
    archive.close()

    dataset_manifest_filename.unlink(missing_ok=False)
    assert manifest_filepath.exists(), f"{manifest_filepath} does not exist after downloading"


async def download_images_async(progress_hook: typing.Callable[[str, Path, int, int], None] = None) -> None:
    """
    :param progress_hook: ```
        def progress_hook(url, filename, index, total) -> None
        ```
    :return: None
    """
    import csv
    import httpx
    import hashlib

    import asyncio

    httpx_log = logging.getLogger('httpx._client')
    httpx_log.setLevel(logging.INFO)

    if not image_dir.exists():
        import os
        log.info(f"Creating imagedir at {image_dir}")
        image_dir.mkdir(parents=True)

    class AsyncState:
        total = 0
        queue = asyncio.Queue()

        limits = httpx.Limits(max_connections=10)
        client = httpx.AsyncClient(limits=limits)

        async def produce(self):
            with open(manifest_filepath, "r", encoding='utf-8') as manifest_file:
                manifest_csv_reader = csv.DictReader(manifest_file)
                for image_manifest in manifest_csv_reader:
                    await self.queue.put(image_manifest)
                    self.total = self.queue.qsize()

        async def consume_manifest(self, image_manifest):
            image_file_basename = hashlib.sha1(str(image_manifest).encode('utf-8'), usedforsecurity=False).hexdigest()
            image_target_path = (image_dir / image_file_basename).with_suffix('.jpg')
            image_url = image_manifest['Link']

            if progress_hook is not None:
                index = self.total - self.queue.qsize()
                progress_hook(image_url, image_target_path, index, self.total)

            if image_target_path.exists():
                log.debug(f"{image_target_path} already exists")
                return

            log.debug(f"Downloading {image_url} to {image_target_path}")
            image_response = await self.client.get(image_url)
            with open(image_target_path, "wb") as image_file:
                image_file.write(image_response.content)

        async def consumer_loop(self, ):
            while not self.queue.empty():
                manifest = await self.queue.get()
                try:
                    await self.consume_manifest(manifest)
                finally:
                    self.queue.task_done()

    state = AsyncState()
    consumers = []
    for i in range(20):
        consumers.append(asyncio.create_task(state.consumer_loop()))

    await state.produce()
    await asyncio.gather(*consumers)
    await state.queue.join()
    [consumer.cancel() for consumer in consumers]


def download_images():
    import asyncio
    import tqdm

    with tqdm.tqdm() as progress:
        def progress_hook(url, filename, index, total):
            if progress.total != total:
                progress.total = total
                progress.refresh()
            progress.update()

        asyncio.run(download_images_async(progress_hook=progress_hook))


import tensorflow as tf
import math


def get_dataset(shapes) -> (tf.data.Dataset, tf.data.Dataset):
    log.info("Loading WikiArt dataset...")
    init_dataset()

    return load_dataset_from_path(image_dir, shapes)


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

    for root, dirnames, filenames in os.walk(image_dir):
        for filename in filenames:
            filepath = Path(root) / Path(filename)
            if not filename.endswith(".jpg"):
                log.debug(f"Ignoring {filepath} because it has an invalid suffix")
                continue

            image = tf.keras.utils.load_img(path=filepath, interpolation="lanczos",
                                            color_mode='rgb' if shape[2] == 3 else 'rgba')
            image = _preprocess_image(image, (shape[1], shape[0], shape[2]))
            yield image


def _image_dataset_from_directory(image_dir: Path, shape):
    def generate_image_tensors():
        for image in _load_images_from_directory(image_dir, shape):
            tensor: np.ndarray = tf.keras.utils.img_to_array(image, 'channels_last', dtype="float32")
            tensor = tensor / 255.0
            tensor: tf.Tensor = tf.convert_to_tensor(tensor)
            yield tensor

    dataset = tf.data.Dataset.from_generator(generate_image_tensors,
                                             output_signature=tf.TensorSpec((shape[0], shape[1], 3)))
    return dataset


def load_dataset_from_path(image_dir, shapes) -> (tf.data.Dataset, tf.data.Dataset):
    def _create_content_and_style_dataset(subset):
        # todo use different content images
        content_dataset: tf.data.Dataset = _image_dataset_from_directory(image_dir / subset, shapes['content'])
        style_dataset: tf.data.Dataset = _image_dataset_from_directory(image_dir / subset, shapes['style'])
        content_dataset = content_dataset.shuffle(buffer_size=10, seed=217289)

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


def init_dataset():
    if not test_complete():
        if not test_manifest_exists():
            download_manifest()
        if not test_images_exist():
            download_images()


def get_dataset_debug(shapes) -> (tf.data.Dataset, tf.data.Dataset):
    log.info("Loading Debug WikiArt dataset...")
    init_dataset()
    training_dir = debug_image_dir / "training"
    validation_dir = debug_image_dir / "validation"
    for needed_image_dir in [debug_image_dir, training_dir, validation_dir]:
        if not needed_image_dir.exists():
            log.info(f"{needed_image_dir} does not exist. Creating it.")
            needed_image_dir.mkdir(parents=True)

    num_images = 100
    if len(list(debug_image_dir.iterdir())) != num_images:
        log.info(f"Copying debug images to {debug_image_dir}")
        images = image_dir.iterdir()
        for i in range(num_images):
            image = next(images)
            subset = "training" if i < 80 else "validation"
            debug_image_path = debug_image_dir / subset / image.name
            shutil.copyfile(image, debug_image_path)

    if shapes['content'][0] is None:
        training_dataset, validation_dataset = load_dataset_from_path(debug_image_dir,
                                                                      {key: shape[1:] for key, shape in shapes.items()})
        training_dataset = training_dataset.batch(9)
        validation_dataset = validation_dataset.batch(9)
    else:
        training_dataset, validation_dataset = load_dataset_from_path(debug_image_dir, shapes)

    training_dataset, validation_dataset = training_dataset.cache(), validation_dataset.cache()
    return training_dataset, validation_dataset
