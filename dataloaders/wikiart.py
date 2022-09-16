import math
import random
import shutil
import typing
import unittest
from pathlib import Path
import csv

import tqdm

from . import common
import logging
import hashlib

log = logging.getLogger(__name__)

content_target_dir = Path(__file__).parent.parent.absolute() / "data" / "screenshots"
style_target_dir = Path(__file__).parent.parent.absolute() / "data" / "wikiart"
style_image_dir = style_target_dir / 'images'
content_image_dir = content_target_dir / 'images'
style_debug_image_dir = style_target_dir / 'debug_images'
content_debug_image_dir = content_target_dir / 'debug_images'
manifest_filepath = style_target_dir / 'wikiart_scraped.csv'

BLACKLISTED_IMAGE_HASHES = [
    "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0",
    "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0",
    "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0",
    "0",
]

NUM_WIKIART_IMAGES = 124170


def test_manifest_exists():
    return manifest_filepath.exists()


def test_images_exist(thorough=False):
    if not thorough:
        return (style_image_dir / "a6ab05c7e9f6e8810d3567c699f620b07600ae19.jpg").exists()
    import time
    start = time.time_ns()
    filecount = len(list(style_image_dir.iterdir()))
    log.info(f"took {(time.time_ns() - start) * 1e-6}ms to count {filecount} images")
    return filecount == NUM_WIKIART_IMAGES - len(BLACKLISTED_IMAGE_HASHES)


def test_complete():
    return test_manifest_exists() and test_images_exist()


def download_manifest(force=False):
    if test_manifest_exists() and not force:
        return

    import kaggle

    kaggle.api.dataset_download_file(dataset='antoinegruson/-wikiart-all-images-120k-link',
                                     file_name='wikiart_scraped.csv',
                                     path=style_target_dir)

    import zipfile

    dataset_manifest_filename = str(manifest_filepath) + ".zip"

    print(f"Extracting {dataset_manifest_filename} to {style_target_dir} ...")
    archive = zipfile.ZipFile(mode="r", file=dataset_manifest_filename)
    archive.extractall(path=style_target_dir)
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
    import httpx

    import asyncio

    httpx_log = logging.getLogger('httpx._client')
    httpx_log.setLevel(logging.INFO)

    if not style_image_dir.exists():
        import os
        log.info(f"Creating imagedir at {style_image_dir}")
        style_image_dir.mkdir(parents=True)

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
            image_target_path = image_manifest_to_filepath(image_manifest)
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


def get_dataset(shapes, batch_size, **kwargs) -> (tf.data.Dataset, tf.data.Dataset):
    log.info("Loading WikiArt dataset...")
    init_dataset()

    filepaths = sorted(map(lambda image_manifest: image_manifest_to_filepath(image_manifest), _read_dataset_manifest()))
    filepaths = list(filter(lambda path: path.stem not in BLACKLISTED_IMAGE_HASHES, filepaths))
    if 'seed' in kwargs:
        rng = random.Random(x=kwargs['seed'])
        rng.shuffle(filepaths)

    validation_split_index = math.floor(len(filepaths) * 0.8)
    shape_without_batches = shapes

    training_style_dataset = common.image_dataset_from_filepaths(filepaths[:validation_split_index],
                                                                 shape_without_batches['style'])
    validation_style_dataset = common.image_dataset_from_filepaths(filepaths[validation_split_index:],
                                                                   shape_without_batches['style'])

    training_content_dataset, validation_content_dataset = \
        common.load_training_and_validation_dataset_from_directory(content_image_dir, shape_without_batches['content'],
                                                                   **kwargs)

    training_dataset = common.pair_up_content_and_style_datasets(content_dataset=training_content_dataset,
                                                                 style_dataset=training_style_dataset,
                                                                 shapes=shape_without_batches)
    validation_dataset = common.pair_up_content_and_style_datasets(content_dataset=validation_content_dataset,
                                                                   style_dataset=validation_style_dataset,
                                                                   shapes=shape_without_batches)

    if batch_size is not None:
        training_dataset = training_dataset.batch(batch_size)
        validation_dataset = validation_dataset.batch(batch_size)

    if 'cache_dir' in kwargs:
        cache_dir = kwargs['cache_dir']
        training_dataset = training_dataset.cache(filename=str(cache_dir / "wikiart_training_dataset"))
        validation_dataset = validation_dataset.cache(filename=str(cache_dir / "wikiart_validation_dataset"))

        for name, dataset in {"training_dataset": training_dataset, "validation_dataset": validation_dataset}.items():
            if not (cache_dir / f"wikiart_{name}.index").exists():
                log.info(f"Caching {name} into {cache_dir}. This could take a while")
                # immediately cache everything
                for _ in tqdm.tqdm(iterable=dataset, desc=name, file=sys.stdout):
                    pass

    return training_dataset, validation_dataset


def init_dataset():
    if not test_complete():
        if not test_manifest_exists():
            download_manifest()
        if not test_images_exist():
            download_images()


def get_dataset_debug(shapes, batch_size=1, **kwargs) -> (tf.data.Dataset, tf.data.Dataset):
    log.info("Loading Debug WikiArt dataset...")
    init_dataset()
    training_dir = style_debug_image_dir / "training"
    validation_dir = style_debug_image_dir / "validation"
    for needed_image_dir in [style_debug_image_dir, training_dir, validation_dir]:
        if not needed_image_dir.exists():
            log.info(f"{needed_image_dir} does not exist. Creating it.")
            needed_image_dir.mkdir(parents=True)

    num_images = 100
    if len(list(style_debug_image_dir.iterdir())) != num_images:
        log.info(f"Copying debug images to {style_debug_image_dir}")
        images = style_image_dir.iterdir()
        for i in range(num_images):
            image = next(images)
            subset = "training" if i < 80 else "validation"
            debug_image_path = style_debug_image_dir / subset / image.name
            shutil.copyfile(image, debug_image_path)

    if batch_size is not None:
        training_dataset, validation_dataset = \
            common.load_content_and_style_dataset_from_paths(content_debug_image_dir,
                                                             style_debug_image_dir,
                                                             shapes,
                                                             **kwargs)
        training_dataset = training_dataset.batch(batch_size)
        validation_dataset = validation_dataset.batch(batch_size)
    else:
        training_dataset, validation_dataset = common.load_content_and_style_dataset_from_paths(content_debug_image_dir,
                                                                                                style_debug_image_dir,
                                                                                                shapes,
                                                                                                **kwargs)
    if "cache_dir" in kwargs:
        cache_dir = kwargs["cache_dir"]
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        training_dataset = training_dataset.cache(filename=str(cache_path / "debug_training_dataset"))
        validation_dataset = validation_dataset.cache(filename=str(cache_path / "debug_validation_dataset"))
        log.info(f"Caching datasets into {cache_dir}. This could take a while")
        for name, dataset in {"training_dataset": training_dataset, "validation_dataset": validation_dataset}.items():
            # immediately cache everything
            for _ in tqdm.tqdm(iterable=dataset, desc=name):
                pass

    return training_dataset, validation_dataset


def _read_dataset_manifest():
    with open(manifest_filepath, "r", encoding='utf-8') as manifest_file:
        manifest_csv_reader = csv.DictReader(manifest_file)
        for image_manifest in manifest_csv_reader:
            yield image_manifest


def image_manifest_to_filepath(image_manifest):
    image_file_basename = hashlib.sha1(str(image_manifest).encode('utf-8'), usedforsecurity=False).hexdigest()
    image_target_path = (style_image_dir / image_file_basename).with_suffix('.jpg')
    return image_target_path

