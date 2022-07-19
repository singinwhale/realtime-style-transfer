import typing
from pathlib import Path

import logging

import tqdm.asyncio

log = logging.getLogger()

target_dir = Path(__file__).parent.parent.absolute() / "data" / "wikiart"
image_dir = target_dir / 'images' / 'painting'
manifest_filepath = target_dir / 'wikiart_scraped.csv'


def test_manifest_exists():
    return manifest_filepath.exists()


def test_images_exist():
    return len(list(image_dir.glob("*.jpg"))) == 124110


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
        os.mkdir(image_dir)

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


def get_dataset() -> (tf.data.Dataset, tf.data.Dataset):
    log.info("Loading WikiArt dataset...")
    if not test_complete():
        if not test_manifest_exists():
            download_manifest()
        if not test_images_exist():
            download_images()

    args = {
        'seed': 219793472,
        'image_size': (1920, 1080),
        'validation_split': 0.2
    }
    training_dataset = tf.keras.utils.image_dataset_from_directory(image_dir.parent, subset="training", **args)
    validation_dataset = tf.keras.utils.image_dataset_from_directory(image_dir.parent, subset="validation", **args)
    training_dataset.cache()
    validation_dataset.cache()
    return (training_dataset, validation_dataset)
