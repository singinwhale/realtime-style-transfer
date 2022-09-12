
from .wikiart import *
from . import wikiart
from pathlib import Path
import unittest


class WikiartDataloaderTests(unittest.TestCase):
    acceptance_data_dir = Path(__file__).parent.parent / "test" / "acceptance_data"

    def test_image_manifest_to_filepath(self):
        filepaths = list(sorted(map(lambda image_manifest: image_manifest_to_filepath(image_manifest),
                                    wikiart._read_dataset_manifest())))
        self.assertEqual(len(filepaths), NUM_WIKIART_IMAGES)
