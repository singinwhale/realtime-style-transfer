import numpy as np

from . import wikiart
from pathlib import Path
import tensorflow as tf
import unittest


class WikiartDataloaderTests(unittest.TestCase):
    acceptance_data_dir = Path(__file__).parent.parent / "test" / "acceptance_data"

    def test_image_dataset_from_directory(self):
        shape = (960, 1920, 3)
        dataset = wikiart._image_dataset_from_directory(wikiart.style_debug_image_dir, shape)
        image: tf.Tensor
        for image in dataset:
            self.assertEqual(image.shape, shape)
            imagedata: np.ndarray = image.numpy()
            mean = imagedata.mean()
            self.assertLessEqual(mean, 1)
            self.assertGreaterEqual(mean, 0)
