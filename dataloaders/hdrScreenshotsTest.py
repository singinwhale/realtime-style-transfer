import numpy as np

from .hdrScreenshots import *
from . import hdrScreenshots
from tracing import logsetup
from pathlib import Path
import unittest

TEST_SCREENSHOT_DIR = Path(__file__).parent.parent / "test" / "test_screenshots"
TEST_SCREENSHOT_PNG_FILE = TEST_SCREENSHOT_DIR / "HighresScreenshot_2022.09.30-10.02.06.png"


class HdrScreenshotLoaderTests(unittest.TestCase):
    acceptance_data_dir = Path(__file__).parent.parent / "test" / "acceptance_data"

    def test_load_unreal_hdr_screenshot(self):
        screenshot_data: np.ndarray = load_unreal_hdr_screenshot(TEST_SCREENSHOT_PNG_FILE, [
            ("FinalImage", 3),
            ("BaseColor", 3),
            ("ShadowMask", 1),
            ("AmbientOcclusion", 1),
            ("Metallic", 1),
            ("Specular", 1),
            ("Roughness", 1),
            ("ViewNormal", 3),
            ("SceneDepth", 1),
            ("LightingModel", 3),
        ])

        self.assertEqual(screenshot_data.shape, (1080, 1920, 18))

    def test_get_unreal_hdr_screenshot_dataset(self):
        import tensorflow as tf
        expected_shape = (960, 1920, 7)
        dataset: tf.data.Dataset = get_debug_unreal_hdr_screenshot_dataset('validation', [
            ("FinalImage", 3),
            ("BaseColor", 3),
            ("ShadowMask", 1),
        ], expected_shape)

        num_items = 0
        for i, image in enumerate(dataset):
            num_items += 1
            self.assertEqual(image.shape, expected_shape, f"{i}")

        self.assertEqual(num_items, 5)