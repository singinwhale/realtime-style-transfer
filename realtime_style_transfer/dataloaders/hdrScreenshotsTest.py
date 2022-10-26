from .hdrScreenshots import *
from .common import content_hdr_debug_image_dir
from pathlib import Path
import unittest
import tensorflow as tf
from .commonTest import TEST_SCREENSHOT_PNG_FILE


class HdrScreenshotLoaderTests(unittest.TestCase):
    acceptance_data_dir = Path(__file__).parent.parent / "test" / "acceptance_data"

    def test_load_unreal_hdr_screenshot(self):
        screenshot_data, base_png_filename = load_unreal_hdr_screenshot(TEST_SCREENSHOT_PNG_FILE, [
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
        dataset: tf.data.Dataset = get_unreal_hdr_screenshot_dataset(content_hdr_debug_image_dir / "validation", [
            ("FinalImage", 3),
            ("BaseColor", 3),
            ("ShadowMask", 1),
        ], expected_shape)

        num_items = 0
        for i, image in enumerate(dataset):
            num_items += 1
            self.assertEqual(image.shape, expected_shape, f"{i}")

        self.assertEqual(num_items, 5)

    def test_get_unreal_hdr_screenshot_dataset_with_ground_truth(self):
        expected_input_shape = (480, 960, 7)
        expected_output_shape = (960, 1920, 3)
        dataset: tf.data.Dataset = get_unreal_hdr_screenshot_dataset(content_hdr_debug_image_dir / "validation", [
            ("FinalImage", 3),
            ("BaseColor", 3),
            ("ShadowMask", 1),
        ], expected_input_shape, output_shape=expected_output_shape)

        num_items = 0
        for i, image in enumerate(dataset):
            num_items += 1
            self.assertEqual(image[0].shape, expected_input_shape, f"input {i}")
            self.assertEqual(image[1].shape, expected_output_shape, f"ground_truth {i}")

        self.assertEqual(num_items, 5)
