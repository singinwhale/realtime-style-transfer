import unittest
from pathlib import Path
from . import common
import tensorflow as tf

TEST_SCREENSHOT_DIR = Path(__file__).parent.parent.parent / "test" / "test_screenshots"
TEST_SCREENSHOT_PNG_FILE = TEST_SCREENSHOT_DIR / "HighresScreenshot_2022.10.19-17.36.24.png"


class CommonDataloadersTests(unittest.TestCase):
    def test_image_dataset_from_filepaths(self):
        input_shape = (480, 960, 3)
        dataset = common.image_dataset_from_filepaths([TEST_SCREENSHOT_PNG_FILE], input_shape)

        self.assertTrue(dataset.element_spec.is_compatible_with(tf.TensorSpec(input_shape)),
                        f"{dataset.element_spec} is not compatible with expected {input_shape}")
        element = dataset.get_single_element()
        self.assertTupleEqual(tuple(element.shape), input_shape)

    def test_image_dataset_from_filepaths_with_ground_truth(self):
        input_shape = (480, 960, 3)
        output_shape = (960, 1920, 3)
        dataset = common.image_dataset_from_filepaths([TEST_SCREENSHOT_PNG_FILE], input_shape,
                                                      output_shape=output_shape)

        self.assertTrue(dataset.element_spec[0].is_compatible_with(tf.TensorSpec(input_shape)),
                        f"dataset.element_spec[0]:{dataset.element_spec[0]} is not compatible with expected input_shape {input_shape}")
        self.assertTrue(dataset.element_spec[1].is_compatible_with(tf.TensorSpec(output_shape)),
                        f"dataset.element_spec[1]:{dataset.element_spec[1]} is not compatible with expected output_shape {output_shape}")
        element = dataset.get_single_element()
        self.assertTupleEqual(tuple(element[0].shape), input_shape)
        self.assertTupleEqual(tuple(element[1].shape), output_shape)
