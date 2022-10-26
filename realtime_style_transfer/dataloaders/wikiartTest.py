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

    def test_get_dataset(self):
        input_shape = {
            'content': (480, 960, 21),
            'style_weights': (480, 960, 0),
            'style': (960, 1920, 3),
        }
        output_shape = (960, 1920, 3)
        ground_truth_shape = {
            'content': output_shape,
            'style': output_shape,
        }

        batch_size = 2
        training_dataset, validation_dataset = wikiart.get_hdr_dataset(input_shape,
                                                                       output_shape=output_shape,
                                                                       batch_size=batch_size,
                                                                       seed=34789082)

        for dataset_name, dataset in [("training_dataset", training_dataset),
                                      ("validation_dataset", validation_dataset)]:
            with self.subTest(dataset_name):
                for i, shapes in enumerate([input_shape, ground_truth_shape]):
                    for shape_name, shape in shapes.items():
                        specd_shape = tuple(dataset.element_spec[i][shape_name].shape)
                        expected_shape = (None,) + shape
                        self.assertTupleEqual(specd_shape, expected_shape,
                                              f"output {i} {shape_name} {specd_shape} does not match expected shape {expected_shape}")

                batch = next(iter(dataset))

                for i, shapes in enumerate([input_shape, ground_truth_shape]):
                    for shape_name, shape in shapes.items():
                        element_shape = tuple(batch[i][shape_name].shape)
                        expected_shape = (batch_size,) + shape
                        self.assertTupleEqual(element_shape, expected_shape,
                                              f"output {i} {shape_name} {specd_shape} does not match expected shape {expected_shape}")
