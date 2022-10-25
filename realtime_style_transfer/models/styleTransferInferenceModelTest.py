import shutil

from . import styleTransferInferenceModel
from . import stylePrediction
from . import styleTransfer
import unittest
from pathlib import Path
import tempfile
import tensorflow as tf


class StyleTransferInferenceModelTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_styles = 2

    def setUp(self):
        self.input_element_shape = (480, 960, 3)
        self.output_element_shape = (1920, 3840, 3)
        self.style_weights_shape = (1920, 3840, self.num_styles - 1)
        self.input_shape = {
            "style": (self.num_styles,) + self.input_element_shape,
            "style_weights": self.style_weights_shape,
            "content": self.input_element_shape
        }
        self.style_transfer_input_shape = {
            "style_weights": self.style_weights_shape,
            "content": self.input_element_shape
        }
        self.inferenceModel = styleTransferInferenceModel.make_style_transfer_inference_model(
            num_styles=self.num_styles,
            style_transfer_factory_func=lambda: styleTransfer.create_style_transfer_model(
                input_shape=self.input_element_shape,
                output_shape=self.output_element_shape,
                bottleneck_res_y=120,
                bottleneck_num_filters=128,
                num_styles=self.num_styles,
                name="StyleTransferTestModel"),
            style_predictor_factory_func=lambda num_style_params: stylePrediction.create_style_prediction_model(
                self.input_element_shape, stylePrediction.StyleFeatureExtractor.DUMMY, num_style_params),
            name="StyleTransferInferenceTestModel"
        )

        self.inferenceModel.inference.compile()

    def test_output_shape_matches(self):
        self.assertTupleEqual(self.inferenceModel.inference.output_shape, (None,) + self.output_element_shape)

    def test_inference(self):
        y = self.inferenceModel.inference.predict(
            {name: tf.zeros((1,) + shape) for name, shape in self.input_shape.items()})
        self.assertTupleEqual(y.shape, (1,) + self.output_element_shape)

    def test_save_transfer_model(self):
        temp_path = tempfile.mkdtemp()
        try:
            self.inferenceModel.transfer.save(temp_path)
        finally:
            shutil.rmtree(temp_path)

    def test_save_prediction_model(self):
        temp_path = tempfile.mkdtemp()
        try:
            self.inferenceModel.style_predictor.save(temp_path)
        finally:
            shutil.rmtree(temp_path)

    def test_load_model(self):
        test_checkpoint = Path(__file__).parent.parent.parent / "test" / \
                          "acceptance_data" / "inference.checkpoint" / "checkpoint"
        self.inferenceModel.inference.load_weights(test_checkpoint)
