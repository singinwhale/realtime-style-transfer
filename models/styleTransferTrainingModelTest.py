import os
import shutil

from . import styleTransferTrainingModel
from . import stylePrediction
from . import styleLoss
from . import styleTransfer
import unittest
import tensorflow as tf
import tempfile


class StyleTransferTrainingModelTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_styles = 1

    def setUp(self):
        self.input_element_shape = (480, 960, 3)
        self.style_weights_shape = (480, 960, self.num_styles - 1)
        self.input_shape = {
            "style": (self.num_styles,) + self.input_element_shape,
            "style_weights": self.style_weights_shape,
            "content": self.input_element_shape
        }
        self.trainingModel = styleTransferTrainingModel.make_style_transfer_training_model(
            input_shape=self.input_shape,
            style_transfer_factory_func=lambda: styleTransfer.create_style_transfer_model(
                self.input_element_shape, num_styles=self.num_styles, name="StyleTransferTestModel"),
            style_predictor_factory_func=lambda num_style_params: stylePrediction.create_style_prediction_model(
                self.input_element_shape, stylePrediction.StyleFeatureExtractor.DUMMY, num_style_params),
            style_loss_func_factory_func=lambda: styleLoss.make_style_loss_function(
                styleLoss.StyleLossModelMobileNet(self.input_element_shape, name="StyleLossTestModel"),
                self.input_shape,
                self.input_element_shape),
            name="StyleTransferTrainingTestModel"
        )

        self.trainingModel.training.compile()

    def test_training(self):
        def single_input_generator():
            yield {
                "style": tf.zeros((self.num_styles,) + self.input_element_shape),
                "style_weights": tf.zeros(self.style_weights_shape),
                "content": tf.zeros(self.input_element_shape),
            }

        inputs = tf.data.Dataset.from_generator(single_input_generator,
                                                output_signature={
                                                    "style": tf.TensorSpec(
                                                        (self.num_styles,) + self.input_element_shape),
                                                    "style_weights": tf.TensorSpec(self.style_weights_shape),
                                                    "content": tf.TensorSpec(self.input_element_shape)}
                                                )
        inputs = inputs.batch(1)
        self.trainingModel.training.fit(inputs)
