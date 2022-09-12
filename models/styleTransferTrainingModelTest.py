import os
import shutil

from . import styleTransferTrainingModel
from . import stylePrediction
from . import styleLoss
from . import styleTransferFunctional
import unittest
import tensorflow as tf
import tempfile


class StyleTransferTraingingModelTest(unittest.TestCase):

    def setUp(self):
        self.input_element_shape = (480, 960, 3)
        self.input_shape = {
            "style": self.input_element_shape,
            "content": self.input_element_shape
        }
        self.trainingModel = styleTransferTrainingModel.StyleTransferTrainingModel(
            style_transfer_factory_func=lambda: styleTransferFunctional.create_style_transfer_model(
                self.input_element_shape,
                "StyleTransferTestModel"),
            style_predictor_factory_func=lambda num_style_params: stylePrediction.create_style_prediction_model(
                self.input_element_shape, stylePrediction.StyleFeatureExtractor.DUMMY, num_style_params),
            style_loss_func_factory_func=lambda: styleLoss.make_style_loss_function(
                styleLoss.StyleLossModelMobileNet(self.input_element_shape, name="StyleLossTestModel")),
            name="StyleTransferTrainingTestModel"
        )

        self.trainingModel.compile()

    def test_training(self):

        def single_input_generator():
            yield {"style": tf.zeros(self.input_element_shape),
                   "content": tf.zeros(self.input_element_shape)}

        inputs = tf.data.Dataset.from_generator(single_input_generator,
                                                output_signature={"style": tf.TensorSpec(self.input_element_shape),
                                                                  "content": tf.TensorSpec(self.input_element_shape)})
        inputs = inputs.batch(1)
        self.trainingModel.fit(inputs)

    def test_save_transfer_model(self):
        temp_path = tempfile.mkdtemp()
        try:
            self.trainingModel.style_transfer_model.save(temp_path)
        finally:
            shutil.rmtree(temp_path)

    def test_save_prediction_model(self):
        temp_path = tempfile.mkdtemp()
        try:
            self.trainingModel.style_predictor.save(temp_path)
        finally:
            shutil.rmtree(temp_path)
