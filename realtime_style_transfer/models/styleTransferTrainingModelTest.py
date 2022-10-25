from . import styleTransferTrainingModel
from . import stylePrediction
from . import styleLoss
from . import styleTransfer
import unittest
import tensorflow as tf


class StyleTransferTrainingModelTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_styles = 1

    def setUp(self):
        self.input_element_shape = (240, 480, 3)
        self.output_element_shape = (480, 960, 3)
        self.style_weights_shape = (480, 960, self.num_styles - 1)
        self.bottleneck_res_y = 30
        self.bottleneck_num_filters = 4
        self.ground_truth_shape = {
            'content': self.output_element_shape,
            'style': (self.num_styles,) + self.output_element_shape
        }
        self.input_shape = {
            "style": (self.num_styles,) + self.output_element_shape,
            "style_weights": self.style_weights_shape,
            "content": self.input_element_shape
        }
        self.trainingModel = styleTransferTrainingModel.make_style_transfer_training_model(
            style_transfer_factory_func=lambda: styleTransfer.create_style_transfer_model(
                self.input_shape['content'], self.output_element_shape, self.bottleneck_res_y,
                self.bottleneck_num_filters,
                num_styles=self.num_styles, name="StyleTransferTestModel"),
            style_predictor_factory_func=lambda num_style_params: stylePrediction.create_style_prediction_model(
                self.output_element_shape, stylePrediction.StyleFeatureExtractor.DUMMY, num_style_params),
            style_loss_func_factory_func=lambda: styleLoss.make_style_loss_function(
                styleLoss.StyleLossModelDummy(self.output_element_shape, name="StyleLossTestModel"),
                self.output_element_shape,
                self.num_styles),
            name="StyleTransferTrainingTestModel"
        )

        self.trainingModel.training.compile(run_eagerly=False)

    def test_training(self):
        def single_input_generator():
            for i in range(2):
                yield ({name: tf.zeros(shape) for name, shape in self.input_shape.items()},
                       {name: tf.zeros(shape) for name, shape in self.ground_truth_shape.items()})

        inputs = tf.data.Dataset.from_generator(
            single_input_generator,
            output_signature=({name: tf.TensorSpec(shape, name=name)
                               for name, shape in self.input_shape.items()},
                              {name: tf.TensorSpec(shape, name=name)
                               for name, shape in self.ground_truth_shape.items()})
        )
        inputs = inputs.batch(2)
        self.trainingModel.training.fit(inputs)
