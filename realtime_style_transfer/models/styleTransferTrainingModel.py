import logging

import tensorflow as tf
import typing
from .styleTransferInferenceModel import make_style_transfer_inference_model

log = logging.getLogger(__name__)


# noinspection PyAbstractClass
class StyleTransferTrainingModel(tf.keras.Model):

    def __init__(self,
                 style_loss_func: typing.Callable[[typing.Dict, tf.Tensor], typing.Dict],
                 loss_model: tf.keras.Model,
                 inference_model,
                 *args,
                 **kwargs):
        super().__init__(inference_model.input, inference_model.output, *args, **kwargs)

        self.style_loss_func = style_loss_func
        self.loss_model = loss_model
        self.inference_model = inference_model
        self.style_losses = {}

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        losses = self.style_loss_func(y_pred, y)
        self.style_losses = losses
        return losses['loss']

    def compute_metrics(self, x, y, y_pred, sample_weight):
        return self.style_losses

    def reset_metrics(self):
        self.style_losses = {}


def make_style_transfer_training_model(style_predictor_factory_func: typing.Callable[[int], tf.keras.Model],
                                       style_transfer_factory_func: typing.Callable[[], tf.keras.Model],
                                       style_loss_func_factory_func: typing.Callable[
                                           [], typing.Callable[[typing.Dict, tf.Tensor], typing.Dict]],
                                       name="StyleTransferTrainingModel"):
    inference_model = make_style_transfer_inference_model(
        num_styles=1,
        style_predictor_factory_func=style_predictor_factory_func,
        style_transfer_factory_func=style_transfer_factory_func,
        name=name
    )
    style_loss_func, loss_model = style_loss_func_factory_func()

    y_pred_input = {
        'content': loss_model.input['ground_truth']['content'],
        'style': loss_model.input['ground_truth']['style'],
    }
    training_model = StyleTransferTrainingModel(style_loss_func, loss_model, inference_model.inference)

    symbolic_losses = style_loss_func(inference_model.inference.outputs, y_pred_input)

    class StyleTransferModels:
        def __init__(self):
            self.loss_model = tf.keras.Model((inference_model.inputs, y_pred_input), symbolic_losses)
            self.training = training_model
            self.inference = inference_model.inference
            self.transfer = inference_model.transfer
            self.style_predictor = inference_model.style_predictor

    return StyleTransferModels()
