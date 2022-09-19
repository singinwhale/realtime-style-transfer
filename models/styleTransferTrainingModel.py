import logging

import tensorflow as tf
import typing
from .styleTransferInferenceModel import make_style_transfer_inference_model

log = logging.getLogger(__name__)


def make_style_transfer_training_model(input_shape,
                                       style_predictor_factory_func: typing.Callable[[int], tf.keras.Model],
                                       style_transfer_factory_func: typing.Callable[[], tf.keras.Model],
                                       style_loss_func_factory_func: typing.Callable[
                                           [], typing.Callable[[typing.Dict, tf.Tensor], typing.Dict]],
                                       name="StyleTransferTrainingModel"):
    inference_model = make_style_transfer_inference_model(
        input_shape,
        style_predictor_factory_func=style_predictor_factory_func,
        style_transfer_factory_func=style_transfer_factory_func,
        name=name
    )

    losses = style_loss_func_factory_func()(inference_model.inputs, inference_model.inference.output)
    inference_model.inference.add_loss(losses['loss'])
    for loss_name, loss_value in losses.items():
        if loss_name == "loss":
            continue
        inference_model.inference.add_metric(value=loss_value, name=loss_name)

    class StyleTransferModels:
        def __init__(self):
            self.training = inference_model.inference
            self.transfer = inference_model.transfer
            self.style_predictor = inference_model.style_predictor

    return StyleTransferModels()
