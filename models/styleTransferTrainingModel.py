import logging

import tensorflow as tf
import typing
from types import SimpleNamespace

log = logging.getLogger(__name__)


def make_style_transfer_training_model(input_shape,
                                       style_predictor_factory_func: typing.Callable[[int], tf.keras.Model],
                                       style_transfer_factory_func: typing.Callable[[], tf.keras.Model],
                                       style_loss_func_factory_func: typing.Callable[
                                           [], typing.Callable[[typing.Dict, tf.Tensor], typing.Dict]],
                                       name="StyleTransferTrainingModel"):
    inputs = {'content': tf.keras.layers.Input(shape=input_shape['content']),
              'style': tf.keras.layers.Input(shape=input_shape['style'])}
    content_input, style_input = (inputs['content'], inputs['style'])

    style_transfer_model, num_style_parameters = style_transfer_factory_func()
    style_predictor = style_predictor_factory_func(num_style_parameters)
    style_params = style_predictor(style_input)

    stylized_image = style_transfer_model({
        "content": content_input,
        "style_params": style_params,
    })

    model = tf.keras.Model(inputs=inputs, outputs=stylized_image, name=name)
    losses = style_loss_func_factory_func()(inputs, stylized_image)
    model.add_loss(losses['loss'])
    for loss_name, loss_value in losses.items():
        if loss_name == "loss":
            continue
        model.add_metric(value=loss_value, name=loss_name)

    class StyleTransferModels:
        def __init__(self):
            self.training = model
            self.transfer = style_transfer_model
            self.style_predictor = style_predictor

    return StyleTransferModels()
