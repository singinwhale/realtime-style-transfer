import logging

import tensorflow as tf
import typing

log = logging.getLogger(__name__)


def make_style_transfer_inference_model(num_styles,
                                        style_predictor_factory_func: typing.Callable[[int], tf.keras.Model],
                                        style_transfer_factory_func: typing.Callable[[], tf.keras.Model],
                                        name="StyleTransferInferenceModel"):
    style_transfer_model, num_style_parameters = style_transfer_factory_func()
    style_predictor = style_predictor_factory_func(num_style_parameters)

    inputs = {
        'content': style_transfer_model.input['content'],
        'style_weights': style_transfer_model.input['style_weights'],
        'style': tf.keras.Input((num_styles,) + style_predictor.input.shape[1:]),
    }
    content_input, style_weights, style_input = (inputs['content'], inputs['style_weights'], inputs['style'])

    style_params = []
    for style_image in tf.unstack(style_input, axis=1):
        style_params.append(style_predictor(style_image))

    style_params = tf.stack(style_params, axis=1)

    stylized_image = style_transfer_model({
        "content": content_input,
        'style_weights': style_weights,
        "style_params": style_params,
    })

    model = tf.keras.Model(inputs=inputs, outputs=stylized_image, name=name)

    class StyleTransferModels:
        def __init__(self):
            self.inputs = inputs
            self.inference = model
            self.transfer = style_transfer_model
            self.style_predictor = style_predictor

    return StyleTransferModels()
