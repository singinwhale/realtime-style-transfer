import typing

import tensorflow as tf

import logging

log = logging.getLogger(__name__)


class ConditionalInstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, num_feature_maps, epsilon=1e-5):
        super(ConditionalInstanceNormalization, self).__init__(name="ConditionalInstanceNormalization")
        self.epsilon = epsilon
        self.num_feature_maps = num_feature_maps

    def call(self, x):
        inputs = x
        x = inputs['inputs']
        scale = inputs['scale']
        bias = inputs['bias']

        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        x = normalized

        x = x * scale + bias
        return x

    def get_config(self):
        return {
            "epsilon": self.epsilon,
            "num_feature_maps": self.num_feature_maps,
        }


def expand(input_shape, filters, size, strides, name, apply_dropout=False) -> tf.keras.Model:
    name = f"expand_{name}"
    initializer = tf.random_normal_initializer(0., 0.02)

    content_inputs = tf.keras.layers.Input(shape=input_shape)
    scale = tf.keras.layers.Input(shape=(1, 1, filters))
    bias = tf.keras.layers.Input(shape=(1, 1, filters))
    inputs = {
        "inputs": content_inputs,
        "scale": scale,
        "bias": bias,
    }
    result = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=size, strides=strides, padding='same',
        kernel_initializer=initializer,
        use_bias=False,  # todo: investigate why this is False in the magenta project
        name=f"{name}_conv")(content_inputs)

    instance_norm_args = {
        "inputs": result,
        "scale": scale,
        "bias": bias,
    }
    result = ConditionalInstanceNormalization(filters)(instance_norm_args)

    if apply_dropout:
        result = tf.keras.layers.Dropout(0.5)(result)

    result = tf.keras.layers.ReLU()(result)

    return tf.keras.Model(inputs, result, name=name)


def residual_block(input_shape, filters, size, strides, name, apply_dropout=False) -> tf.keras.Model:
    name = f"residual_block_{name}"
    initializer = tf.random_uniform_initializer(0., 0.05)
    inputs = tf.keras.Input(shape=input_shape)
    fx = tf.keras.layers.Conv2D(filters, size, name=f"{name}_conv0", strides=strides, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
    fx = tf.keras.layers.BatchNormalization()(fx)
    fx = tf.keras.layers.Conv2D(filters, size, name=f"{name}_conv1", strides=strides, padding='same', kernel_initializer=initializer)(fx)
    out = tf.keras.layers.Add()([inputs, fx])
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.BatchNormalization()(out)
    return tf.keras.models.Model(inputs, out, name=f"{name}")


def contract(filters, size, strides, name, apply_dropout=False) -> tf.keras.Sequential:
    name = f"contract_{name}"
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential(name=name)
    result.add(
        tf.keras.layers.Conv2D(filters, size,
                               strides=strides,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False,
                               name=f"{name}_conv")
    )

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


class AssignLayer(tf.keras.layers.Layer):

    def __init__(self, variable, **kwargs):
        super().__init__(name="AssignLayer", **kwargs)
        self.variable = variable

    def call(self, inputs, *args, **kwargs):
        log.debug(f"Assigning to variable {self.variable.name}")
        self.variable.assign(inputs, name=f"assign_{self.variable.name.replace(':', '')}")


def StyleTransferModelFunctional(input_shape,
                                 style_predictor_factory_func: typing.Callable[[int], tf.keras.Model],
                                 style_loss_func_factory_func: typing.Callable[[], typing.Callable[[typing.Dict, tf.Tensor], typing.Dict]],
                                 name="StyleTransferModel"):
    decoder_layer_specs = [
        {"filters": 64, "size": 3, "strides": 2},
        {"filters": 32, "size": 3, "strides": 2},
    ]

    inputs = {'content': tf.keras.layers.Input(shape=input_shape['content'][1:]), 'style': tf.keras.layers.Input(shape=input_shape['style'][1:])}
    content_input, style_input = (inputs['content'], inputs['style'])

    num_style_parameters = sum(map(lambda spec: spec['filters'] * 2, decoder_layer_specs))
    style_predictor = style_predictor_factory_func(num_style_parameters)
    style_params = style_predictor(style_input)

    x = tf.keras.Sequential(layers=[
        contract(32, 9, 1, name="0"),
        contract(64, 3, 2, name="1"),
        contract(128, 3, 2, name="2"),
    ], name="encoder")(content_input)

    residual_block_input_shape = (input_shape['content'][1] // 4, input_shape['content'][2] // 4, 128)
    x = tf.keras.Sequential(layers=[
        residual_block(residual_block_input_shape, 128, 3, 1, name="0"),
        residual_block(residual_block_input_shape, 128, 3, 1, name="1"),
        residual_block(residual_block_input_shape, 128, 3, 1, name="2"),
        residual_block(residual_block_input_shape, 128, 3, 1, name="3"),
        residual_block(residual_block_input_shape, 128, 3, 1, name="4"),
    ], name="bottleneck")(x)

    input_filters = residual_block_input_shape[-1]
    style_norm_param_lower_bound = 0
    for i, decoder_layer_spec in enumerate(decoder_layer_specs):
        style_norm_scale_upper_bound = style_norm_param_lower_bound + decoder_layer_spec["filters"]
        style_norm_offset_upper_bound = style_norm_scale_upper_bound + decoder_layer_spec["filters"]
        scale = style_params[:, style_norm_param_lower_bound:style_norm_scale_upper_bound]
        offset = style_params[:, style_norm_scale_upper_bound:style_norm_offset_upper_bound]
        scale, offset = tf.expand_dims(scale, -2, name="expand_scale_0"), tf.expand_dims(offset, -2, name="expand_offset_0")
        scale, offset = tf.expand_dims(scale, -2, name="expand_scale_1"), tf.expand_dims(offset, -2, name="expand_offset_1")
        style_norm_param_lower_bound = style_norm_offset_upper_bound

        input_shape = (residual_block_input_shape[0] * 2 ** i, residual_block_input_shape[1] * 2 ** i, input_filters)
        expand_layer = expand(input_shape=input_shape, name=i,
                              filters=decoder_layer_spec["filters"],
                              size=decoder_layer_spec["size"],
                              strides=decoder_layer_spec["strides"])
        x = expand_layer({
            "inputs": x,
            "scale": scale,
            "bias": offset,
        })
        input_filters = decoder_layer_spec["filters"]

    x = tf.keras.Sequential(layers=[
        tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=1, strides=1, padding='same',
                                        kernel_initializer=tf.random_uniform_initializer(0, 0.02),
                                        bias_initializer=tf.constant_initializer(0.0)),
        tf.keras.layers.Activation(tf.nn.sigmoid),
    ], name="Top")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
    losses = style_loss_func_factory_func()(inputs, x)
    model.add_loss(losses['loss'])
    for loss_name, loss_value in losses.items():
        if loss_name == "loss":
            continue
        model.add_metric(value=loss_value, name=loss_name)
    return model
