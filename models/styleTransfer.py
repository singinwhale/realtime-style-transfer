import typing

import tensorflow as tf

import logging

log = logging.getLogger(__name__)


class StyleParamStack:

    def __init__(self, style_params, style_weights):
        self.style_params = style_params
        self.style_weights = style_weights
        self.lower_bound = 0

    def get_params(self, num_params):
        with tf.name_scope("StyleParamStack.get_params"):
            lower_bound = self.lower_bound
            upper_bound = lower_bound + num_params
            self.lower_bound = upper_bound
            return self.style_params[..., lower_bound:upper_bound]

    def make_content_and_style_input(self, content, num_params):
        return {
            'content': content,
            'style_weights': self.style_weights,
            'style_params': self.get_params(num_params)
        }


def _apply_style_weights(style_weights, style_params):
    with tf.name_scope("apply_style_weights"):
        if style_params.shape[-2] == 2:
            def weight_style(params, weights):
                return tf.einsum('byxc,biif->byxf', weights, params)

            return weight_style(tf.expand_dims(style_params[..., 0, :], -2), style_weights) + \
                   weight_style(tf.expand_dims(style_params[..., 1, :], -2), 1.0 - style_weights)

        return style_params


class ConditionalInstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""
    NumParamsPerFeature: int = 2

    def __init__(self, num_feature_maps, num_styles, name, epsilon=1e-5):
        super().__init__(name=f"ConditionalInstanceNormalization_{name}")
        self.epsilon = epsilon
        self.num_feature_maps = num_feature_maps
        self.num_styles = num_styles

    def call(self, x, **kwargs):
        inputs = x
        x = inputs['content']
        style_weights = inputs['style_weights']
        style_params = StyleParamStack(inputs['style_params'], style_weights)
        scale = _apply_style_weights(style_weights, style_params.get_params(self.num_feature_maps))
        bias = _apply_style_weights(style_weights, style_params.get_params(self.num_feature_maps))

        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)

        inv = tf.math.rsqrt(variance + self.epsilon) * scale
        x = x * tf.cast(inv, x.dtype) + tf.cast(bias - mean * inv, x.dtype)
        return x

    def get_config(self):
        return {
            "epsilon": self.epsilon,
            "num_feature_maps": self.num_feature_maps,
            "num_styles": self.num_feature_maps,
        }

    @classmethod
    def get_style_params_shape_and_num(cls, num_feature_maps, num_styles):
        num_style_params = cls.NumParamsPerFeature * num_feature_maps
        style_params_shape = (1, num_styles, num_style_params)
        return style_params_shape, num_style_params

    @classmethod
    def get_style_weights_shape(cls, content_shape: typing.Tuple, num_styles, multiplier=1) -> typing.Tuple:
        style_weights_shape = list(content_shape)
        style_weights_shape[-1] = num_styles - 1
        style_weights_shape[-2] *= multiplier
        style_weights_shape[-3] *= multiplier
        return tuple(style_weights_shape)


def expand(input_shape: typing.Tuple, num_styles, filters, size, strides, name, activation=tf.nn.relu):
    name = f"expand_{name}"
    initializer = tf.random_normal_initializer(0., 0.02)

    content_inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32, name="input_content")
    style_weights_input = tf.keras.Input(
        shape=ConditionalInstanceNormalization.get_style_weights_shape(
            input_shape,
            num_styles,
            multiplier=strides
        ),
        name="style_weights_input")

    style_params_shape, num_style_params = ConditionalInstanceNormalization.get_style_params_shape_and_num(
        filters,
        num_styles
    )
    style_params = tf.keras.layers.Input(shape=style_params_shape, name="input_scale")
    inputs = {
        "content": content_inputs,
        "style_weights": style_weights_input,
        "style_params": style_params,
    }
    result = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=size, strides=strides, padding='same',
        kernel_initializer=initializer,
        name=f"{name}_conv")(content_inputs)

    instance_norm_args = {
        "content": result,
        "style_weights": style_weights_input,
        "style_params": style_params,
    }
    result = ConditionalInstanceNormalization(filters, num_styles, 0)(instance_norm_args)

    result = tf.keras.layers.Activation(activation)(result)

    expand_block_model = tf.keras.Model(inputs, result, name=name)
    upscale_factor = strides
    return expand_block_model, num_style_params, upscale_factor


def residual_block(input_shape: typing.Tuple, num_styles, filters, size, strides, name):
    name = f"residual_block_{name}"
    initializer = tf.random_uniform_initializer(0., 0.05)
    content_input = tf.keras.Input(shape=input_shape, name="content_input")
    style_weights_input = tf.keras.Input(
        shape=ConditionalInstanceNormalization.get_style_weights_shape(input_shape, num_styles),
        name="style_weights_input")
    num_conv_and_norms = 2
    style_params_shape, num_style_params = \
        ConditionalInstanceNormalization.get_style_params_shape_and_num(filters * num_conv_and_norms, num_styles)

    style_params_input = tf.keras.Input(shape=style_params_shape, name="style_params_input")
    inputs = {
        'content': content_input,
        'style_weights': style_weights_input,
        'style_params': style_params_input,
    }
    style_params = StyleParamStack(style_params_input, style_weights_input)

    fx = content_input
    for i in range(num_conv_and_norms):
        fx = tf.keras.layers.Conv2D(filters, size, name=f"{name}_conv{i}", strides=strides, activation=tf.nn.relu,
                                    padding='same',
                                    kernel_initializer=initializer)(fx)
        fx = ConditionalInstanceNormalization(filters, num_styles, name=i)(
            style_params.make_content_and_style_input(
                fx,
                filters * ConditionalInstanceNormalization.NumParamsPerFeature
            )
        )

    out = tf.keras.layers.Add()([content_input, fx])
    return (tf.keras.models.Model(inputs, out, name=f"{name}"), num_style_params)


def contract(input_shape, filters, size, strides, name) -> tf.keras.Sequential:
    name = f"contract_{name}"
    initializer = tf.random_normal_initializer(0., 0.02)

    inputs = tf.keras.Input(shape=input_shape, name="contract_input")

    x = tf.keras.layers.Conv2D(filters, size,
                               strides=strides,
                               padding='same',
                               kernel_initializer=initializer,
                               activation=tf.nn.relu,
                               name=f"{name}_conv")(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    result = tf.keras.layers.ReLU()(x)

    return tf.keras.Model(inputs, result, name=name)


def calc_next_conv_dims(initial, filters, mult) -> typing.Tuple:
    if len(initial) == 3: return (int(initial[0] * mult), int(initial[1] * mult), filters)
    return (initial[0], initial[1] * mult, initial[2] * mult, filters)


def create_style_transfer_model(input_shape, num_styles,
                                name="StyleTransferModel"):
    contract_blocks = [
        contract(input_shape, 32, 9, 1, name="0"),
        contract(calc_next_conv_dims(input_shape, 32, 1), 64, 3, 2, name="1"),
        contract(calc_next_conv_dims(input_shape, 64, 2 ** -1), 128, 3, 2, name="2"),
    ]

    res_input_shape = calc_next_conv_dims(input_shape, 128, 2 ** -2)
    residual_blocks = [
        residual_block(res_input_shape, num_styles, 128, 3, 1, name="0"),
        residual_block(res_input_shape, num_styles, 128, 3, 1, name="1"),
        residual_block(res_input_shape, num_styles, 128, 3, 1, name="2"),
        residual_block(res_input_shape, num_styles, 128, 3, 1, name="3"),
        residual_block(res_input_shape, num_styles, 128, 3, 1, name="4"),
    ]

    expand_blocks = [
        expand(input_shape=res_input_shape, name="0", num_styles=num_styles,
               filters=64, size=3, strides=2),
        expand(input_shape=calc_next_conv_dims(res_input_shape, 64, 2 ** 1), name="1", num_styles=num_styles,
               filters=32, size=3, strides=2),
        expand(input_shape=calc_next_conv_dims(res_input_shape, 32, 2 ** 2), name="2", num_styles=num_styles,
               filters=3, size=9, strides=1,
               activation=tf.nn.sigmoid),
    ]

    num_style_parameters = (sum(map(lambda block_w_params: block_w_params[1], residual_blocks)) +
                            sum(map(lambda block_w_params: block_w_params[1], expand_blocks)))

    inputs = {'content': tf.keras.layers.Input(shape=input_shape),
              'style_weights': tf.keras.layers.Input(
                  shape=ConditionalInstanceNormalization.get_style_weights_shape(input_shape, num_styles)),
              'style_params': tf.keras.layers.Input(
                  shape=(num_styles, num_style_parameters)
              )}
    content_input, style_weights, style_params_input = (inputs['content'],
                                                        inputs['style_weights'],
                                                        inputs['style_params'])

    with tf.name_scope("expand_style_params"):
        style_params_input = tf.expand_dims(style_params_input, 1, name="expand_style_params_0")
        # style_params_input = tf.expand_dims(style_params_input, 1, name="expand_style_params_1")

    x = content_input
    for contract_block in contract_blocks:
        x = contract_block(x)

    style_weights_mips = _get_style_weight_mips(style_weights)

    style_params_stack = StyleParamStack(style_params_input, style_weights)
    for residual_block_layer, num_style_features in residual_blocks:
        x = {
            'content': x,
            'style_weights': style_weights_mips[x.shape[-2]],
            'style_params': style_params_stack.get_params(num_style_features)
        }
        x = residual_block_layer(x)

    for expand_block, num_style_features, upscale_factor in expand_blocks:
        x = {
            'content': x,
            'style_weights': style_weights_mips[x.shape[-2] * upscale_factor],
            'style_params': style_params_stack.get_params(num_style_features)
        }
        x = expand_block(x)

    model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
    return model, num_style_parameters


def _get_style_weight_mips(style_weights):
    with tf.name_scope("style_weight_mips"):
        style_weights_mips = {}
        downsample_half_layer = tf.keras.layers.AvgPool2D(2)
        last_style_weights_mip = style_weights
        last_style_weights_mip = style_weights_mips[last_style_weights_mip.shape[-2]] = last_style_weights_mip
        last_style_weights_mip = style_weights_mips[last_style_weights_mip.shape[-2]] = downsample_half_layer(
            last_style_weights_mip)
        last_style_weights_mip = style_weights_mips[last_style_weights_mip.shape[-2]] = downsample_half_layer(
            last_style_weights_mip)
        return style_weights_mips
