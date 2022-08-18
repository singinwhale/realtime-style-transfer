import typing

import tensorflow as tf
import numpy as np

import logging

log = logging.getLogger(__name__)


class ConditionalInstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, num_feature_maps, batch_size, epsilon=1e-5):
        super(ConditionalInstanceNormalization, self).__init__()
        self.epsilon = epsilon
        self.scale = None
        self.offset = None
        self.num_feature_maps = num_feature_maps

    def call(self, x, **kwargs):
        assert x.shape[-1] == self.num_feature_maps
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        scale = self.scale
        offset = self.offset
        if scale is None or offset is None:
            return normalized

        return scale * normalized + offset


def expand(filters, size, strides, batch_size, name, apply_dropout=False) -> tf.keras.Sequential:
    name = f"expand_{name}"
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential(name=name)
    result.add(
        tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=size, strides=strides,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False, name=f"{name}_conv"))

    result.add(ConditionalInstanceNormalization(filters, batch_size))

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def residual_block(input_shape, filters, size, strides, name, apply_dropout=False) -> tf.keras.Model:
    name = f"residual_block_{name}"
    inputs = tf.keras.Input(shape=input_shape)
    fx = tf.keras.layers.Conv2D(filters, size, strides=strides, activation='relu', padding='same')(inputs)
    fx = tf.keras.layers.BatchNormalization()(fx)
    fx = tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same')(fx)
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
                               use_bias=False, name=f"{name}_conv"))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


class StyleTransferModel(tf.keras.Model):

    def __init__(self, input_shape,
                 style_predictor_factory_func: typing.Callable[[typing.List[tf.keras.layers.Layer]], tf.keras.Model],
                 style_loss_func_factory_func: typing.Callable,
                 name="StyleTransferModel"):
        """Modified u-net generator model (https://arxiv.org/abs/1611.07004).
        Args:
          style_predictor_factory_func: function taking a list of batch norm layers from the style transfer model and
            produces a style prediction network
        Returns:
          Generator model
        """
        super(StyleTransferModel, self).__init__(name=name)

        self.style_loss = style_loss_func_factory_func()

        # encoder_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
        #                                                               input_shape=input_shape['content'][-3:])
        # encoder_model.trainable = True
        # self.encoder = encoder_model

        self.encoder = tf.keras.Sequential(layers=[
            contract(32, 9, 1, name="0"),
            contract(64, 3, 2, name="1"),
            contract(128, 3, 2, name="2"),
        ], name="encoder")

        res_input_shape = (input_shape['content'][1] // 4, input_shape['content'][2] // 4, 128)
        log.debug(f"res_input_shape: {res_input_shape}")
        self.bottleneck = tf.keras.Sequential(layers=[
            residual_block(res_input_shape, 128, 3, 1, name="0"),
            residual_block(res_input_shape, 128, 3, 1, name="1"),
            residual_block(res_input_shape, 128, 3, 1, name="2"),
            residual_block(res_input_shape, 128, 3, 1, name="3"),
            residual_block(res_input_shape, 128, 3, 1, name="4"),
        ], name="bottleneck")

        self.top = tf.keras.Sequential(layers=[
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=1, strides=1, padding='same'),  # -> (1, 480, 960, 3)
        ], name="Top")
        self.style_predictor_factory_func = style_predictor_factory_func

    def build(self, input_shape):
        log.debug(f"input_shape: {input_shape}")
        batch_size = input_shape['content'][0]
        self.decoder_layers = [
            expand(64, 3, 2, batch_size, name="1"),
            expand(32, 3, 2, batch_size, name="2"),
        ]
        self.decoder = tf.keras.Sequential(layers=self.decoder_layers, name="decoder")

        self.style_predictor = self.style_predictor_factory_func(self.get_normalization_layers())


    def call(self, inputs, training=None, mask=None):
        content_input, style_input = (inputs['content'], inputs['style'])
        style_params = self.style_predictor(style_input)

        self.apply_style_params(style_params)
        x = self.encoder(content_input)
        x = self.bottleneck(x)
        x = self.decoder(x)

        x = self.top(x)
        x = tf.nn.sigmoid(x)
        losses = self.style_loss(inputs, x)
        self.add_loss(losses['loss'])
        for loss_name, loss_value in losses.items():
            if loss_name == "loss":
                continue
            self.add_metric(value=loss_value, name=loss_name)

        return x

    def get_normalization_layers(self) -> typing.List[ConditionalInstanceNormalization]:
        potential_layers = [layer for decoder in self.decoder_layers for layer in decoder.layers]
        normalization_layers = list(filter(lambda layer: isinstance(layer, ConditionalInstanceNormalization),
                                           potential_layers))
        return normalization_layers

    def apply_style_params(self, style_params):
        style_norm_param_lower_bound = 0

        for normalization_layer in self.get_normalization_layers():
            style_norm_scale_upper_bound = style_norm_param_lower_bound + normalization_layer.num_feature_maps
            style_norm_offset_upper_bound = style_norm_scale_upper_bound + normalization_layer.num_feature_maps
            scale = style_params[:, style_norm_param_lower_bound:style_norm_scale_upper_bound]
            offset = style_params[:, style_norm_scale_upper_bound:style_norm_offset_upper_bound]
            scale, offset = tf.expand_dims(scale, -2, name="expand_scale_0"), tf.expand_dims(offset, -2, name="expand_offset_0")
            scale, offset = tf.expand_dims(scale, -2, name="expand_scale_1"), tf.expand_dims(offset, -2, name="expand_offset_1")
            normalization_layer.scale = (scale)
            normalization_layer.offset = (offset)
            style_norm_param_lower_bound = style_norm_offset_upper_bound
