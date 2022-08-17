import typing

import keras.layers
import tensorflow as tf

import logging

log = logging.getLogger(__name__)


class ConditionalInstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, num_feature_maps, epsilon=1e-5):
        super(ConditionalInstanceNormalization, self).__init__()
        self.epsilon = epsilon
        self.scale = None
        self.offset = None
        self.num_feature_maps = num_feature_maps

    def call(self, x):
        assert x.shape[-1] == self.num_feature_maps
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        if self.scale is None or self.offset is None:
            return normalized

        return self.scale * normalized + self.offset


def expand(filters, size, strides, name, apply_dropout=False) -> tf.keras.Sequential:
    name = f"expand_{name}"
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential(name=name)
    result.add(
        tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=size, strides=strides,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False, name=f"{name}_conv"))

    result.add(ConditionalInstanceNormalization(filters))

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def residual_block(input_shape, filters, size, strides, name, apply_dropout=False) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    fx = keras.layers.Conv2D(filters, size, strides=strides, activation='relu', padding='same')(inputs)
    fx = keras.layers.BatchNormalization()(fx)
    fx = keras.layers.Conv2D(filters, size, strides=strides, padding='same')(fx)
    out = keras.layers.Add()([inputs, fx])
    out = keras.layers.ReLU()(out)
    out = keras.layers.BatchNormalization()(out)
    return keras.models.Model(inputs, out)


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

    result.add(keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


class StyleTransferModel(tf.keras.Model):

    def __init__(self, input_shape,
                 style_predictor_factory_func: typing.Callable[[typing.List[tf.keras.layers.Layer]], tf.keras.Model],
                 style_loss_func: typing.Callable,
                 name="StyleTransferModel"):
        """Modified u-net generator model (https://arxiv.org/abs/1611.07004).
        Args:
          style_predictor_factory_func: function taking a list of batch norm layers from the style transfer model and
            produces a style prediction network
        Returns:
          Generator model
        """
        super(StyleTransferModel, self).__init__(name=name)

        self.style_loss = style_loss_func

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

        self.decoder_layers = [
            expand(64, 3, 2, name="1"),
            expand(32, 3, 2, name="2"),
        ]

        self.top = tf.keras.Sequential(layers=[
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=1, strides=1, padding='same'),  # -> (1, 480, 960, 3)
        ], name="Top")

        potential_layers = self.encoder.layers + [layer for decoder in self.decoder_layers for layer in decoder.layers]
        normalization_layers = list(filter(lambda layer: isinstance(layer, ConditionalInstanceNormalization),
                                           potential_layers))

        log.debug(f"Found {len(normalization_layers)} normalization layers")
        self.normalization_layers: typing.List[tf.keras.layers.BatchNormalization] = normalization_layers
        self.style_predictor = style_predictor_factory_func(self.normalization_layers)

    def call(self, inputs, training=None, mask=None):
        content_input, style_input = (inputs['content'], inputs['style'])
        style_params = self.style_predictor(style_input)

        style_norm_param_lower_bound = 0
        for normalization_layer in self.normalization_layers:
            style_norm_param_upper_bound = style_norm_param_lower_bound + normalization_layer.num_feature_maps
            scale = style_params[:, style_norm_param_lower_bound: style_norm_param_upper_bound]
            offset = style_params[:, style_norm_param_lower_bound: style_norm_param_upper_bound]
            scale, offset = tf.expand_dims(scale, -2), tf.expand_dims(offset, -2)
            scale, offset = tf.expand_dims(scale, -2), tf.expand_dims(offset, -2)
            normalization_layer.scale = scale
            normalization_layer.offset = offset

        x = self.encoder(content_input)
        x = self.bottleneck(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x)

        x = self.top(x)
        x = tf.nn.sigmoid(x)
        losses = self.style_loss(inputs, x)
        self.add_loss(losses['loss'])
        for loss_name, loss_value in losses.items():
            if loss_name == "loss":
                continue
            self.add_metric(value=loss_value, name=loss_name)

        return x
