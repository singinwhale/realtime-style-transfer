import typing

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
        return self.scale * normalized + self.offset


def upsample(filters, size, name, apply_dropout=False) -> tf.keras.Sequential:
    """Upsamples an input.
    Conv2DTranspose => Batchnorm => Dropout => Relu
    Args:
      filters: number of filters
      size: filter size
      name: name infix of the layers in this upsampler
      norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
      apply_dropout: If True, adds the dropout layer
    Returns:
      Upsample Sequential Model
    """
    name = f"upsample_{name}"
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential(name=name)
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False, name=f"{name}_conv"))

    result.add(ConditionalInstanceNormalization(filters))

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

        encoder_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                                       input_shape=input_shape['content'][-3:])
        encoder_model.trainable = False

        encoder_model_layers = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        encoder_model_layers = [encoder_model.get_layer(layer_name).output for layer_name in encoder_model_layers]

        self.encoder = tf.keras.Model(inputs=encoder_model.input, outputs=encoder_model_layers)
        self.decoder_layers = [
            upsample(512, 4, name="0"),  # (bs, 16, 16, 1024)
            upsample(256, 4, name="1"),  # (bs, 32, 32, 512)
            upsample(128, 4, name="2"),  # (bs, 64, 64, 256)
            upsample(64, 4, name="3"),  # (bs, 128, 128, 128)
        ]

        self.top = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, padding='same',
                                                   name="top")  # 64x64 -> 128x128

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

        x_skip = self.encoder(content_input)

        x = x_skip[-1]
        skip_connections = reversed(x_skip[:-1])

        for decoder_layer, skip_connection in zip(self.decoder_layers, skip_connections):
            x = decoder_layer(x)
            if skip_connection is not None:
                x = tf.keras.layers.Concatenate()([x, skip_connection])

        x = self.top(x)

        self.add_loss(self.style_loss(inputs, x))

        return x
