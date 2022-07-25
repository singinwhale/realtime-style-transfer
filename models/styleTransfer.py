import typing

import tensorflow as tf

import logging

log = logging.getLogger(__name__)


class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


def upsample(filters, size, name, norm_type='batchnorm', apply_dropout=False) -> tf.keras.Sequential:
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

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization(name=f"{name}_BN"))
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


class StyleTransferModel(tf.keras.Model):

    def __init__(self, input_shape,
                 style_predictor_factory_func: typing.Callable[[typing.List[tf.keras.layers.Layer]], tf.keras.Model],
                 norm_type='batchnorm', name="StyleTransferModel"):
        """Modified u-net generator model (https://arxiv.org/abs/1611.07004).
        Args:
          norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
          style_predictor_factory_func: function taking a list of batch norm layers from the style transfer model and
            produces a style prediction network
        Returns:
          Generator model
        """
        super(StyleTransferModel, self).__init__(name=name)
        encoder_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False,
                                                                       input_shape=input_shape[-3:])
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
            upsample(512, 4, name="0", norm_type=norm_type),  # (bs, 16, 16, 1024)
            upsample(256, 4, name="1", norm_type=norm_type),  # (bs, 32, 32, 512)
            upsample(128, 4, name="2", norm_type=norm_type),  # (bs, 64, 64, 256)
            upsample(64, 4, name="3", norm_type=norm_type),  # (bs, 128, 128, 128)
        ]

        self.top = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, padding='same',
                                                   name="top")  # 64x64 -> 128x128

        batchnorm_layers = []
        for layer in self.encoder.layers + [layer for decoder in self.decoder_layers for layer in decoder.layers]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                batchnorm_layers.append(layer)

        log.debug(f"Found {len(batchnorm_layers)} batchnorm layers")
        self.batchnorm_layers: typing.List[tf.keras.layers.BatchNormalization] = batchnorm_layers
        self.style_predictor = style_predictor_factory_func(self.batchnorm_layers)

    def call(self, inputs, training=None, mask=None):
        if self.inbound_nodes:
            if inputs.shape != self.input_shape:
                raise ValueError(f"Input does not have the expected shape: {inputs.shape} vs. expected {self.input_shape}")

        log.debug(inputs)
        expected_shapes = (inputs.shape[0],) + inputs.shape[-3:]
        content_input, style_input = (
            tf.squeeze(tf.gather(inputs, indices=[0], axis=1), axis=1),
            tf.squeeze(tf.gather(inputs, indices=[1], axis=1), axis=1)
        )
        assert content_input.shape == expected_shapes, \
            f"content_input: {content_input.shape} vs. expected {expected_shapes}"
        assert style_input.shape == expected_shapes, \
            f"style_input: {style_input.shape} vs. expected {expected_shapes}"
        style_params = self.style_predictor(style_input)

        for i, batchnorm_layer in enumerate(self.batchnorm_layers):
            beta = style_params[:, i * 2]
            gamma = style_params[:, i * 2 + 1]

            batchnorm_layer.beta = beta
            batchnorm_layer.gamma = gamma

        x_skip = self.encoder(content_input)

        x = x_skip[-1]
        skip_connections = reversed(x_skip[:-1])

        for decoder_layer, skip_connection in zip(self.decoder_layers, skip_connections):
            x = decoder_layer(x)
            if skip_connection is not None:
                x = tf.keras.layers.Concatenate()([x, skip_connection])

        return x
