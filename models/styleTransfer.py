import tensorflow as tf


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


def upsample(filters, size, name, norm_type='batchnorm', apply_dropout=False):
    """Upsamples an input.
    Conv2DTranspose => Batchnorm => Dropout => Relu
    Args:
      filters: number of filters
      size: filter size
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
                                        use_bias=False))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


class StyleTransferModel(tf.keras.Model):

    def __init__(self, input_shape, norm_type='batchnorm', name="StyleTransferModel"):
        """Modified u-net generator model (https://arxiv.org/abs/1611.07004).
        Args:
          norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
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
            upsample(512, 4, norm_type),  # (bs, 16, 16, 1024)
            upsample(256, 4, norm_type),  # (bs, 32, 32, 512)
            upsample(128, 4, norm_type),  # (bs, 64, 64, 256)
            upsample(64, 4, norm_type),  # (bs, 128, 128, 128)
        ]

        self.top = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, padding='same', name="top")  #64x64 -> 128x128

    def call(self, inputs, training=None, mask=None):
        x_skip = self.encoder(inputs)

        x = x_skip[-1]
        skip_connections = reversed(x_skip[:-1])

        for decoder_layer, skip_connection in zip(self.decoder_layers, skip_connections):
            x = decoder_layer(x)
            if skip_connection is not None:
                print(skip_connection)
                x = tf.keras.layers.Concatenate()([x, skip_connection])

        return x
