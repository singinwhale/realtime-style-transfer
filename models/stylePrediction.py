import tensorflow as tf

import logging

log = logging.getLogger(__name__)

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 1. / 3.,
        "mode": "fan_out",
        "distribution": "uniform"
    }
}


class StylePredictionModelBase(tf.keras.Model):
    feature_extractor = None

    def __init__(self, num_top_parameters, num_style_parameters=100, dropout_rate=0.2, name="StylePredictionModel"):
        super(StylePredictionModelBase, self).__init__(name=name)

        self.dropout_rate = dropout_rate
        log.info(f"Using {num_style_parameters} style parameters")
        self.style_predictor = tf.keras.layers.Dense(
            num_style_parameters,
            activation=tf.keras.activations.softmax,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            bias_initializer=tf.constant_initializer(1),
            name="style_predictor")

        num_norm_parameters = num_top_parameters
        log.debug(f"Using {num_norm_parameters} norm parameters")
        self.style_norm_predictor = tf.keras.layers.Dense(
            num_norm_parameters,
            activation=tf.keras.activations.softmax,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            bias_initializer=tf.constant_initializer(0.5),
            name="style_norm_predictor")

    def call(self, inputs, training=None, mask=None):
        x = self.feature_extractor(inputs)

        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        if self.dropout_rate > 0:
            x = tf.keras.layers.Dropout(self.dropout_rate, name="top_dropout")(x)

        x = self.style_predictor(x)
        x = self.style_norm_predictor(x)
        return x


class StylePredictionModelEfficientNet(StylePredictionModelBase):

    def __init__(self, input_shape, num_top_parameters, dropout_rate=0.2, name="StylePredictionModelEfficientNet"):
        super(StylePredictionModelEfficientNet, self).__init__(num_top_parameters, dropout_rate=dropout_rate, name=name)
        self.feature_extractor = \
            tf.keras.applications.efficientnet_v2.EfficientNetV2S(include_top=False,
                                                                  input_shape=input_shape['style'][-3:])
        self.feature_extractor.trainable = True


class StylePredictionModelMobileNet(StylePredictionModelBase):

    def __init__(self, input_shape, num_top_parameters, dropout_rate=0.2, name="StylePredictionModel"):
        super(StylePredictionModelMobileNet, self).__init__(num_top_parameters, dropout_rate=dropout_rate, name=name)
        self.feature_extractor = \
            tf.keras.applications.MobileNetV3Small(include_top=False,
                                                   input_shape=input_shape['style'][-3:])
        self.feature_extractor.trainable = True
