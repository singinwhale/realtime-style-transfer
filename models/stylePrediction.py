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

    def __init__(self, batchnorm_layers, dropout_rate=0.2, name="StylePredictionModel"):
        super(StylePredictionModelBase, self).__init__(name=name)

        self.dropout_rate = dropout_rate
        num_style_parameters = len(batchnorm_layers) * 2
        # todo: add second dense layer as suggested in Ghiasi et al., 2017
        self.top = tf.keras.layers.Dense(
            num_style_parameters,
            activation=tf.keras.activations.softmax,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            bias_initializer=tf.constant_initializer(0),
            name="style_predictions")

    def call(self, inputs, training=None, mask=None):
        x = self.feature_extractor(inputs)

        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        if self.dropout_rate > 0:
            x = tf.keras.layers.Dropout(self.dropout_rate, name="top_dropout")(x)

        x = self.top(x)
        return x


class StylePredictionModelEfficientNet(StylePredictionModelBase):

    def __init__(self, input_shape, batchnorm_layers, dropout_rate=0.2, name="StylePredictionModelEfficientNet"):
        super(StylePredictionModelEfficientNet, self).__init__(batchnorm_layers, dropout_rate, name=name)
        self.feature_extractor = \
            tf.keras.applications.efficientnet_v2.EfficientNetV2S(include_top=False,
                                                                  input_shape=input_shape['style'][-3:])


class StylePredictionModelMobileNet(StylePredictionModelBase):

    def __init__(self, input_shape, batchnorm_layers, dropout_rate=0.2, name="StylePredictionModel"):
        super(StylePredictionModelMobileNet, self).__init__(batchnorm_layers, dropout_rate, name=name)
        self.feature_extractor = \
            tf.keras.applications.MobileNetV3Small(include_top=False,
                                                   input_shape=input_shape['style'][-3:])
