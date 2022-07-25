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


class StylePredictionModel(tf.keras.Model):

    def __init__(self, input_shape, batchnorm_layers, dropout_rate=0.2, name="StylePredictionModel"):
        super(StylePredictionModel, self).__init__(name=name)
        self.efficientNet = tf.keras.applications.efficientnet_v2.EfficientNetV2S(include_top=False,
                                                                                  input_shape=input_shape[-3:])
        self.dropout_rate = dropout_rate
        self.num_style_parameters = len(batchnorm_layers) * 2

    def call(self, inputs, training=None, mask=None):
        x = self.efficientNet(inputs)

        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        if self.dropout_rate > 0:
            x = tf.keras.layers.Dropout(self.dropout_rate, name="top_dropout")(x)

        x = tf.keras.layers.Dense(
            self.num_style_parameters,
            activation=tf.keras.activations.softmax,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            bias_initializer=tf.constant_initializer(0),
            name="style_predictions")(x)
        return x
