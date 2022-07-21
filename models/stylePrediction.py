import tensorflow as tf


class StylePredictionModel(tf.keras.Model):

    def __init__(self, input_shape, name="StylePredictionModel"):
        super(StylePredictionModel, self).__init__(name=name)
        self.efficientNet = tf.keras.applications.efficientnet_v2.EfficientNetV2S(include_top=False,
                                                                                  input_shape=input_shape[-3:])
        self.top = tf.keras.layers.Dense(100, activation=tf.nn.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.efficientNet(inputs)
        x = self.top(x)
        return x
