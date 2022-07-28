import tensorflow as tf
from tensorflow import keras


class SummaryImageCallback(keras.callbacks.Callback):

    def __init__(self, sample_datapoint):
        self.sample_datapoint = sample_datapoint

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            tf.summary.image('style', self.sample_datapoint['style'], step=0)
            tf.summary.image('content', self.sample_datapoint['content'], step=0)
            self.write_sample(0)

    def on_epoch_end(self, epoch, logs=None):
        self.write_sample(epoch + 1)

    def write_sample(self, index):
        tf.summary.image('transfer', self.model.predict(self.sample_datapoint), step=index)
