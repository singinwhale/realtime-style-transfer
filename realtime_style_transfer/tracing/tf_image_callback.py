import time

import tensorflow as tf


class SummaryImageCallback(tf.keras.callbacks.Callback):
    model: tf.keras.models.Model

    def __init__(self, validation_datapoint, training_datapoint):
        self.validation_datapoint = validation_datapoint
        self.training_datapoint = training_datapoint

    def on_epoch_begin(self, epoch, logs=None):
        if epoch != 0:
            return

        tf.summary.image('validation_style', self.validation_datapoint[0]['style'][:, 0, ...], step=0)
        tf.summary.image('validation_content', self.validation_datapoint[1]['content'], step=0)
        tf.summary.image('training_style', self.training_datapoint[0]['style'][:, 0, ...], step=0)
        tf.summary.image('training_content', self.training_datapoint[1]['content'], step=0)
        self.write_sample(0)

    def on_epoch_end(self, epoch, logs=None):
        self.write_sample(epoch + 1)

    def write_sample(self, index):
        start = time.perf_counter()
        validation_prediction = self.model.predict(self.validation_datapoint[0])
        end = time.perf_counter()
        training_prediction = self.model.predict(self.training_datapoint[0])
        tf.summary.scalar('prediction_time', start - end, step=index,
                          description="Duration of the training in seconds")
        tf.summary.image('validation_prediction', validation_prediction, step=index)
        tf.summary.image('training_prediction', training_prediction, step=index)
