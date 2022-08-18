import tensorflow as tf
from tensorflow import keras


class CheckpointCallback(keras.callbacks.Callback):
    model: tf.keras.models.Model

    def __init__(self, log_dir):
        super(CheckpointCallback).__init__()
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(self.log_dir / "checkpoint")
