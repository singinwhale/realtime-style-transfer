import tensorflow as tf
from pathlib import Path


class MetricsCallback(tf.keras.callbacks.Callback):
    model: tf.keras.models.Model

    def __init__(self, log_dir: Path):
        super().__init__()
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs=None):
        training_writer = tf.summary.create_file_writer(str(self.log_dir / 'training'))
        validation_writer = tf.summary.create_file_writer(str(self.log_dir / 'validation'))
        log: str
        for log, value in logs.items():
            writer = training_writer
            if log.startswith('val_'):
                log = log[4:]
                writer = validation_writer
            with writer.as_default(epoch):
                tf.summary.scalar(log, tf.reduce_mean(value))
