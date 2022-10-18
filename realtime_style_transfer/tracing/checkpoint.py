import tensorflow as tf
from pathlib import Path


class CheckpointCallback(tf.keras.callbacks.Callback):
    model: tf.keras.models.Model

    def __init__(self, log_dir, cadence=1):
        super(CheckpointCallback).__init__()
        self.log_dir = Path(log_dir)
        self.cadence = cadence

    def on_epoch_end(self, epoch, logs=None):
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

        self.model.save_weights(self.log_dir / "latest_epoch_weights")
        if epoch % self.cadence == 0:
            self.model.save_weights(self.log_dir / f"epoch_{epoch}_weights")
