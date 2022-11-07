import tensorflow as tf
from pathlib import Path


class CheckpointCallback(tf.keras.callbacks.Callback):
    model: tf.keras.models.Model

    def __init__(self, log_dir, cadence=1):
        super(CheckpointCallback).__init__()
        self.log_dir = Path(log_dir)
        self.cadence = cadence
        self.checkpoint_manager = None
        self.checkpoint = None
        self.step = tf.Variable(0, trainable=False)

    def on_epoch_end(self, epoch, logs=None):
        self.step.assign(epoch)
        if self.checkpoint is None:
            self.checkpoint = tf.train.Checkpoint(self.model)
        if self.checkpoint_manager is None:
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                                                                 directory=str(self.log_dir),
                                                                 checkpoint_interval=10,
                                                                 max_to_keep=5,
                                                                 step_counter=self.step)
        self.checkpoint_manager.save(epoch, check_interval=True)
        self.checkpoint.write(str(self.log_dir / "latest_epoch_checkpoint"))
        self.model.save_weights(str(self.log_dir / "latest_epoch_weights"))
