import tensorflow as tf
from pathlib import Path


class CheckpointCallback(tf.keras.callbacks.Callback):
    model: tf.keras.models.Model

    def __init__(self, log_dir, cadence=1):
        super(CheckpointCallback).__init__()
        self.checkpoint_log_dir = Path(log_dir) / "checkpoints"
        self.weights_log_dir = Path(log_dir) / "weights"
        self.cadence = cadence
        self.cadence_checkpoint_manager = None
        self.continuous_checkpoint_manager = None
        self.checkpoint = None
        self.step = tf.Variable(0, trainable=False)

    def on_epoch_end(self, epoch, logs=None):
        self.step.assign(epoch)
        if self.checkpoint is None:
            self.checkpoint = tf.train.Checkpoint(self.model)
            self.cadence_checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                                                                         checkpoint_name="ckpt",
                                                                         directory=str(self.checkpoint_log_dir),
                                                                         checkpoint_interval=self.cadence,
                                                                         max_to_keep=5,
                                                                         step_counter=self.step)
            self.continuous_checkpoint_manager = tf.train.CheckpointManager(self.checkpoint,
                                                                            checkpoint_name="latest_ckpt",
                                                                            directory=str(self.checkpoint_log_dir),
                                                                            max_to_keep=1,
                                                                            step_counter=self.step)
        self.checkpoint.save_counter.assign(epoch)
        self.cadence_checkpoint_manager.save(epoch, check_interval=True)
        self.checkpoint.save_counter.assign(epoch)
        self.continuous_checkpoint_manager.save(epoch)
        self.model.save_weights(str(self.weights_log_dir / "latest_epoch_weights"))
