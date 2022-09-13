import logging

import tensorflow as tf

from .common import visitModelLayers

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class GradientsCallback(tf.keras.callbacks.Callback):
    model: tf.keras.models.Model

    def __init__(self, datapoint):
        super().__init__()
        self.datapoint = datapoint

    def on_epoch_end(self, epoch, logs=None):
        with tf.GradientTape(persistent=True) as grad_tape:
            grad_tape.watch(self.datapoint)
            _ = self.model(self.datapoint)
            loss = self.model.losses[0]
            grads = grad_tape.gradient(loss, self.model.trainable_weights, loss)

            grads = {weights.name: grads for weights, grads in zip(self.model.trainable_weights, grads)}

            def write_grads(layer, layer_path):
                for i, weights in enumerate(layer.trainable_weights):
                    grad = grads[weights.name]
                    tf.summary.histogram(f"{layer_path}/{i}_grad", grad, epoch, buckets=100)
                    axes = list(range(len(grad.shape)))
                    mean, variance = tf.nn.moments(grad, axes=axes)
                    tf.summary.scalar(f"{layer_path}/{i}_grad_mean", mean, epoch)
                    tf.summary.scalar(f"{layer_path}/{i}_grad_var", variance, epoch)

            visitModelLayers(self.model, True, write_grads)
