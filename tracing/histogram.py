import logging

import tensorflow as tf
from tensorflow import keras

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def writeModelHistogramSummary(model, step=None):
    counter = 0

    def descend_to_layer(layer: tf.keras.layers.Layer, path):
        nonlocal counter
        layer_path = f"{path}/{layer.name}"
        if not layer.trainable:
            log.debug(f"skipping {layer_path} because it is not trainable")
            return

        if hasattr(layer, "layers"):
            for sub_layer in layer.layers:
                descend_to_layer(sub_layer, layer_path)
        else:
            for i, weights in enumerate(layer.get_weights()):
                log.debug(f"Writing layer {layer_path}/{i} for weights {weights.shape}")
                tf.summary.histogram(f"{layer_path}/{i}", weights, step)
                counter = counter + 1

    for layer in model.layers:
        descend_to_layer(layer, "")

    log.debug(f"Wrote {counter} histograms")


class HistogramCallback(keras.callbacks.Callback):
    model: tf.keras.models.Model

    def on_epoch_end(self, epoch, logs=None):
        writeModelHistogramSummary(self.model, step=epoch)

    def on_predict_end(self, logs=None):
        writeModelHistogramSummary(self.model, step=-1)
