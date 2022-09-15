import typing

import tensorflow as tf

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def visitModelLayers(model, only_trainable, visitor_func: typing.Callable[[tf.keras.layers.Layer, str], None]):
    counter = 0

    def descend_to_layer(layer: tf.keras.layers.Layer, path):
        nonlocal counter
        layer_path = f"{path}/{layer.name}"
        if only_trainable and not layer.trainable:
            log.debug(f"skipping {layer_path} because it is not trainable")
            return

        if hasattr(layer, "layers"):
            for sub_layer in layer.layers:
                descend_to_layer(sub_layer, layer_path)
        else:
            visitor_func(layer, layer_path)
            counter += 1

    for layer in model.layers:
        descend_to_layer(layer, "")

    log.debug(f"Visited {counter} layers")
