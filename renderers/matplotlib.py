import logging

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def predict_datapoint(datapoint, model):
    fig, subplots = plt.subplots(2, 2, sharex=True, sharey=True, dpi=600)

    for plot, name in zip(subplots.flatten(), ("content", "style", "prediction", "prediction_remapped")):
        plot.title.set_text(name)

    content_plot, style_plot, prediction_plot, prediction_remapped_plot = subplots.flatten()
    content_plot.imshow(tf.squeeze(datapoint['content']))
    style_plot.imshow(tf.squeeze(datapoint['style']))
    prediction: tf.Tensor = tf.squeeze(model(datapoint))
    prediction_remapped = prediction
    min, max = tf.reduce_min(prediction), tf.reduce_max(prediction)
    if min < 0 or max > 1:
        logging.warning(f"prediction has values that are not between 0 and 1: min: {min} max: {max}. Remapping")
        prediction_remapped = (prediction.numpy() - min) / (max - min)

    prediction_plot.imshow(prediction)
    prediction_remapped_plot.imshow(prediction_remapped)
    plt.show()
