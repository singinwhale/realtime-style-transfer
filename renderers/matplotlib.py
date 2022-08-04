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


def predict_datapoint(validation_log_datapoint, training_log_datapoint, model):
    fig, subplots = plt.subplots(2, 2, sharex=True, sharey=True, dpi=600)

    for plot, name in zip(subplots.flatten(), ("content", "style", "validation_prediction", "training_prediction")):
        plot.title.set_text(name)
        plot.axis('off')

    content_plot, style_plot, valiation_prediction_plot, training_prediction_plot = subplots.flatten()
    content_plot.imshow(tf.squeeze(validation_log_datapoint['content']))
    style_plot.imshow(tf.squeeze(validation_log_datapoint['style']))
    valiation_prediction_plot.imshow(tf.squeeze(model(validation_log_datapoint)))
    training_prediction_plot.imshow(tf.squeeze(model(training_log_datapoint)))

    plt.show()
