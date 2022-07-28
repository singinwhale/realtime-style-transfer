import logging

import tensorflow as tf
import matplotlib.pyplot as plt

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def predict_datapoint(datapoint, model):
    logging.debug(datapoint)
    fig, (content_plot, style_plot, transfer_plot) = plt.subplots(1, 3, sharex=True, sharey=True)
    content_plot.imshow(tf.squeeze(datapoint['content'])*255.0)
    style_plot.imshow(tf.squeeze(datapoint['style'])*255.0)
    transfer_plot.imshow(tf.squeeze(model(datapoint))*255.0)
    plt.show()
