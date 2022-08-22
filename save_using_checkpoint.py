from tracing import logsetup

import numpy as np

from pathlib import Path
import tensorflow as tf
import logging
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint_path', '-C', type=Path, required=True)
argparser.add_argument('--outpath', '-o', type=Path, required=True)

args = argparser.parse_args()
checkpoint_path = args.checkpoint_path
outpath = args.outpath

image_shape = (None, 960, 1920, 3)

log = logging.getLogger()

tf.config.set_visible_devices([], 'GPU')

from models import styleTransferFunctional, stylePrediction, styleLoss

input_shape = {'content': image_shape, 'style': image_shape}
output_shape = image_shape

style_loss_model = styleLoss.StyleLossModelMobileNet(output_shape)


def style_loss_callback():
    return styleLoss.make_style_loss_function(style_loss_model)


def build_style_prediction_callback(batchnorm_layers):
    return stylePrediction.StylePredictionModelMobileNet(input_shape, batchnorm_layers)


style_transfer_model = styleTransferFunctional.StyleTransferModelFunctional(
    input_shape,
    build_style_prediction_callback,
    style_loss_callback
)
element = {
    'content': tf.convert_to_tensor(np.zeros((1, 960, 1920, 3))),
    'style': tf.convert_to_tensor(np.zeros((1, 960, 1920, 3))),
}
log.info(f"Running inference to build model...")
# call once to build model
style_transfer_model(element)
# log.info(f"Loading weights...")
# style_transfer_model.load_weights(filepath=str(checkpoint_path))

log.info(f"Saving model...")
style_transfer_model.save(filepath=str(outpath), include_optimizer=False, save_format='tf')
