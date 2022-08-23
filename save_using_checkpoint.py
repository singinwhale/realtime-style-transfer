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
outpath: Path = args.outpath

image_shape = (960, 1920, 3)

log = logging.getLogger()

tf.config.set_visible_devices([], 'GPU')

from models import styleTransferFunctional, stylePrediction, styleLoss, styleTransferTrainingModel

input_shape = {'content': image_shape, 'style': image_shape}
output_shape = image_shape


def build_style_loss_function():
    style_loss_model = styleLoss.StyleLossModelMobileNet(output_shape)
    return styleLoss.make_style_loss_function(style_loss_model)


def build_style_prediction_model(batchnorm_layers):
    # return stylePrediction.StylePredictionModelMobileNet(input_shape, batchnorm_layers)
    return stylePrediction.create_style_prediction_model(
        input_shape['style'],
        stylePrediction.StyleFeatureExtractor.MOBILE_NET,
        batchnorm_layers,
    )


def build_style_transfer_model():
    return styleTransferFunctional.create_style_transfer_model(input_shape['content'])


style_transfer_models = styleTransferTrainingModel.make_style_transfer_training_model(
    input_shape,
    build_style_prediction_model,
    build_style_transfer_model,
    build_style_loss_function
)

element = {
    'content': tf.convert_to_tensor(np.zeros((1, 960, 1920, 3))),
    'style': tf.convert_to_tensor(np.zeros((1, 960, 1920, 3))),
}
log.info(f"Running inference to build model...")
# call once to build models
style_transfer_models.training(element)
# log.info(f"Loading weights...")
# style_transfer_model.load_weights(filepath=str(checkpoint_path))

log.info(f"Saving model...")
style_transfer_models.transfer.save(filepath=str(outpath.with_suffix(".transfer.tf")), include_optimizer=False,
                                    save_format='tf')
style_transfer_models.style_predictor.save(filepath=str(outpath.with_suffix(".predictor.tf")), include_optimizer=False,
                                           save_format='tf')
