import sys

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from tracing import logsetup

from pathlib import Path
import tensorflow as tf
import logging
import argparse

from dataloaders import common

log = logging.getLogger()

argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint_path', '-C', type=Path, required=True)

args = argparser.parse_args()
checkpoint_path = args.checkpoint_path

from shape_config import *

num_styles = 1
config = ShapeConfig(hdr=True, num_styles=num_styles)

from models import styleTransfer, stylePrediction, styleTransferTrainingModel, styleLoss
from dataloaders import wikiart

style_transfer_training_model = styleTransferTrainingModel.make_style_transfer_training_model(
    config.input_shape,
    style_predictor_factory_func=lambda num_top_parameters: stylePrediction.create_style_prediction_model(
        config.input_shape['style'][1:], stylePrediction.StyleFeatureExtractor.MOBILE_NET, num_top_parameters
    ),
    style_transfer_factory_func=lambda: styleTransfer.create_style_transfer_model(config.input_shape['content'],
                                                                                  num_styles=num_styles),
    style_loss_func_factory_func=lambda: styleLoss.make_style_loss_function(
        styleLoss.StyleLossModelVGG(config.output_shape),
        config.input_shape, config.output_shape)
)

log.info("Loading dataset...")
training_dataset, validation_dataset = wikiart.get_hdr_dataset(config.input_shape, 1, channels=config.channels,
                                                               seed=278992)

# call once to build model
log.info("Compiling model...")
style_transfer_training_model.training.compile(run_eagerly=False)
log.info("Building model...")
style_transfer_training_model.training.build(config.input_shape)
log.info(f"Loading weights from {checkpoint_path}")
style_transfer_training_model.training.load_weights(filepath=str(checkpoint_path))

channel_contributions = {n: 0.0 for n, c in config.channels}


def get_gradients(sample):
    with tf.GradientTape() as tape:
        tape.watch(sample)
        losses = style_transfer_training_model.loss_model(sample, training=True)
        grads = tape.gradient(losses["loss"], sample)

    return grads


num_samples = 0
validation_progress = tqdm.tqdm(validation_dataset, file=sys.stdout, total=validation_dataset.num_samples, ascii=False)
for sample in validation_progress:
    gradients = get_gradients(sample)
    content_gradients = gradients['content']
    channel_lower_bound = 0
    status = []
    for channel, num_components in config.channels:
        channel_upper_bound = channel_lower_bound + num_components
        total_influence = tf.reduce_mean(tf.abs(content_gradients[..., channel_lower_bound:channel_upper_bound]))
        channel_lower_bound = channel_upper_bound
        channel_contributions[channel] += total_influence
        status.append(f"{channel}({num_components}): {total_influence:0.05f}")

    validation_progress.write(', '.join(status))
    num_samples += 1


for channel, contribution in channel_contributions.items():
    channel_contributions[channel] = contribution / num_samples

channels_by_contributions = sorted(channel_contributions.items(), key=lambda i: i[1], reverse=True)

for channel, contribution in channels_by_contributions:
    log.info(f"{channel}: {contribution}")


