import sys
import timeit

import matplotlib.pyplot as plt
import numpy as np
import shap
import tqdm

import dataloaders.common
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


def input_dict_to_list(input_dict: dict):
    input_list = [0 for i in range(len(input_dict))]
    input_keys = {n: int(i[-1]) - 1 for n, i in zip(style_transfer_training_model.training.input.keys(),
                                                    style_transfer_training_model.training.input_names)}
    for name, value in input_dict.items():
        index = input_keys[name]
        input_list[index] = value

    return input_list


validation_dataset_iter = iter(validation_dataset)
test_sample = input_dict_to_list(next(validation_dataset_iter))
background_sample = input_dict_to_list(next(validation_dataset_iter))


shap_explainer = shap.DeepExplainer(
    (style_transfer_training_model.training.inputs, style_transfer_training_model.loss_model.output['loss']),
    background_sample)
shap_explanation = shap_explainer.shap_values(test_sample)

print(shap_explanation)
