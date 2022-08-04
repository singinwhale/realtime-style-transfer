import datetime

import logsetup

import os

from pathlib import Path

import tensorflow as tf

import logging

log = logging.getLogger()

from models import styleTransfer, stylePrediction, styleLoss
from renderers.matplotlib import predict_datapoint

input_shape = {'content': (None, 960 // 2, 1920 // 2, 3), 'style': (None, 960 // 2, 1920 // 2, 3)}
output_shape = (None, 960 // 2, 1920 // 2, 3)

style_loss_model = styleLoss.StyleLossModelMobileNet(output_shape)
style_transfer_model = styleTransfer.StyleTransferModel(
    input_shape,
    lambda batchnorm_layers: stylePrediction.StylePredictionModelMobileNet(
        input_shape, batchnorm_layers),
    lambda x, y_pred: styleLoss.style_loss(style_loss_model, x, y_pred)
)

style_transfer_model.load_weights()
predict_datapoint(log_datapoint, style_transfer_model)
