import logsetup

import tensorflow as tf
from tensorflow import keras

import logging

log = logging.getLogger()

from dataloaders import wikiart

from models import styleTransfer, stylePrediction, styleLoss

input_shape = {'content': (None, 1920, 960, 3), 'style': (None, 1920, 960, 3)}
output_shape = (None, 1920, 960, 3)

training_dataset, validation_dataset = wikiart.get_dataset_debug(input_shape)

style_loss_model = styleLoss.StyleLossModelEfficientNet(output_shape)
style_transfer_model = styleTransfer.StyleTransferModel(input_shape,
                                                        lambda batchnorm_layers: stylePrediction.StylePredictionModel(
                                                            input_shape, batchnorm_layers),
                                                        lambda x, y_pred: styleLoss.style_loss(style_loss_model, x, y_pred))

style_transfer_model.compile()

style_transfer_model.fit(training_dataset, epochs=2)
log.warning(style_transfer_model.evaluate(validation_dataset))
