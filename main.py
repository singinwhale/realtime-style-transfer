import logsetup

import tensorflow as tf
from tensorflow import keras

import logging

log = logging.getLogger()

from dataloaders import wikiart

from models import styleTransfer, stylePrediction, styleLoss

input_shape = (None, 2, 1920, 960, 3)
output_shape = (None,) + input_shape[-3:]

training_dataset, validation_dataset = wikiart.get_dataset_debug(input_shape[1:3])

style_transfer_model = styleTransfer.StyleTransferModel(input_shape,
                                                        lambda batchnorm_layers: stylePrediction.StylePredictionModel(
                                                            input_shape, batchnorm_layers))

style_loss_model = styleLoss.StyleLossModelEfficientNet(output_shape)
style_loss_model.compile()
style_loss_model.build(output_shape)

style_transfer_model.compile(
    loss=lambda content_and_style_inputs, style_transfer_results: styleLoss.style_loss(style_loss_model,
                                                                                       inputs=content_and_style_inputs,
                                                                                       outputs=style_transfer_results))
style_transfer_model.build(input_shape)

style_transfer_model.fit(training_dataset, epochs=2)
log.warning(style_transfer_model.evaluate(validation_dataset))
