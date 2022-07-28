import logsetup

import os

os.environ['Path'] += r";C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.1.3\target-windows-x64"

from pathlib import Path

import tensorflow as tf
from tensorflow import keras

import logging

log = logging.getLogger()

log_dir = Path(__file__).parent / 'logs'

from dataloaders import wikiart
from models import styleTransfer, stylePrediction, styleLoss
from tf_image_callback import SummaryImageCallback

input_shape = {'content': (None, 1920//2, 960//2, 3), 'style': (None, 1920//2, 960//2, 3)}
output_shape = (None, 1920//2, 960//2, 3)

with tf.profiler.experimental.Profile(str(log_dir)):
    training_dataset, validation_dataset = wikiart.get_dataset_debug(input_shape)

    for datapoint in validation_dataset:
        image_callback = SummaryImageCallback(datapoint)
        break
    summary_writer = tf.summary.create_file_writer(logdir=str(log_dir))
    with summary_writer.as_default() as summary:

            style_loss_model = styleLoss.StyleLossModelMobileNet(output_shape)
            style_transfer_model = styleTransfer.StyleTransferModel(
                input_shape,
                lambda batchnorm_layers: stylePrediction.StylePredictionModelMobileNet(
                    input_shape, batchnorm_layers),
                lambda x, y_pred: styleLoss.style_loss(style_loss_model, x, y_pred)
            )

            style_transfer_model.compile()

            tb_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir))
            style_transfer_model.fit(x=training_dataset, validation_data=validation_dataset, epochs=2, callbacks=[tb_callback, image_callback])