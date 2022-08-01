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
from renderers.matplotlib import predict_datapoint

input_shape = {'content': (None, 960 // 2, 1920 // 2, 3), 'style': (None, 960 // 2, 1920 // 2, 3)}
output_shape = (None, 960 // 2, 1920 // 2, 3)

#with tf.profiler.experimental.Profile(str(log_dir)):
training_dataset, validation_dataset = wikiart.get_dataset_debug(input_shape)

for datapoint in validation_dataset.shuffle(buffer_size=10, seed=373893289).unbatch().batch(1):
    log_datapoint = datapoint
    image_callback = SummaryImageCallback(log_datapoint)
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
    predict_datapoint(log_datapoint, style_transfer_model)
    style_transfer_model.fit(x=training_dataset, validation_data=validation_dataset, epochs=20,
                             callbacks=[tb_callback, image_callback])
    predict_datapoint(log_datapoint, style_transfer_model)
