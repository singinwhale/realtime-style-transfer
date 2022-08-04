import datetime

import dataloaders.common
import logsetup
from pathlib import Path

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
from tensorflow import keras

import logging

log = logging.getLogger()

log_dir = Path(__file__).parent / 'logs' / str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f"))
log_dir.mkdir(exist_ok=True, parents=True, )

from dataloaders import wikiart
from models import styleTransfer, stylePrediction, styleLoss
from tf_image_callback import SummaryImageCallback
from renderers.matplotlib import predict_datapoint

input_shape = {'content': (None, 960 // 2, 1920 // 2, 3), 'style': (None, 960 // 2, 1920 // 2, 3)}
output_shape = (None, 960 // 2, 1920 // 2, 3)

# with tf.profiler.experimental.Profile(str(log_dir)):
training_dataset, validation_dataset = wikiart.get_dataset_debug(input_shape, batch_size=3)

validation_log_datapoint = dataloaders.common.get_single_sample_from_dataset(validation_dataset)
training_log_datapoint = dataloaders.common.get_single_sample_from_dataset(training_dataset)
image_callback = SummaryImageCallback(validation_log_datapoint, training_log_datapoint)

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
    predict_datapoint(validation_log_datapoint, training_log_datapoint, style_transfer_model)
    style_transfer_model.fit(x=training_dataset, validation_data=validation_dataset, epochs=10,
                             callbacks=[tb_callback, image_callback])
    style_transfer_model.save_weights(log_dir / "last_training_checkpoint")
    predict_datapoint(validation_log_datapoint, training_log_datapoint, style_transfer_model)
