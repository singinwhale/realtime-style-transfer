import tracing.logsetup

import datetime

import dataloaders.common
from pathlib import Path

import tensorflow as tf

from tracing.checkpoint import CheckpointCallback
from tracing.histogram import HistogramCallback

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

tf.debugging.disable_traceback_filtering()

import logging

log = logging.getLogger()

cache_root_dir = Path(__file__).parent / 'cache'
log_root_dir = Path(__file__).parent / 'logs'
log_dir = log_root_dir / str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f"))
log_dir.mkdir(exist_ok=True, parents=True, )

from dataloaders import wikiart
from models import stylePrediction, styleLoss, styleTransferFunctional, styleTransferTrainingModel
from tracing.tf_image_callback import SummaryImageCallback
from renderers.matplotlib import predict_datapoint

resolution_divider = 2
input_shape = {'content': (960 // resolution_divider, 1920 // resolution_divider, 3),
               'style': (960 // resolution_divider, 1920 // resolution_divider, 3)}
output_shape = (960 // resolution_divider, 1920 // resolution_divider, 3)

# with tf.profiler.experimental.Profile(str(log_dir)):
# training_dataset, validation_dataset = wikiart.get_dataset_debug(input_shape, batch_size=4)
training_dataset, validation_dataset = wikiart.get_dataset(input_shape, batch_size=16, cache_dir=cache_root_dir, seed=347890842)

cache_root_dir.mkdir(exist_ok=True)

validation_log_datapoint = dataloaders.common.get_single_sample_from_dataset(validation_dataset)
training_log_datapoint = dataloaders.common.get_single_sample_from_dataset(training_dataset)
image_callback = SummaryImageCallback(validation_log_datapoint, training_log_datapoint)
checkpoint_callback = CheckpointCallback(log_dir / "checkpoints", cadence=10)
histogram_callback = HistogramCallback()

# tf.debugging.enable_check_numerics()
summary_writer = tf.summary.create_file_writer(logdir=str(log_dir))

with summary_writer.as_default() as summary:
    style_loss_model = styleLoss.StyleLossModelMobileNet(output_shape)
    style_transfer_training_model = styleTransferTrainingModel.make_style_transfer_training_model(
        input_shape,
        style_predictor_factory_func=lambda num_top_parameters: stylePrediction.create_style_prediction_model(
            input_shape['style'], stylePrediction.StyleFeatureExtractor.MOBILE_NET, num_top_parameters
        ),
        style_transfer_factory_func=lambda: styleTransferFunctional.create_style_transfer_model(input_shape['content']),
        style_loss_func_factory_func=lambda: styleLoss.make_style_loss_function(style_loss_model),
    )

    style_transfer_training_model.training.compile()

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir))
    predict_datapoint(validation_log_datapoint, training_log_datapoint, style_transfer_training_model.training,
                      callbacks=[histogram_callback])
    style_transfer_training_model.training.fit(x=training_dataset, validation_data=validation_dataset, epochs=300,
                                      callbacks=[tb_callback, image_callback, checkpoint_callback, histogram_callback])
    predict_datapoint(validation_log_datapoint, training_log_datapoint, style_transfer_training_model.training,
                      callbacks=[histogram_callback])
