import typing
import os
from pathlib import Path

os.environ['PATH'] += r";C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.1.3\target-windows-x64"
os.environ['PATH'] += r";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\extras\CUPTI\lib64"

import datetime

import dataloaders.common

import tensorflow as tf

physical_devices: typing.List[tf.config.PhysicalDevice] = tf.config.list_physical_devices('GPU')
try:
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=23 * 1024)]
    )
finally:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# tf.debugging.disable_traceback_filtering()
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

import logging
import tracing

log = logging.getLogger()

cache_root_dir = Path(__file__).parent / 'cache'
log_root_dir = Path(__file__).parent / 'logs'
log_dir = log_root_dir / str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f"))
log_dir.mkdir(exist_ok=True, parents=True, )
tracing.logsetup.enable_logfile(log_dir)

from dataloaders import wikiart
from models import stylePrediction, styleLoss, styleTransfer, styleTransferTrainingModel
from tracing.tf_image_callback import SummaryImageCallback
from renderers.matplotlib import predict_datapoint
from tracing.textSummary import capture_model_summary
from tracing.checkpoint import CheckpointCallback
from tracing.histogram import HistogramCallback, write_model_histogram_summary
from tracing.gradients import GradientsCallback

resolution_divider = 2
num_styles = 1
input_shape = {'content': (960 // resolution_divider, 1920 // resolution_divider, 3),
               'style_weights': (960 // resolution_divider, 1920 // resolution_divider, num_styles - 1),
               'style': (num_styles, 960 // resolution_divider, 1920 // resolution_divider, 3)}
output_shape = (960 // resolution_divider, 1920 // resolution_divider, 3)

# training_dataset, validation_dataset = wikiart.get_dataset_debug(input_shape, batch_size=4)
training_dataset, validation_dataset = wikiart.get_dataset(input_shape, batch_size=8,
                                                           cache_dir=cache_root_dir, seed=347890842)

cache_root_dir.mkdir(exist_ok=True)

validation_log_datapoint = dataloaders.common.get_single_sample_from_dataset(validation_dataset)
training_log_datapoint = dataloaders.common.get_single_sample_from_dataset(training_dataset)
image_callback = SummaryImageCallback(validation_log_datapoint, training_log_datapoint)
checkpoint_callback = CheckpointCallback(log_dir / "checkpoints", cadence=10)
histogram_callback = HistogramCallback()
gradients_callback = GradientsCallback(training_log_datapoint)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir),
                                                      histogram_freq=5,
                                                      write_graph=False,
                                                      profile_batch=0, )

# tf.debugging.enable_check_numerics()
summary_writer = tf.summary.create_file_writer(logdir=str(log_dir))

with summary_writer.as_default() as summary:
    style_loss_model = styleLoss.StyleLossModelVGG(output_shape)
    style_transfer_training_model = styleTransferTrainingModel.make_style_transfer_training_model(
        input_shape,
        style_predictor_factory_func=lambda num_top_parameters: stylePrediction.create_style_prediction_model(
            input_shape['style'][1:], stylePrediction.StyleFeatureExtractor.MOBILE_NET, num_top_parameters
        ),
        style_transfer_factory_func=lambda: styleTransfer.create_style_transfer_model(input_shape['content'],
                                                                                      num_styles=num_styles),
        style_loss_func_factory_func=lambda: styleLoss.make_style_loss_function(style_loss_model, input_shape,
                                                                                output_shape),
    )

    style_transfer_training_model.training.compile(run_eagerly=False, optimizer='adam')

    latest_epoch_weights_path = log_root_dir / "2022-09-14-19-01-08.839135" / "checkpoints" / "latest_epoch_weights"
    log.info(f"Loading weights from {latest_epoch_weights_path}")
    try:
        style_transfer_training_model.training.load_weights(latest_epoch_weights_path)
    except Exception as e:
        log.warning(f"Could not load weights: {e}")

    summary_text = capture_model_summary(style_transfer_training_model.training)
    tf.summary.text('summary', f"```\n{summary_text}\n```", -1)
    summary_text = capture_model_summary(style_transfer_training_model.training, detailed=True)
    tf.summary.text('summary_detailed', f"```\n{summary_text}\n```", -1)

    # write_model_histogram_summary(style_transfer_training_model.training, -1)
    # with tf.profiler.experimental.Profile(str(log_dir)) as profiler:
    style_transfer_training_model.training.fit(x=training_dataset.prefetch(5), validation_data=validation_dataset,
                                               epochs=500,
                                               callbacks=[  # tensorboard_callback,
                                                   image_callback,
                                                   checkpoint_callback,
                                                   # histogram_callback,
                                               ])
    predict_datapoint(validation_log_datapoint, training_log_datapoint, style_transfer_training_model.training,
                      callbacks=[histogram_callback])

log.info("Finished successfully")
