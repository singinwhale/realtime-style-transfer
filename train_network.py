import typing
import os
from pathlib import Path

os.environ['PATH'] += r";C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.1.3\target-windows-x64"
os.environ['PATH'] += r";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\extras\CUPTI\lib64"

import datetime

import realtime_style_transfer.dataloaders as dataloaders

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
import realtime_style_transfer.tracing as tracing

log = logging.getLogger()

continue_from = None
# continue_from = ("2022-10-19-13-37-01.057203", 162)

cache_root_dir = Path(__file__).parent / 'cache'
cache_root_dir.mkdir(exist_ok=True)
log_root_dir = Path(__file__).parent / 'logs'
log_dir = log_root_dir / (
    str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")) if continue_from is None else continue_from[0])
log_dir.mkdir(exist_ok=True, parents=True, )
tracing.logsetup.enable_logfile(log_dir)

from realtime_style_transfer.dataloaders import wikiart
from realtime_style_transfer.models import stylePrediction, styleLoss, styleTransfer, styleTransferTrainingModel
from realtime_style_transfer.tracing.tf_image_callback import SummaryImageCallback
from realtime_style_transfer.renderers.matplotlib import predict_datapoint
from realtime_style_transfer.tracing.textSummary import capture_model_summary
from realtime_style_transfer.tracing.checkpoint import CheckpointCallback
from realtime_style_transfer.tracing.histogram import HistogramCallback
from realtime_style_transfer.tracing.gradients import GradientsCallback
from realtime_style_transfer.tracing.metrics import MetricsCallback

from realtime_style_transfer.shape_config import ShapeConfig

config = ShapeConfig(hdr=True, num_styles=1)
# training_dataset, validation_dataset = wikiart.get_dataset_debug(input_shape, batch_size=4)

# training_dataset, validation_dataset = wikiart.get_dataset_debug(input_shape, batch_size=8,
#                                                           cache_dir=cache_root_dir, seed=347890842)
training_dataset, validation_dataset = wikiart.get_hdr_dataset_debug(shapes=config.input_shape,
                                                                     output_shape=config.output_shape,
                                                                     batch_size=2,
                                                                     cache_dir=cache_root_dir,
                                                                     seed=34789082,
                                                                     channels=config.channels)

validation_log_datapoint = dataloaders.common.get_single_sample_from_dataset(validation_dataset)
training_log_datapoint = dataloaders.common.get_single_sample_from_dataset(training_dataset)
image_callback = SummaryImageCallback(validation_log_datapoint, training_log_datapoint)
checkpoint_callback = CheckpointCallback(log_dir / "checkpoints", cadence=10)
histogram_callback = HistogramCallback()
metrics_callback = MetricsCallback(log_dir)
gradients_callback = GradientsCallback(training_log_datapoint)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir),
                                                      update_freq=1,
                                                      histogram_freq=0,
                                                      write_graph=False,
                                                      profile_batch=0, )

# tf.debugging.enable_check_numerics()
summary_writer = tf.summary.create_file_writer(logdir=str(log_dir))

with summary_writer.as_default() as summary:
    tf.summary.text("config", str(config), step=-1)
    style_loss_model = styleLoss.StyleLossModelVGG(config.output_shape)
    style_transfer_training_model = styleTransferTrainingModel.make_style_transfer_training_model(
        style_predictor_factory_func=lambda num_top_parameters: stylePrediction.create_style_prediction_model(
            config.input_shape['style'][1:], stylePrediction.StyleFeatureExtractor.MOBILE_NET, num_top_parameters
        ),
        style_transfer_factory_func=lambda: styleTransfer.create_style_transfer_model(
            input_shape=config.input_shape['content'],
            output_shape=config.output_shape,
            bottleneck_res_y=config.bottleneck_res_y,
            bottleneck_num_filters=config.bottleneck_num_filters,
            num_styles=config.num_styles),
        style_loss_func_factory_func=lambda: styleLoss.make_style_loss_function(style_loss_model,
                                                                                config.output_shape,
                                                                                config.num_styles,),
    )

    style_transfer_training_model.training.compile(run_eagerly=True)  # True for Debugging, False for performance

    if continue_from is not None:
        latest_epoch_weights_path = log_root_dir / continue_from[0] / "checkpoints" / "latest_epoch_weights"
        log.info(f"Loading weights from {latest_epoch_weights_path}")
        try:
            style_transfer_training_model.training.load_weights(latest_epoch_weights_path)
            pass
        except Exception as e:
            log.warning(f"Could not load weights: {e}")

    summary_text = capture_model_summary(style_transfer_training_model.training)
    tf.summary.text('summary', f"```\n{summary_text}\n```", -1)
    summary_text = capture_model_summary(style_transfer_training_model.training, detailed=True)
    tf.summary.text('summary_detailed', f"```\n{summary_text}\n```", -1)

    # write_model_histogram_summary(style_transfer_training_model.training, -1)
    # with tf.profiler.experimental.Profile(str(log_dir)) as profiler:
    style_transfer_training_model.training.fit(x=training_dataset.prefetch(5), validation_data=validation_dataset,
                                               epochs=300,
                                               callbacks=[  # tensorboard_callback,
                                                   image_callback,
                                                   checkpoint_callback,
                                                   metrics_callback,
                                                   # histogram_callback,
                                               ])
    predict_datapoint(validation_log_datapoint, training_log_datapoint, style_transfer_training_model.training,
                      callbacks=[histogram_callback])

log.info("Finished successfully")
