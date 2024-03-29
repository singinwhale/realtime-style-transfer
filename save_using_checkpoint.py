import shutil

from realtime_style_transfer.models import styleLoss, styleTransferTrainingModel
from realtime_style_transfer.shape_config import ShapeConfig
import numpy as np

from pathlib import Path
import tensorflow as tf
import logging
import argparse
import tf2onnx

argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint_path', '-C', type=Path, required=True)
argparser.add_argument('--tensorflow', '-t', action='store_true')
argparser.add_argument('--onnx', '-x', action='store_true')
argparser.add_argument('--outpath', '-o', type=Path, required=True)
argparser.add_argument('--network_spec', '-n', type=str, required=False)

args = argparser.parse_args()
checkpoint_path: Path = args.checkpoint_path
with_tensorflow: bool = args.tensorflow
with_onnx: bool = args.onnx
outpath: Path = args.outpath
network_spec: str = args.network_spec

log = logging.getLogger()

# tf.config.set_visible_devices([], 'GPU')
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

from realtime_style_transfer.models import styleTransfer, stylePrediction, styleTransferInferenceModel

if network_spec:
    config = ShapeConfig.from_spec(network_spec, 1, True)
else:
    config = ShapeConfig(hdr=True, num_styles=1)

style_loss_model = styleLoss.StyleLossModelMobileNet(config.output_shape)

style_transfer_training_model = styleTransferTrainingModel.make_style_transfer_training_model(
    style_predictor_factory_func=lambda num_top_parameters: stylePrediction.create_style_prediction_model(
        config.input_shape['style'][1:], config.style_feature_extractor_type, num_top_parameters
    ),
    style_transfer_factory_func=lambda: styleTransfer.create_style_transfer_model(
        config.input_shape['content'],
        config.output_shape, config.bottleneck_res_y, config.bottleneck_num_filters, config.num_styles
    ),
    style_loss_func_factory_func=lambda: styleLoss.make_style_loss_function(
        style_loss_model,
        config.output_shape,
        config.num_styles),
)

element = {input_name: (tf.zeros((1,) + input_shape)) for input_name, input_shape in config.input_shape.items()}
ground_truth_element = {
    'content': tf.zeros((1,) + config.output_shape),
    'style': tf.zeros((1, config.num_styles) + config.output_shape)
}

log.info(f"Running inference to build model...")
# call once to build models
style_transfer_training_model.training(element)
log.info(f"Loading Checkpoint...")
checkpoint = tf.train.Checkpoint(style_transfer_training_model.training)
load_status = checkpoint.restore(str(checkpoint_path))
# load_status = style_transfer_training_model.inference.load_weights(filepath=str(checkpoint_path))
load_status.assert_nontrivial_match()

style_transfer_training_model.training(element)
style_transfer_training_model.loss_model((element, ground_truth_element))

predictor_path = outpath.with_suffix(".predictor.tf")
transfer_path = outpath.with_suffix(".transfer.tf")

if with_tensorflow:
    log.info(f"Saving model as tensorflow...")
    style_transfer_training_model.transfer.save(filepath=str(transfer_path), include_optimizer=False,
                                                save_format='tf')
    predictor_path = outpath.with_suffix(".predictor.tf")
    style_transfer_training_model.style_predictor.save(filepath=str(predictor_path), include_optimizer=False,
                                                       save_format='tf')
    style_transfer_training_model.loss_model.save(filepath=str(outpath.with_suffix('.loss.tf')),
                                                  include_optimizer=False,
                                                  save_format='tf')
    style_transfer_training_model.training.save(filepath=str(outpath.with_suffix('.training.tf')),
                                                  include_optimizer=True,
                                                  save_format='tf')

if with_onnx:
    log.info("Saving style predictor model as ONNX...")
    tf2onnx.convert.from_keras(style_transfer_training_model.style_predictor,
                               [tf.TensorSpec(style_transfer_training_model.style_predictor.input_shape, name='style')],
                               output_path=predictor_path.with_suffix('.onnx'))

    input_to_spec = {input_tensor.name: tf.TensorSpec(input_tensor.shape, name=name) for name, input_tensor in
                     style_transfer_training_model.transfer.input.items()}

    log.info("Saving transfer model as ONNX...")
    tf2onnx.convert.from_keras(style_transfer_training_model.transfer,
                               [input_to_spec[input_name] for input_name in
                                style_transfer_training_model.transfer.input_names],
                               output_path=transfer_path.with_suffix('.onnx'))
log.info("Saving checkpoint...")
checkpoint_outdir = outpath.with_suffix(".checkpoint")
checkpoint_outdir.mkdir(exist_ok=True)
for checkpoint_file in checkpoint_path.parent.glob(f"{checkpoint_path.stem}*"):
    shutil.copy(checkpoint_file, checkpoint_outdir / Path("checkpoint").with_suffix(checkpoint_file.suffix))
log.info("Done")
