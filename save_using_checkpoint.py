import shutil

from tracing import logsetup

import numpy as np

from pathlib import Path
import tensorflow as tf
import logging
import argparse
import tf2onnx

argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint_path', '-C', type=Path, required=True)
argparser.add_argument('--outpath', '-o', type=Path, required=True)

args = argparser.parse_args()
checkpoint_path:Path = args.checkpoint_path
outpath: Path = args.outpath

image_shape = (960, 1920, 3)

log = logging.getLogger()

tf.config.set_visible_devices([], 'GPU')

from models import styleTransfer, stylePrediction, styleLoss, styleTransferTrainingModel

input_shape = {'content': image_shape, 'style': image_shape}
output_shape = image_shape


def build_style_loss_function():
    style_loss_model = styleLoss.StyleLossModelMobileNet(output_shape)
    return styleLoss.make_style_loss_function(style_loss_model, input_shape, output_shape)


def build_style_prediction_model(batchnorm_layers):
    # return stylePrediction.StylePredictionModelMobileNet(input_shape, batchnorm_layers)
    return stylePrediction.create_style_prediction_model(
        input_shape['style'],
        stylePrediction.StyleFeatureExtractor.MOBILE_NET,
        batchnorm_layers,
    )


num_style_norm_params = None


def build_style_transfer_model():
    global num_style_norm_params
    transfer_model_data = styleTransfer.create_style_transfer_model(input_shape['content'])
    num_style_norm_params = transfer_model_data[1]
    return transfer_model_data


style_transfer_models = styleTransferTrainingModel.make_style_transfer_training_model(
    input_shape,
    build_style_prediction_model,
    build_style_transfer_model,
    build_style_loss_function
)

element = {
    'content': tf.convert_to_tensor(np.zeros((1, 960, 1920, 3))),
    'style': tf.convert_to_tensor(np.zeros((1, 960, 1920, 3))),
}
log.info(f"Running inference to build model...")
# call once to build models
style_transfer_models.training(element)
log.info(f"Loading weights...")
style_transfer_models.training.load_weights(filepath=str(checkpoint_path))

log.info(f"Saving model...")
transfer_path = outpath.with_suffix(".transfer.tf")
style_transfer_models.transfer.save(filepath=str(transfer_path), include_optimizer=False,
                                    save_format='tf')
predictor_path = outpath.with_suffix(".predictor.tf")
style_transfer_models.style_predictor.save(filepath=str(predictor_path), include_optimizer=False,
                                           save_format='tf')

log.info("Saving style predictor model as ONNX...")
tf2onnx.convert.from_keras(style_transfer_models.style_predictor,
                           [tf.TensorSpec((None,) + image_shape, name='style')],
                           output_path=predictor_path.with_suffix('.onnx'))

log.info("Saving transfer model as ONNX...")
tf2onnx.convert.from_keras(style_transfer_models.transfer,
                           [
                               tf.TensorSpec((None,) + image_shape, name='content'),
                               tf.TensorSpec((None, num_style_norm_params), name='style_params')
                           ], output_path=transfer_path.with_suffix('.onnx'))
log.info("Saving checkpoint...")
checkpoint_outdir = outpath.with_suffix(".checkpoint")
checkpoint_outdir.mkdir(exist_ok=True)
for checkpoint_file in checkpoint_path.parent.glob(f"{checkpoint_path.stem}*"):
    shutil.copy(checkpoint_file, checkpoint_outdir / Path("checkpoint").with_suffix(checkpoint_file.suffix))
log.info("Done")
