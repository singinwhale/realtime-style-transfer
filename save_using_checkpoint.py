import shutil

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

args = argparser.parse_args()
checkpoint_path: Path = args.checkpoint_path
with_tensorflow: bool = args.tensorflow
with_onnx: bool = args.onnx
outpath: Path = args.outpath

image_shape = (960, 1920, 3)
num_styles = 2
style_weights_shape = (960, 1920, num_styles - 1)
styles_shape = (num_styles,) + image_shape

log = logging.getLogger()

tf.config.set_visible_devices([], 'GPU')

from realtime_style_transfer.models import styleTransfer, stylePrediction, styleTransferInferenceModel

input_shape = {
    'content': image_shape,
    'style_weights': style_weights_shape,
    'style': styles_shape,
}
output_shape = image_shape


def build_style_prediction_model(batchnorm_layers):
    # return stylePrediction.StylePredictionModelMobileNet(input_shape, batchnorm_layers)
    return stylePrediction.create_style_prediction_model(
        image_shape,
        stylePrediction.StyleFeatureExtractor.MOBILE_NET,
        batchnorm_layers,
    )


num_style_norm_params = None


def build_style_transfer_model():
    global num_style_norm_params
    transfer_model_data = styleTransfer.create_style_transfer_model(input_shape['content'], num_styles)
    num_style_norm_params = transfer_model_data[1]
    return transfer_model_data


style_transfer_models = styleTransferInferenceModel.make_style_transfer_inference_model(
    input_shape,
    build_style_prediction_model,
    build_style_transfer_model
)

element = {
    'content': tf.convert_to_tensor(np.zeros((1,) + image_shape)),
    'style_weights': tf.convert_to_tensor(np.zeros((1,) + style_weights_shape)),
    'style': tf.convert_to_tensor(np.zeros((1,) + styles_shape)),
}
log.info(f"Running inference to build model...")
# call once to build models
style_transfer_models.inference(element)
log.info(f"Loading weights...")
style_transfer_models.inference.load_weights(filepath=str(checkpoint_path))

predictor_path = outpath.with_suffix(".predictor.tf")
transfer_path = outpath.with_suffix(".transfer.tf")

if with_tensorflow:
    log.info(f"Saving model as tensorflow...")
    style_transfer_models.transfer.save(filepath=str(transfer_path), include_optimizer=False,
                                        save_format='tf')
    predictor_path = outpath.with_suffix(".predictor.tf")
    style_transfer_models.style_predictor.save(filepath=str(predictor_path), include_optimizer=False,
                                               save_format='tf')

if with_onnx:
    log.info("Saving style predictor model as ONNX...")
    tf2onnx.convert.from_keras(style_transfer_models.style_predictor,
                               [tf.TensorSpec((None,) + image_shape, name='style')],
                               output_path=predictor_path.with_suffix('.onnx'))

    log.info("Saving transfer model as ONNX...")
    tf2onnx.convert.from_keras(style_transfer_models.transfer,
                               [
                                   tf.TensorSpec((None,) + image_shape, name='content'),
                                   tf.TensorSpec((None, num_styles, num_style_norm_params), name='style_params'),
                                   tf.TensorSpec((None,) + style_weights_shape, name='style_weights'),
                               ], output_path=transfer_path.with_suffix('.onnx'))
log.info("Saving checkpoint...")
checkpoint_outdir = outpath.with_suffix(".checkpoint")
checkpoint_outdir.mkdir(exist_ok=True)
for checkpoint_file in checkpoint_path.parent.glob(f"{checkpoint_path.stem}*"):
    shutil.copy(checkpoint_file, checkpoint_outdir / Path("checkpoint").with_suffix(checkpoint_file.suffix))
log.info("Done")
