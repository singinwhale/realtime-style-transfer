import timeit

import matplotlib.pyplot as plt
import numpy as np

import dataloaders.common
from tracing import logsetup

from pathlib import Path
import tensorflow as tf
import logging
import argparse

from dataloaders import common

log = logging.getLogger()

argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint_path', '-C', type=Path, required=True)
argparser.add_argument('--style_image_path', '-s', type=Path, action='append')
argparser.add_argument('--content_image_path', '-c', type=Path, required=True)
argparser.add_argument('--style_weights_path', '-w', type=Path)
argparser.add_argument('--outpath', '-o', type=Path, required=False)

args = argparser.parse_args()
checkpoint_path = args.checkpoint_path
style_image_paths = args.style_image_path
content_image_path = args.content_image_path
style_weights_path = args.style_weights_path
outpath = args.outpath

image_shape = (960, 1920, 3)
num_styles = len(style_image_paths)
style_weights_shape = (960, 1920, num_styles - 1)

if num_styles > 1:
    if style_weights_path is None:
        style_weights = tf.convert_to_tensor(
            np.triu(np.ones(style_weights_shape[0:2]), 480).reshape((1,) + style_weights_shape))
    else:
        style_weights = dataloaders.common.image_dataset_from_filepaths([style_weights_path], style_weights_shape)
        style_weights = style_weights.batch(1).get_single_element()
else:
    style_weights = tf.ones((1,) + style_weights_shape)

element = {
    'content': common.image_dataset_from_filepaths([content_image_path], image_shape).batch(1).get_single_element(),
    'style_weights': style_weights,
    'style': tf.stack(list(common.image_dataset_from_filepaths(style_image_paths, image_shape).batch(1)), axis=1)
}

from models import styleTransfer, stylePrediction, styleTransferInferenceModel
from renderers.matplotlib import predict_datapoint

input_shape = {
    'content': image_shape,
    'style_weights': style_weights_shape,
    'style': (num_styles,) + image_shape,
}
output_shape = image_shape

style_transfer_inference_model = styleTransferInferenceModel.make_style_transfer_inference_model(
    input_shape,
    style_predictor_factory_func=lambda num_top_parameters: stylePrediction.create_style_prediction_model(
        input_shape['style'][1:], stylePrediction.StyleFeatureExtractor.MOBILE_NET, num_top_parameters
    ),
    style_transfer_factory_func=lambda: styleTransfer.create_style_transfer_model(input_shape['content'],
                                                                                  num_styles=num_styles),
)

# call once to build model
style_transfer_inference_model.inference(element)
log.info(f"Loading weights from {checkpoint_path}")
style_transfer_inference_model.inference.load_weights(filepath=str(checkpoint_path))
predict_datapoint(element, element, style_transfer_inference_model.inference)


def do_inference():
    style_transfer_inference_model.inference.predict(element)


results = timeit.repeat(do_inference, repeat=20, number=1)
log.info(f"Fastest time was {min(results)*1000:.2f}ms average = {sum(results) / len(results) * 1000:.2f}ms")

# save result if required
if outpath is not None:
    result = style_transfer_inference_model.training(element)
    image_data = result.numpy().squeeze()
    tf.keras.utils.save_img(outpath, image_data, data_format='channels_last')
