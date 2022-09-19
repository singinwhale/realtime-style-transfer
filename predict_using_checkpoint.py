from tracing import logsetup

from pathlib import Path
import tensorflow as tf
import logging
import argparse

from dataloaders import common

argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint_path', '-C', type=Path, required=True)
argparser.add_argument('--style_image_path', '-s', type=Path, action='append')
argparser.add_argument('--content_image_path', '-c', type=Path, required=True)
argparser.add_argument('--outpath', '-o', type=Path, required=False)

args = argparser.parse_args()
checkpoint_path = args.checkpoint_path
style_image_paths = args.style_image_path
content_image_path = args.content_image_path
outpath = args.outpath

image_shape = (960, 1920, 3)
num_styles = len(style_image_paths)
style_weights_shape = (960, 1920, num_styles - 1)

element = {
    'content': common.image_dataset_from_filepaths([content_image_path], image_shape).batch(1).get_single_element(),
    'style_weights': tf.linalg.band_part(tf.ones((1,) + style_weights_shape), 0, -1),
    'style': tf.stack(list(common.image_dataset_from_filepaths(style_image_paths, image_shape).batch(1)), axis=1)
}

log = logging.getLogger()

from models import styleTransfer, stylePrediction, styleLoss, styleTransferInferenceModel
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
# style_transfer_inference_model.inference.load_weights(filepath=str(checkpoint_path))
predict_datapoint(element, element, style_transfer_inference_model.inference)

# save result if required
if outpath is not None:
    result = style_transfer_inference_model.training(element)
    image_data = result.numpy().squeeze()
    tf.keras.utils.save_img(outpath, image_data, data_format='channels_last')
