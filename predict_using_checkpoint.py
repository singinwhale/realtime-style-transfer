from pathlib import Path
import tensorflow as tf
import logging
import argparse

from dataloaders import common

argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint_path', '-C', type=Path, required=True)
argparser.add_argument('--style_image_path', '-s', type=Path, required=True)
argparser.add_argument('--content_image_path', '-c', type=Path, required=True)
argparser.add_argument('--outpath', '-o', type=Path, required=False)

args = argparser.parse_args()
checkpoint_path = args.checkpoint_path
style_image_path = args.style_image_path
content_image_path = args.content_image_path
outpath = args.outpath

image_shape = (960, 1920, 3)

datapoint = common.pair_up_content_and_style_datasets(
    content_dataset=common.image_dataset_from_filepaths([content_image_path], image_shape).batch(1),
    style_dataset=common.image_dataset_from_filepaths([style_image_path], image_shape).batch(1),
    shapes={
        'content': image_shape,
        'style': image_shape,
    }
)

log = logging.getLogger()

from models import styleTransferFunctional, stylePrediction, styleLoss, styleTransferTrainingModel
from renderers.matplotlib import predict_datapoint

input_shape = {'content': image_shape, 'style': image_shape}
output_shape = image_shape

style_loss_model = styleLoss.StyleLossModelMobileNet(output_shape)

style_transfer_training_model = styleTransferTrainingModel.make_style_transfer_training_model(
    input_shape,
    style_predictor_factory_func=lambda num_top_parameters: stylePrediction.create_style_prediction_model(
        input_shape['style'], stylePrediction.StyleFeatureExtractor.MOBILE_NET, num_top_parameters
    ),
    style_transfer_factory_func=lambda: styleTransferFunctional.create_style_transfer_model(input_shape['content']),
    style_loss_func_factory_func=lambda: styleLoss.make_style_loss_function(style_loss_model),
)
element = datapoint.get_single_element()

# call once to build model
style_transfer_training_model.training(element)
style_transfer_training_model.training.load_weights(filepath=str(checkpoint_path))
predict_datapoint(element, element, style_transfer_training_model.training)

# save result if required
if outpath is not None:
    result = style_transfer_training_model.training(element)
    image_data = result.numpy().squeeze()
    tf.keras.utils.save_img(outpath, image_data, data_format='channels_last')
