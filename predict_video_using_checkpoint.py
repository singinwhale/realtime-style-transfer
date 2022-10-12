from pathlib import Path
import tensorflow as tf
import numpy as np
import logging
import argparse
import math
from moviepy.editor import *

from dataloaders import common, hdrScreenshots, wikiart
from models import styleTransfer, stylePrediction, styleLoss, styleTransferTrainingModel

argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint_path', '-C', type=Path, required=True)
argparser.add_argument('--style_image_path', '-s', type=Path, action='append')
argparser.add_argument('--outpath', '-o', type=Path, required=True)

args = argparser.parse_args()
checkpoint_path = args.checkpoint_path
style_image_paths = args.style_image_path
outpath = args.outpath

from realtime_style_transfer.shape_config import ShapeConfig

config = ShapeConfig(hdr=True, num_styles=1)

content_dataset = hdrScreenshots.get_unreal_hdr_screenshot_dataset(wikiart.content_hdr_image_dir / "training", config.channels,
                                                                   config.hdr_input_shape['content'])
template_datapoint = {
    'style': tf.expand_dims(
        common.image_dataset_from_filepaths(style_image_paths, config.image_shape).batch(1).get_single_element(), 0),
    'style_weights': tf.zeros((1,) + config.input_shape['style_weights'])
}

log = logging.getLogger()

style_loss_model = styleLoss.StyleLossModelMobileNet(config.output_shape)

style_transfer_training_model = styleTransferTrainingModel.make_style_transfer_training_model(
    config.input_shape,
    style_predictor_factory_func=lambda num_top_parameters: stylePrediction.create_style_prediction_model(
        config.input_shape['style'][1:], stylePrediction.StyleFeatureExtractor.MOBILE_NET, num_top_parameters
    ),
    style_transfer_factory_func=lambda: styleTransfer.create_style_transfer_model(config.input_shape['content'], config.num_styles),
    style_loss_func_factory_func=lambda: styleLoss.make_style_loss_function(style_loss_model, config.input_shape,
                                                                            config.output_shape),
)
element = {
    'content': tf.zeros((1,) + config.input_shape['content']),
    'style': tf.zeros((1,) + config.input_shape['style']),
    'style_weights': tf.zeros((1,) + config.input_shape['style_weights']),
}

# call once to build model
log.info("Building model.")
style_transfer_training_model.training(element)

log.info(f"Loading weights from {checkpoint_path}")
style_transfer_training_model.training.load_weights(filepath=str(checkpoint_path))

frames = list()
for i, content_image in enumerate(content_dataset):
    log.info(f"Generating frame {i}")
    element = dict(template_datapoint)
    element['content'] = tf.expand_dims(content_image, 0)
    predicted_frame = style_transfer_training_model.training.predict(element, batch_size=1)

    frames.append((np.squeeze(predicted_frame) * 255).astype(int))

fps = 8
clip = VideoClip(make_frame=lambda t: frames[math.floor(t * fps)], duration=len(frames) / fps)
clip.write_videofile(str(outpath), fps=fps, bitrate='7M')
