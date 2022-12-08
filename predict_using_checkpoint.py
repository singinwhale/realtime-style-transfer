import os

import PIL.Image
from matplotlib import pyplot as plt

os.environ['PATH'] += r";C:\Program Files\NVIDIA Corporation\Nsight Systems 2022.1.3\target-windows-x64"
os.environ['PATH'] += r";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\extras\CUPTI\lib64"

from pathlib import Path
import tensorflow as tf
import numpy as np
import logging
import argparse
import math
from moviepy.editor import *
from tqdm import tqdm
from contextlib import nullcontext

from realtime_style_transfer.dataloaders import common, hdrScreenshots, wikiart
from realtime_style_transfer.models import styleTransfer, stylePrediction, styleLoss, styleTransferTrainingModel

argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint_path', '-C', type=Path, required=True)
argparser.add_argument('--style_image_path', '-s', type=Path, action='append')
argparser.add_argument('--style_weights_paths', '-w', type=Path, required=False, action='append')
argparser.add_argument('--content', '-c', type=Path, required=True)
argparser.add_argument('--outpath', '-o', type=Path, required=True)
argparser.add_argument('--profile_data_dir', '-p', type=Path, required=False)

args = argparser.parse_args()
checkpoint_path: Path = args.checkpoint_path
style_image_paths = args.style_image_path
style_weights_paths = args.style_weights_paths
content_path = args.content
outpath: Path = args.outpath
profile_data_dir: Path = args.profile_data_dir

from realtime_style_transfer.shape_config import ShapeConfig

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

config = ShapeConfig(hdr=True, num_styles=len(style_image_paths))
if config.num_channels > 3:
    content_image = hdrScreenshots.get_unreal_hdr_screenshot_dataset_from_filepaths([content_path],
                                                                                    config.channels,
                                                                                    config.input_shape['content']) \
        .batch(1) \
        .get_single_element()
else:
    content_image = common.image_dataset_from_filepaths([content_path], config.image_shape).batch(
        1).get_single_element()

log = logging.getLogger()

style_loss_model = styleLoss.StyleLossModelMobileNet(config.output_shape)

style_transfer_inference_model = styleTransferTrainingModel.make_style_transfer_inference_model(
    num_styles=config.num_styles,
    style_predictor_factory_func=lambda num_top_parameters: stylePrediction.create_style_prediction_model(
        config.input_shape['style'][1:], config.style_feature_extractor_type, num_top_parameters
    ),
    style_transfer_factory_func=lambda: styleTransfer.create_style_transfer_model(
        config.input_shape['content'],
        config.output_shape, config.bottleneck_res_y, config.bottleneck_num_filters, config.num_styles
    )
)
element = config.get_dummy_input_element()[0]

# call once to build model
log.info("Building model.")


def setup_model(model: tf.keras.Model):
    model.trainable = False
    model.compile(run_eagerly=False)


setup_model(style_transfer_inference_model.style_predictor)
setup_model(style_transfer_inference_model.transfer)
setup_model(style_transfer_inference_model.inference)
style_transfer_inference_model.inference(element)

log.info(f"Loading weights from {checkpoint_path}")
load_status = style_transfer_inference_model.inference.load_weights(filepath=str(checkpoint_path))
load_status.assert_nontrivial_match()

style_images = common.image_dataset_from_filepaths(style_image_paths, config.image_shape).batch(
    config.num_styles).batch(1).get_single_element()

element = {
    'style': style_images,
    'content': content_image
}

if 'style_weights' in config.input_shape:
    element['style_weights'] = common.image_dataset_from_filepaths(
        style_weights_paths, config.input_shape['style_weights']).batch(1).get_single_element()

predicted_frame = np.uint8(style_transfer_inference_model.inference.predict(element).squeeze() * 255)

plt.imshow(predicted_frame)
PIL.Image.fromarray(predicted_frame, mode="RGB").save(outpath)
PIL.Image.fromarray(np.uint8(
    common.image_dataset_from_filepaths([content_path], config.image_shape)
    .get_single_element()
    .numpy()
    .squeeze() * 255), mode="RGB").save((outpath.parent / outpath.stem).with_suffix(".content" + outpath.suffix))
#PIL.Image.fromarray(np.uint8(style_images.numpy().squeeze() * 255), mode="RGB").save(
#    (outpath.parent / outpath.stem).with_suffix(".style" + outpath.suffix))
plt.show()
