import os

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
argparser.add_argument('--outpath', '-o', type=Path, required=True)
argparser.add_argument('--profile_data_dir', '-p', type=Path, required=False)

args = argparser.parse_args()
checkpoint_path: Path = args.checkpoint_path
style_image_paths = args.style_image_path
outpath: Path = args.outpath
profile_data_dir: Path = args.profile_data_dir

from realtime_style_transfer.shape_config import ShapeConfig

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

config = ShapeConfig(hdr=True, num_styles=1)

content_dataset = hdrScreenshots.get_unreal_hdr_screenshot_dataset(
    common.content_target_dir / "lyra_hdr_images_continuous", config.channels,
    config.input_shape['content'])

log = logging.getLogger()

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
        config.num_styles,
        config.with_depth_loss),
)
element = config.get_dummy_input_element()[0]

# call once to build model
log.info("Building model.")
def setup_model(model:tf.keras.Model):
    model.trainable = False
    model.compile(run_eagerly=False)


setup_model(style_transfer_training_model.training)
setup_model(style_transfer_training_model.style_predictor)
setup_model(style_transfer_training_model.transfer)
style_transfer_training_model.training(element)

log.info(f"Loading weights from {checkpoint_path}")
load_status = style_transfer_training_model.training.load_weights(filepath=str(checkpoint_path))
load_status.assert_nontrivial_match()

style_params = style_transfer_training_model.style_predictor(
    common.image_dataset_from_filepaths(style_image_paths, config.image_shape)
    .batch(1)
    .get_single_element())

template_datapoint = {
    'style_params': tf.expand_dims(style_params, 0)
}

frames = list()
if profile_data_dir:
    profile_data_dir.mkdir(exist_ok=True, parents=True)

with tf.profiler.experimental.Profile(str(profile_data_dir)) if profile_data_dir else nullcontext():
    content_progress = tqdm(enumerate(content_dataset.prefetch(5)), file=sys.stdout, total=content_dataset.num_samples,
                            desc="Generating Frames")
    for i, content_image in content_progress:
        element = dict(template_datapoint)
        element['content'] = tf.expand_dims(content_image, 0)
        predicted_frame = style_transfer_training_model.transfer.predict(element, batch_size=1, verbose=0)

        frames.append((np.squeeze(predicted_frame) * 255).astype(int))

fps = 30
clip = VideoClip(make_frame=lambda t: frames[math.floor(t * fps)], duration=len(frames) / fps)
clip.write_videofile(str(outpath), fps=fps, bitrate='7M')
