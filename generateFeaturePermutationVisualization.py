#!/usr/bin/env python3
import math
import sys

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from pathlib import Path
import logging
import argparse
import pickle
import datetime
import tensorflow as tf

from realtime_style_transfer.dataloaders import *

log = logging.getLogger()

argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint_path', '-C', type=Path, required=True)
argparser.add_argument('--outpath', '-o', type=Path, required=True)

args = argparser.parse_args()
checkpoint_path: Path = args.checkpoint_path
outpath: Path = args.outpath

from realtime_style_transfer.shape_config import *

config = ShapeConfig(hdr=True)

from realtime_style_transfer.models import styleTransfer, stylePrediction, styleTransferTrainingModel, styleLoss
from realtime_style_transfer.dataloaders import wikiart


def progressify(iterable, **kwargs):
    return tqdm.tqdm(iterable, file=sys.stdout, ascii=False, **kwargs)


def save_tensor_image(tensor: tf.Tensor, name: str, normalize=True):
    if normalize:
        mean, variance = tf.nn.moments(tensor, axes=[0,1,2])
        tensor = (tensor - mean) / tf.sqrt(variance) / 2 + 0.5
    tensor = tf.minimum(tf.maximum(tensor, 0), 1)
    image_data = np.uint8(np.squeeze(tensor) * 255)
    image = PIL.Image.fromarray(image_data)
    image.save(outpath / f"{name}.png")
    return image


cache_dir = Path(__file__).parent / "cache"

permutation_importance_data_cache_file_path = cache_dir / "permutation_feature_importance"

log.info("Loading dataset...")
training_dataset, validation_dataset = wikiart.get_hdr_dataset(config.input_shape,
                                                               batch_size=1,
                                                               output_shape=config.output_shape,
                                                               cache_dir=cache_dir,
                                                               channels=config.channels,
                                                               seed=278992)

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
    style_loss_func_factory_func=lambda: styleLoss.make_style_loss_function(
        styleLoss.StyleLossModelVGG(config.output_shape),
        config.output_shape,
        config.num_styles,
        config.with_depth_loss),
)

samples_iterator = iter(validation_dataset)
samples = (
    next(samples_iterator)[0],
    next(samples_iterator)[0],
)

# call once to build model
log.info("Compiling model...")
style_transfer_training_model.inference.compile(run_eagerly=True)
log.info("Building model...")
style_transfer_training_model.inference(samples[0])
log.info(f"Loading weights from {checkpoint_path}")
# load_status = style_transfer_training_model.loss_model.load_weights(filepath=str(checkpoint_path))
checkpoint = tf.train.Checkpoint(style_transfer_training_model.inference)
load_status = checkpoint.restore(str(checkpoint_path))
load_status.assert_nontrivial_match()

style_image_path = wikiart.style_image_dir / "00138f34171c13455d5bd65ce4eab19634ff1df7.jpg"
style_tensor = common.image_dataset_from_filepaths([style_image_path],
                                                   config.input_shape['style'][-3:]).get_single_element()

baseline_sample = {
    'content': samples[0]['content'],
    'style': tf.expand_dims(tf.expand_dims(style_tensor, 0), 0)
}

save_tensor_image(style_transfer_training_model.inference(baseline_sample), "baseline", False)
save_tensor_image(style_tensor, "style", False)

final_images = list()
component_lower_bound = 0
for channel, num_components in progressify(config.channels, desc="Channel"):
    component_upper_bound = component_lower_bound + num_components
    permuted_sample_x = dict(baseline_sample)
    matched_sample = samples[1]
    permuted_sample_content = tf.identity(permuted_sample_x['content']).numpy()
    permuted_sample_content[..., component_lower_bound:component_upper_bound] = \
        matched_sample['content'].numpy()[..., component_lower_bound:component_upper_bound]

    save_tensor_image(baseline_sample['content'][..., component_lower_bound:component_upper_bound],
                      f"baseline_content_{channel}")
    save_tensor_image(samples[1]['content'][..., component_lower_bound:component_upper_bound],
                      f"permutation_content_{channel}")

    permuted_sample_x['content'] = tf.convert_to_tensor(permuted_sample_content)

    final_image = style_transfer_training_model.inference(permuted_sample_x)
    image = save_tensor_image(final_image, f"permuted_{channel}", False)
    final_images.append(image)
    component_lower_bound = component_upper_bound

fig, axes = plt.subplots(math.ceil(len(final_images) / 3), 3, sharex=True, sharey=True)

for final_image, ax in zip(final_images, [ax for axesX in axes for ax in axesX]):
    ax.imshow(final_image)

fig.show()
