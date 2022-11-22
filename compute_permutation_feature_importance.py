#!/usr/bin/env python3

"""
This script uses Permutation Feature Importance as introduced here:
Fisher, Aaron, Cynthia Rudin, and Francesca Dominici. “All models are wrong, but many are useful: Learning a variable’s importance by studying an entire class of prediction models simultaneously.” http://arxiv.org/abs/1801.01489 (2018).

Based on an article by Christoph
"""

import sys

import tqdm

from pathlib import Path
import tensorflow as tf
import logging
import argparse
import pickle
import datetime

from realtime_style_transfer.dataloaders import *


class PermutationFeatureImportanceData():
    channel_contributions: dict
    baseline_losses: dict
    num_samples: int

    def __init__(self):
        self.baseline_losses = {loss_name: 0.0 for loss_name in
                                      ("loss", "style_loss", "feature_loss", "total_variation_loss", "depth_loss")}
        self.channel_contributions = {loss_name: {n: 0.0 for n, c in config.channels}
                                      for loss_name in self.baseline_losses.keys()}

        self.num_samples = 0


log = logging.getLogger()

argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint_path', '-C', type=Path, required=True)

args = argparser.parse_args()
checkpoint_path = args.checkpoint_path

from realtime_style_transfer.shape_config import *

num_styles = 1
config = ShapeConfig(hdr=True, num_styles=num_styles)

from realtime_style_transfer.models import styleTransfer, stylePrediction, styleTransferTrainingModel, styleLoss
from realtime_style_transfer.dataloaders import wikiart


def progressify(iterable, **kwargs):
    return tqdm.tqdm(iterable, file=sys.stdout, ascii=False, **kwargs)


cache_dir = Path(__file__).parent / "cache"

permutation_importance_data_cache_file_path = cache_dir / "permutation_feature_importance"

if permutation_importance_data_cache_file_path.exists():
    log.info(f"Loading cached permutation feature importances from {permutation_importance_data_cache_file_path}" + \
             f"(created at {datetime.datetime.fromtimestamp(permutation_importance_data_cache_file_path.lstat().st_ctime)})")
    permutation_importance_data = pickle.load(permutation_importance_data_cache_file_path.open(mode='rb'))
else:

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
        style_loss_func_factory_func=lambda: styleLoss.make_style_loss_function(styleLoss.StyleLossModelVGG(config.output_shape),
                                                                                config.output_shape,
                                                                                config.num_styles,
                                                                                config.with_depth_loss),
    )

    # call once to build model
    log.info("Compiling model...")
    style_transfer_training_model.loss_model.compile(run_eagerly=True)
    log.info("Building model...")
    style_transfer_training_model.loss_model(next(iter(validation_dataset)))
    log.info(f"Loading weights from {checkpoint_path}")
    #load_status = style_transfer_training_model.loss_model.load_weights(filepath=str(checkpoint_path))
    checkpoint = tf.train.Checkpoint(style_transfer_training_model.inference)
    load_status = checkpoint.restore(str(checkpoint_path))
    load_status.assert_nontrivial_match()

    log.info("Loading full dataset...")
    preloaded_validation_dataset = list(progressify(validation_dataset, total=validation_dataset.num_samples))

    permutation_importance_data = PermutationFeatureImportanceData()

    for i, sample in enumerate(progressify(preloaded_validation_dataset, position=0, desc="Sample")):
        matched_samples = list(preloaded_validation_dataset)
        del matched_samples[i]
        baseline_losses = style_transfer_training_model.loss_model(sample)

        for loss, loss_value in baseline_losses.items():
            prev_loss_value = permutation_importance_data.baseline_losses.setdefault(loss, 0)
            permutation_importance_data.baseline_losses[loss] = prev_loss_value + loss_value


        for j, matched_sample in enumerate(progressify(matched_samples, leave=False, position=1, desc="Permutation")):
            component_lower_bound = 0
            for channel, num_components in progressify(config.channels, leave=False, position=2, desc="Channel"):
                component_upper_bound = component_lower_bound + num_components
                permuted_sample_x, permuted_sample_y = sample
                permuted_sample_content = permuted_sample_x['content'].numpy()
                permuted_sample_content[..., component_lower_bound:component_upper_bound] = \
                    matched_sample[0]['content'].numpy()[..., component_lower_bound:component_upper_bound]
                permuted_sample_x['content'] = tf.convert_to_tensor(permuted_sample_content)

                losses = style_transfer_training_model.loss_model((permuted_sample_x, permuted_sample_y))
                for loss, loss_value in losses.items():
                    permutation_importance_data.channel_contributions[loss][channel] += loss_value \
                                                                                        - baseline_losses[loss]
                component_lower_bound = component_upper_bound

        permutation_importance_data.num_samples = i * (len(preloaded_validation_dataset) - 1)

        permutation_importance_data_cache_file = permutation_importance_data_cache_file_path.open(mode='wb')
        pickle.dump(permutation_importance_data, permutation_importance_data_cache_file)

num_samples_and_permutations = permutation_importance_data.num_samples
for loss, channels in permutation_importance_data.channel_contributions.items():
    for channel, channel_value in channels.items():
        permutation_importance_data.channel_contributions[loss][channel] = (
                channel_value / num_samples_and_permutations).numpy().item()

for loss, channels in permutation_importance_data.channel_contributions.items():
    sorted_channels = list(channels.items())
    sorted_channels.sort(key=lambda x: x[1])
    print("\n\n")
    print(f"{loss}")
    print(f"-" * 29)
    for channel_name, channel_contribution in sorted_channels:
        print(
            f"{channel_name:20s}: {channel_contribution:+03.05f} ({(channel_contribution / permutation_importance_data.baseline_losses[loss]).numpy().item():.05%})")
