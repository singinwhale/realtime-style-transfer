from tracing import logsetup

from pathlib import Path
import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from dataloaders import common
import logging

import metrics

log = logging.getLogger()

argparser = argparse.ArgumentParser()
argparser.add_argument('style_image_path', type=Path)
argparser.add_argument('model_path', type=Path)
argparser.add_argument('output_path', type=Path)

args = argparser.parse_args()
style_image_path: Path = args.style_image_path
model_path: Path = args.model_path
output_path: Path = args.output_path

image_shape = (960, 1920, 3)
sizeof_float = struct.calcsize("f")

log.info(f"Loading style image {style_image_path}")
style_image_dataset = common.image_dataset_from_filepaths([style_image_path], image_shape).batch(1)
style_image = style_image_dataset.get_single_element()

log.info(f"Loading model {model_path}")
style_prediction_model: tf.keras.Model = tf.saved_model.load(model_path)

log.info(f"Predicting Style")
style_params: np.ndarray = style_prediction_model(style_image).numpy()
assert style_params.dtype == np.float32

log.info(f"Writing style_params to {output_path}")
with output_path.open('wb') as style_output_file:
    data = style_params.flatten()
    style_output_file.write(data.tobytes())

metrics.print_stats(metrics.get_stats(style_params))
