from pathlib import Path
import struct
import argparse
import numpy as np
import math
import tensorflow as tf

from realtime_style_transfer.metrics import get_stats, print_stat_comparison

argparser = argparse.ArgumentParser()
argparser.add_argument('input_tensor_path', type=Path)
argparser.add_argument('style_tensor_path', type=Path)
argparser.add_argument('model_path', type=Path)

args = argparser.parse_args()
input_tensor_path: Path = args.input_tensor_path
style_tensor_path: Path = args.style_tensor_path
model_path: Path = args.model_path

image_shape = (960, 1920, 3)
sizeof_float = struct.calcsize("f")

with input_tensor_path.open(mode='rb') as input_tensor_data_file:
    num_elements_in_tensor = math.prod(image_shape)
    tensor_bytes = input_tensor_data_file.read(num_elements_in_tensor * sizeof_float)
    input_nd = np.reshape(struct.unpack(f"{num_elements_in_tensor}f", tensor_bytes), image_shape)
    input_tensor: tf.Tensor = tf.convert_to_tensor(input_nd, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, 0)

with style_tensor_path.open(mode='rb') as style_tensor_data_file:
    tensor_bytes = style_tensor_data_file.read(192 * sizeof_float)
    style_params = np.array(struct.unpack(f"192f", tensor_bytes))
    style_params_tensor = tf.convert_to_tensor(style_params)

style_prediction_model: tf.keras.Model = tf.saved_model.load(model_path)

actual_style_params = style_prediction_model(input_tensor).numpy()

unreal_stats = get_stats(style_params_tensor)
ground_truth_stats = get_stats(actual_style_params)


print_stat_comparison('Unreal', unreal_stats, 'Truth', ground_truth_stats)
