from pathlib import Path
import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from realtime_style_transfer.dataloaders import common
import logging

log = logging.getLogger()

argparser = argparse.ArgumentParser()
argparser.add_argument('style_image_path', type=Path)
argparser.add_argument('content_image_path', type=Path)
argparser.add_argument('transfer_model_path', type=Path)
argparser.add_argument('style_model_path', type=Path)
argparser.add_argument('--output-path', '-o', type=Path, required=False)

args = argparser.parse_args()
style_image_path: Path = args.style_image_path
content_image_path: Path = args.content_image_path
transfer_model_path: Path = args.transfer_model_path
style_model_path: Path = args.style_model_path
output_path: Path = args.output_path

image_shape = (960, 1920, 3)
sizeof_float = struct.calcsize("f")

log.info(f"Loading style image {style_image_path}")
style_image = common.image_dataset_from_filepaths([style_image_path], image_shape).batch(1).get_single_element()
content_image = common.image_dataset_from_filepaths([content_image_path], image_shape).batch(1).get_single_element()

log.info(f"Loading models")
style_prediction_model: tf.keras.Model = tf.saved_model.load(style_model_path)
transfer_model: tf.keras.Model = tf.saved_model.load(transfer_model_path)

log.info(f"Predicting")
style_params = style_prediction_model(style_image)
result = transfer_model({
    'content': content_image,
    'style_params': style_params
})

plt.imshow(np.squeeze(result.numpy()))
plt.show()
