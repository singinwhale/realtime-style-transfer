from pathlib import Path
import struct
import argparse

import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from realtime_style_transfer.dataloaders import common
import logging

log = logging.getLogger()

argparser = argparse.ArgumentParser()
argparser.add_argument('style_image_path', type=Path)
argparser.add_argument('content_image_path', type=Path)
argparser.add_argument('--model_path', type=Path)
argparser.add_argument('--output-path', '-o', type=Path, required=False)

args = argparser.parse_args()
style_image_path: Path = args.style_image_path
content_image_path: Path = args.content_image_path
model_path: Path = args.model_path
output_path: Path = args.output_path

image_shape = (480, 960, 3)
sizeof_float = struct.calcsize("f")

log.info(f"Loading style image {style_image_path}")
style_image = common.image_dataset_from_filepaths([style_image_path], image_shape).batch(1).batch(1).get_single_element()
content_image = common.image_dataset_from_filepaths([content_image_path], image_shape).batch(1).get_single_element()

log.info(f"Loading models")
transfer_model: tf.keras.Model = tf.saved_model.load(model_path)

log.info(f"Predicting")
result = transfer_model({
    'content': content_image,
    'style': style_image
})

predicted_frame = np.uint8(result.numpy().squeeze() * 255)
if output_path:
    PIL.Image.fromarray(predicted_frame, mode="RGB").save(output_path)

plt.imshow(predicted_frame)
plt.show()
