import tensorflow as tf
from pathlib import Path
import math
import struct
import numpy as np


def load_tensor_from_buffer(buffer_filepath: Path, shape):
    num_elements = math.prod(shape)
    num_bytes = num_elements * struct.calcsize("f")
    with buffer_filepath.open(mode='rb') as tensor_data_file:
        tensor_bytes = tensor_data_file.read(num_bytes)

    tensor_np = np.reshape(struct.unpack(f"{num_elements}f", tensor_bytes), shape)
    tensor = tf.convert_to_tensor(tensor_np, dtype=tf.float32)
    return tensor
