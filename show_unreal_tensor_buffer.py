from tracing import logsetup

from pathlib import Path
import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
from dataloaders.tensorbuffer import load_tensor_from_buffer

argparser = argparse.ArgumentParser()
argparser.add_argument('tensor_path', type=Path)
argparser.add_argument('--outpath', '-o', type=Path, required=False)

args = argparser.parse_args()
tensor_path: Path = args.tensor_path
outpath: Path = args.outpath

image_shape = (960, 1920, 3)

tensor = load_tensor_from_buffer(tensor_path, image_shape).numpy()

plt.imshow(tensor)

# save result if required
if outpath is not None:
    import matplotlib.image
    matplotlib.image.imsave(outpath, tensor)

plt.show()