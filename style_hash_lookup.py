import argparse
import sys
from typing import List
import json

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--hash", "-x", type=str, action='append', required=True)
args = parser.parse_args()

style_hashes: List[str] = args.hash

from realtime_style_transfer.dataloaders import wikiart
from realtime_style_transfer.dataloaders.wikiart import _read_dataset_manifest,image_manifest_to_filepath


for image_manifest in _read_dataset_manifest():
    image_filepath = image_manifest_to_filepath(image_manifest)
    if image_filepath.stem in style_hashes:
        print(f"{image_filepath.stem}: {json.dumps(image_manifest)}")
