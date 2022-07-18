import urllib.request
from pathlib import Path
import tarfile

target_dir = Path(__file__).parent.absolute()

url = 'https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/advprop/efficientnet-b3.tar.gz'

checkpoint_filename = target_dir / 'efficientnet-b3.tar.gz'
urllib.request.urlretrieve(url, checkpoint_filename, reporthook=lambda chunk_num, max_chunk_size, total_size: print(
    f"{chunk_num * max_chunk_size / total_size * 100}%"))

print(f"Extracting {checkpoint_filename} to {target_dir} ...")
tar = tarfile.open(checkpoint_filename, "r:gz")
tar.extractall(path=target_dir)
tar.close()

checkpoint_filename.unlink(missing_ok=False)
