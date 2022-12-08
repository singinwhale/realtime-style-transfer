# Realtime Style Transfer

[Overview Page](https://singinwhale.github.io/realtime-style-transfer)

<iframe width="480" height="240" src="https://www.youtube.com/embed/Y437ejhyT_U" title="rst-960-120-32-3 In Engine Footage" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

[[toc]]

## Download

Downloads of the saved tensorflow and ONNX models can be found in the releases section.

[Downloads](https://github.com/singinwhale/realtime-style-transfer/releases)

## Installation

### Install Dependencies

You need the [Anaconda](https://www.anaconda.com/products/distribution#Downloads) package manager.

You also need Nvidia [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive)
and [CuDNN 8.1.*](https://developer.nvidia.com/rdp/cudnn-archive)

### Download and Setup Repo

```shell
git clone https://github.com/singinwhale/realtime-style-transfer.git
cd realtime-style-transfer
conda env create -f ./environment.yml
conda activate realtime-style-transfer
# optional:
export TF_GPU_ALLOCATOR=cuda_malloc_async

```

### Setup Shape Config

This system does not expose all parameters as arguments to the scrips.
You often need to adjust scripts in order to get the model that you want.
Most often these changes are limited to `realtime_style_transfer/shape_config.py` however.

Here you must update the first couple of arguments to match your input data.
The parameters shown here correspond to rst-960-120-32-17.

```python

class ShapeConfig:
    def __init__(self, num_styles=1, hdr=True,
                 bottleneck_res_y=120,
                 bottleneck_num_filters=128,
                 resolution_divider=2,
                 num_channels=17):
```

### Dataset

The full dataset is too big to share publicly on the internet. A small subset of the dataset can be found in the
releases page. Unzip `example-data.zip` to the repository root so that the folder structure is as follows:
```
realtime-style-transfer
├───data
│   ├───screenshots
│   │   └───hdr_images
│   │       ├───training
│   │       └───validation
│   └───wikiart
│       └───images
│...
```
The wikiart dataset will be downloaded completely on first load but screenshots will have to be provided by you.
You can collect screenshots by using the modified unreal engine available [here](https://github.com/singinwhale/UnrealEngine/tree/realtime-style-transfer)
and the Lyra-Project available [here](https://git.singinwhale.com/singinwhale/RealtimeStyleTransferRuntime) and [here](https://github.com/singinwhale/realtime-style-transfer-unreal)

## Training and Inference

### Training

```shell
python train_network.py
```

### Evaluate

#### Using saved models

The saved models can be downloaded from the releases section.

```shell
python predict_using_saved_models.py 
"./path/to/style.png"
"./path/to/content.png"
"./path/to/model.training.tf"
--output-path "./path/to/stylized_image.png" 
```

#### Using checkpoints

Single Style

```shell
python predict_using_checkpoint.py
-C ./logs/rst-960-120-128-17/checkpoints/weights/latest_epoch_weights
-s ./data/wikiart/debug_images/training/00138f34171c13455d5bd65ce4eab19634ff1df7.jpg
-c ./data/screenshots/hdr_images/training/HighresScreenshot_2022.09.30-10.04.46.png
-o ./temp/lyra-960-120-128-17-00138f34171c13455d5bd65ce4eab19634ff1df7.jpg
```

Dual Style

```shell
python predict_using_checkpoint.py
-s "./data/wikiart/debug_images/training/001b4d4e463f967f179bd7fa1d8036999d477014.jpg" # style 1
-s "./data/wikiart/debug_images/training/00138f34171c13455d5bd65ce4eab19634ff1df7.jpg" # style 2
-w "./temp/HighresScreenshot_2022.09.30-10.04.46_shadowmask.png" # weights map
-c "./data/screenshots/hdr_images/training/HighresScreenshot_2022.09.30-10.04.46.png" # content
-C "./logs/rst-960-120-128-18/checkpoints/weights/latest_epoch_weights" # checkpoint
-o "./temp/stylized_image.jpg" # output
```