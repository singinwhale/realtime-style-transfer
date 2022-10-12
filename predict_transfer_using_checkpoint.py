from pathlib import Path
import tensorflow as tf
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt

from realtime_style_transfer.dataloaders import common, tensorbuffer

argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint_path', '-C', type=Path, required=True)
argparser.add_argument('--style_tensor_path', '-st', type=Path, required=True)
argparser.add_argument('--content_image_path', '-c', type=Path, required=True)
argparser.add_argument('--outpath', '-o', type=Path, required=False)

args = argparser.parse_args()
checkpoint_path = args.checkpoint_path
style_tensor_path = args.style_tensor_path
content_image_path = args.content_image_path
outpath = args.outpath

image_shape = (960, 1920, 3)

content_dataset = common.image_dataset_from_filepaths([content_image_path], image_shape).batch(1)
datapoint = {
    'content': content_dataset.get_single_element(),
    'style_params': tensorbuffer.load_tensor_from_buffer(style_tensor_path, (1, 192,))
}

log = logging.getLogger()

from realtime_style_transfer.models import styleTransfer, stylePrediction, styleLoss, styleTransferTrainingModel

input_shape = {'content': image_shape, 'style': image_shape}
output_shape = image_shape

style_loss_model = styleLoss.StyleLossModelMobileNet(output_shape)

style_transfer_training_model = styleTransferTrainingModel.make_style_transfer_training_model(
    input_shape,
    style_predictor_factory_func=lambda num_top_parameters: stylePrediction.create_style_prediction_model(
        input_shape['style'], stylePrediction.StyleFeatureExtractor.MOBILE_NET, num_top_parameters
    ),
    style_transfer_factory_func=lambda: styleTransfer.create_style_transfer_model(input_shape['content']),
    style_loss_func_factory_func=lambda: styleLoss.make_style_loss_function(style_loss_model, {'content': image_shape, 'style': image_shape}, image_shape),
)
element = {
    'content': tf.zeros((1,) + image_shape),
    'style': tf.zeros((1,) + image_shape),
}

# call once to build model
style_transfer_training_model.training(element)
style_transfer_training_model.training.load_weights(filepath=str(checkpoint_path))

style_params_tensor = tensorbuffer.load_tensor_from_buffer(style_tensor_path, (1, 192,))
result = np.squeeze(style_transfer_training_model.transfer(datapoint).numpy())
plt.imshow(result)

# save result if required
if outpath is not None:
    result = style_transfer_training_model.training(element)
    image_data = result.numpy().squeeze()
    tf.keras.utils.save_img(outpath, image_data, data_format='channels_last')

plt.show()
