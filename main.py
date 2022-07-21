import tensorflow as tf
from tensorflow import keras

from dataloaders import wikiart

from models import styleTransfer, stylePrediction, styleLoss

input_shape = (None, 1920, 960, 3)

training_dataset, validation_dataset = wikiart.get_dataset(input_shape[1:3])


style_transfer_model = styleTransfer.StyleTransferModel(input_shape)
style_prediction_model = stylePrediction.StylePredictionModel(input_shape)
style_loss_model = styleLoss.StyleLossModelEfficientNet(input_shape)

style_transfer_model.compile()
style_prediction_model.compile()
style_loss_model.compile()
style_transfer_model.build(input_shape)
style_prediction_model.build(input_shape)
style_loss_model.build(input_shape)

tf.keras.utils.plot_model(style_transfer_model, show_shapes=True, expand_nested=True, dpi=300,
                          to_file="style_transfer_model.png")
tf.keras.utils.plot_model(style_prediction_model, show_shapes=True, expand_nested=True, dpi=300,
                          to_file="style_prediction_model.png")
tf.keras.utils.plot_model(style_loss_model, show_shapes=True, expand_nested=True, dpi=300,
                          to_file="style_loss_model.png")
