import enum

import tensorflow as tf

import logging

log = logging.getLogger(__name__)

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 1. / 3.,
        "mode": "fan_out",
        "distribution": "uniform"
    }
}


class StyleFeatureExtractor(enum.Enum):
    DUMMY = 'DUMMY'
    EFFICIENT_NET = 'EFFICIENT_NET'
    MOBILE_NET = 'MOBILE_NET'


def create_style_prediction_model(input_shape, feature_extractor: StyleFeatureExtractor, num_top_parameters,
                                  num_style_parameters=100, dropout_rate=0.2,
                                  name="StylePredictionModel"):
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)

    if feature_extractor == StyleFeatureExtractor.DUMMY:
        feature_extractor_model = tf.keras.layers.Conv2D(1, 9, 5, padding='same', name="dummy_conv")
    elif feature_extractor == StyleFeatureExtractor.MOBILE_NET:
        feature_extractor_model = tf.keras.applications.MobileNetV3Small(
            include_top=False,
            input_shape=input_shape)
    elif feature_extractor == StyleFeatureExtractor.EFFICIENT_NET:
        feature_extractor_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            include_top=False,
            input_shape=input_shape)
    else:
        raise ValueError(
            f"{feature_extractor} is not a valid value for feature_extractor. Must be a StyleFeatureExtractor")

    feature_extractor_model.trainable = True

    x = feature_extractor_model(inputs)

    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)

    log.info(f"Using {num_style_parameters} style parameters")
    x = tf.keras.layers.Dense(
        num_style_parameters,
        activation=tf.keras.activations.softmax,
        kernel_initializer=DENSE_KERNEL_INITIALIZER,
        bias_initializer=tf.constant_initializer(1),
        name="StylePredictor")(x)

    num_norm_parameters = num_top_parameters
    log.debug(f"Using {num_norm_parameters} norm parameters")
    x = tf.keras.layers.Dense(
        num_norm_parameters,
        activation=tf.keras.activations.softmax,
        kernel_initializer=DENSE_KERNEL_INITIALIZER,
        bias_initializer=tf.constant_initializer(0.5),
        name="StyleNormPredictor")(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name=name)
