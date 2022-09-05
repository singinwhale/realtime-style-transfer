import numpy as np

import tensorflow as tf

image_shape = (None, 960 // 4, 1920 //4, 3)


class StylePredictionModelDummy(tf.keras.Model):
    feature_extractor = None

    def __init__(self, num_top_parameters, name="StylePredictionModelDummy"):
        super().__init__(name=name)

        self.feature_extractor = tf.keras.layers.Conv2D(1, 9, 5, padding='same', name="dummy_conv")

        self.style_norm_predictor = tf.keras.layers.Dense(
            num_top_parameters,
            activation=tf.keras.activations.softmax,
            name="style_norm_predictor")

    def call(self, inputs, training=None, mask=None):
        x = self.feature_extractor(inputs)

        x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
        x = self.style_norm_predictor(x)
        return x


class ConditionalInstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, num_feature_maps, epsilon=1e-5):
        super().__init__(name="ConditionalInstanceNormalization")
        self.epsilon = epsilon
        self.num_feature_maps = num_feature_maps

    def call(self, x, **kwargs):
        inputs = x
        x = inputs['inputs']
        scale = inputs['scale']
        bias = inputs['bias']
        x = x * scale + bias
        return x

    def get_config(self):
        return {
            "epsilon": self.epsilon,
            "num_feature_maps": self.num_feature_maps,
        }


def expand(input_shape, filters, size, strides, name) -> tf.keras.Model:
    name = f"expand_{name}"
    content_inputs = tf.keras.layers.Input(shape=input_shape)
    scale = tf.keras.layers.Input(shape=(1, 1, filters))
    bias = tf.keras.layers.Input(shape=(1, 1, filters))
    inputs = {
        "inputs": content_inputs,
        "scale": scale,
        "bias": bias,
    }
    result = tf.keras.layers.Conv2DTranspose(
        filters=filters, kernel_size=size, strides=strides, padding='same',
        name=f"{name}_conv")(content_inputs)

    instance_norm_params = {
        "inputs": result,
        "scale": scale,
        "bias": bias,
    }
    result = ConditionalInstanceNormalization(filters)(instance_norm_params)

    result = tf.keras.layers.ReLU()(result)

    return tf.keras.Model(inputs, result, name=name)


def make_style_transfer_model(input_shape,
                              name="StyleTransferModel"):
    decoder_layer_specs = [
        {"filters": 64, "size": 3, "strides": 2},
        {"filters": 32, "size": 3, "strides": 2},
    ]

    inputs = {
        'content': tf.keras.layers.Input(shape=input_shape['content'][1:]),
        'style': tf.keras.layers.Input(shape=input_shape['style'][1:]),
    }
    content_input, style_input = (inputs['content'], inputs['style'])

    num_style_parameters = sum(map(lambda spec: spec['filters'] * 2, decoder_layer_specs))
    style_predictor = StylePredictionModelDummy(num_style_parameters)
    style_params = style_predictor(style_input)

    x = content_input

    input_filters = input_shape['content'][-1]
    style_norm_param_lower_bound = 0
    for i, decoder_layer_spec in enumerate(decoder_layer_specs):
        style_norm_scale_upper_bound = style_norm_param_lower_bound + decoder_layer_spec["filters"]
        style_norm_offset_upper_bound = style_norm_scale_upper_bound + decoder_layer_spec["filters"]
        scale = style_params[:, style_norm_param_lower_bound:style_norm_scale_upper_bound]
        offset = style_params[:, style_norm_scale_upper_bound:style_norm_offset_upper_bound]
        scale, offset = tf.expand_dims(scale, -2, name="expand_scale_0"), tf.expand_dims(offset, -2, name="expand_offset_0")
        scale, offset = tf.expand_dims(scale, -2, name="expand_scale_1"), tf.expand_dims(offset, -2, name="expand_offset_1")
        style_norm_param_lower_bound = style_norm_offset_upper_bound

        expand_layer_input_shape = (input_shape['content'][1] * 2 ** i, input_shape['content'][2] * 2 ** i, input_filters)
        expand_layer = expand(input_shape=expand_layer_input_shape, name=i,
                              filters=decoder_layer_spec["filters"],
                              size=decoder_layer_spec["size"],
                              strides=decoder_layer_spec["strides"])
        x = expand_layer({
            "inputs": x,
            "scale": scale,
            "bias": offset,
        })
        input_filters = decoder_layer_spec["filters"]

    model = tf.keras.Model(inputs=inputs, outputs=x, name=name)
    return model


input_shape = {'content': image_shape, 'style': image_shape}
output_shape = image_shape

style_transfer_model = make_style_transfer_model(input_shape)
element = {
    'content': tf.convert_to_tensor(np.zeros((1, image_shape[1], image_shape[2], 3))),
    'style': tf.convert_to_tensor(np.zeros((1,  image_shape[1], image_shape[2], 3))),
}

# call once to build model
style_transfer_model(element)

style_transfer_model.save(filepath="%TEMP%/model", include_optimizer=False, save_format='tf')
