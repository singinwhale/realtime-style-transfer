import math

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import logging

log = logging.getLogger(__name__)


def get_gram_matrix_model(input_spec):
    inputs = tf.keras.Input(input_spec)
    inputs = tf.cast(inputs, tf.float32)
    result = tf.linalg.einsum('bijc,bijd->bcd', inputs, inputs)
    input_shape = tf.shape(inputs)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    outputs = result / (num_locations)
    return tf.keras.Model(inputs, outputs, name="GramMatrix")


def gram_matrix(input_tensor):
    """## Calculate style

    The content of an image is represented by the values of the intermediate feature maps.

    It turns out, the style of an image can be described by the means and correlations across the different feature maps.
    Calculate a Gram matrix that includes this information by taking the outer product of the feature vector with itself at each location,
    and averaging that outer product over all locations. This Gram matrix can be calculated for a particular layer as:

    $$G^l_{cd} = \frac{\sum_{ij} F^l_{ijc}(x)F^l_{ijd}(x)}{IJ}$$

    This can be implemented concisely using the `tf.linalg.einsum` function:
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


class StyleLossModelBase(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = False
        self.content_layers = None
        self.style_layers = None
        self.feature_extractor = None
        self.content_loss_factor = 1
        self.style_loss_factor = 1
        self.total_variation_loss_factor = 1
        self.depth_loss_factor = 1

    def call(self, inputs, **kwargs):
        outputs = self.feature_extractor(inputs)
        # style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        # content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}

        # style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        content_layers = self.content_layers
        style_layers = self.style_layers

        content_dict = {content_layer_name: outputs[content_layer_name] for content_layer_name in content_layers}
        style_dict = {style_layer_name: outputs[style_layer_name] for style_layer_name in style_layers}

        return {'content': content_dict, 'style': style_dict}


class StyleLossModelVGG(StyleLossModelBase):
    """## Extract style and content
    When called on an image, this model returns the gram matrix (style) of the `style_layers` and content of the `content_layers`
    Build a model that returns the style and content tensors.
    """

    def __init__(self, input_shape):
        super().__init__(name='StyleLossModelVGG')

        # style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
        content_layers = ['block5_conv3']

        # Load our model. Load pretrained VGG, trained on ImageNet data
        vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False

        inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)
        x = inputs
        # x = tf.cast(inputs, tf.float16)
        # x = tf.keras.layers.AveragePooling2D(4)(x)
        outputs = {name: vgg.get_layer(name).output for name in (content_layers + style_layers)}
        self.feature_extractor = tf.keras.Model([vgg.input], outputs, name="VGG16LayerAccessor")
        self.feature_extractor.trainable = False

        x = self.feature_extractor(x)

        self.feature_extractor = tf.keras.Model(inputs, x)

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.content_loss_factor = 1e4
        self.style_loss_factor = 1e-3
        self.total_variation_loss_factor = 1e-1
        self.depth_loss_factor = 1e-2

    def call(self, inputs, **kwargs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg16.preprocess_input(inputs)
        return super().call(preprocessed_input, **kwargs)


class StyleLossModelEfficientNet(StyleLossModelBase):
    """## Extract style and content
    When called on an image, this model returns the gram matrix (style) of the `style_layers` and content of the `content_layers`
    Build a model that returns the style and content tensors.
    """

    def __init__(self, input_shape, name="StyleLossModelEfficientNet"):
        super().__init__(name=name)

        style_layer_names = [
            'block2c_add',
            'block3c_add',
            'block4e_add',
        ]
        content_layer_names = [
            'block5e_add',
            'block6f_add',
            'block7b_add',
        ]
        output_layer_names = style_layer_names + content_layer_names

        # Load our model. Load pretrained VGG, trained on ImageNet data
        efficientnet = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False,
                                                                         input_shape=input_shape,
                                                                         include_preprocessing=False)
        efficientnet.trainable = False

        outputs = {name: efficientnet.get_layer(name).output for name in output_layer_names}

        self.feature_extractor = tf.keras.Model([efficientnet.input], outputs, name="EfficientNetLayerAccessor")
        self.feature_extractor.trainable = False

        self.style_layers = style_layer_names
        self.content_layers = content_layer_names
        self.num_style_layers = len(style_layer_names)

    def call(self, inputs, **kwargs):
        """Expects float input in [0,1] but rescales it to [-1, 1]
        """
        x = tf.keras.layers.Rescaling(2.0, -1.0)(inputs)
        return super().call(x, **kwargs)


class StyleLossModelMobileNet(StyleLossModelBase):

    def __init__(self, input_shape, name="StyleLossModelMobileNet"):
        super().__init__(name=name)

        style_layer_names = [
            'expanded_conv_2/Add',
            'expanded_conv_4/Add',
            'expanded_conv_5/Add',
            'expanded_conv_7/Add',
        ]
        content_layer_names = [
            'expanded_conv_9/Add',
            'expanded_conv_10/Add',
        ]
        output_layer_names = style_layer_names + content_layer_names

        mobile_net = tf.keras.applications.MobileNetV3Small(include_top=False, input_shape=input_shape,
                                                            include_preprocessing=False)
        mobile_net.trainable = False

        outputs = {name: mobile_net.get_layer(name).output for name in output_layer_names}

        self.feature_extractor = tf.keras.Model([mobile_net.input], outputs, name="MobileNetV3LayerAccessor")
        self.feature_extractor.trainable = False

        self.style_layers = style_layer_names
        self.content_layers = content_layer_names
        self.num_style_layers = len(style_layer_names)
        self.content_loss_factor = 1e-3
        self.style_loss_factor = 1
        self.total_variation_loss_factor = 1e-3
        self.depth_loss_factor = 1e-4

    def call(self, inputs, **kwargs):
        """Expects float input in [0,1] but rescales it to [-1, 1]
        """
        x = tf.keras.layers.Rescaling(2.0, -1.0)(inputs)
        return super().call(x)


class StyleLossModelDummy(StyleLossModelBase):

    def __init__(self, input_shape, name="StyleLossModelDummy"):
        super().__init__(name=name)

        inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)

        conv1 = tf.keras.layers.Conv2D(3, 3, 1, padding='same', name="dummy_conv1")
        output1 = conv1(inputs)
        conv2 = tf.keras.layers.Conv2D(3, 3, 1, padding='same', name="dummy_conv2")
        output2 = conv2(output1)

        self.style_layers = [
            conv1.name
        ]
        self.content_layers = [
            conv2.name
        ]

        self.feature_extractor = tf.keras.Model(inputs, {conv1.name: output1, conv2.name: output2}, name="Dummy")
        self.feature_extractor.trainable = False

        self.num_style_layers = 1
        self.content_loss_factor = 1
        self.style_loss_factor = 1


class BatchifyLayer(tf.keras.layers.Layer):
    def __init__(self, wrapped_layer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wrapped_layer = wrapped_layer

    def build(self, input_shape):
        if input_shape[0] is None:
            input_shape = list(input_shape)
            input_shape[0] = 1

        self.wrapped_layer.build(input_shape)

    def call(self, inputs, *args, **kwargs):
        def batchify(t):
            t_exp = tf.expand_dims(t, 0)
            t_exp = tf.cast(t_exp, tf.float32)
            result = self.wrapped_layer(t_exp)
            return tf.cast(result, tf.float32)

        mapped = tf.map_fn(batchify, inputs, infer_shape=False,
                           fn_output_signature=tf.TensorSpec(
                               dtype=tf.float32,
                               shape=(None, 384, 384)
                           ))
        return mapped


def get_depth_loss_func(input_shape):
    import tensorflow_hub as hub

    # tf.keras.mixed_precision.set_global_policy('float32')
    midas_model_unbatched = hub.KerasLayer("https://tfhub.dev/intel/midas/v2/2", tags=['serve'], signature='serving_default',
                           input_shape=(3, 384, 384), output_shape=(384, 384))
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    midas_model = BatchifyLayer(midas_model_unbatched, dtype=tf.float32)

    resizing_layer = tf.keras.layers.Resizing(384, 384)

    def normalize_depth(d):
        t = tfp.stats.percentile(d, 50)
        s = tf.reduce_mean(tf.abs(d - t))
        return (d - t) / s

    def ssitrim_loss(d1, d2):
        d1 = normalize_depth(d1)
        d2 = normalize_depth(d2)
        absolute_error = tf.abs(d1 - d2)
        eightieth_percentile = tfp.stats.percentile(absolute_error, 80)
        lower_eighty_percent = tf.boolean_mask(absolute_error, absolute_error < eightieth_percentile)
        return 0.5 * tf.reduce_sum(lower_eighty_percent) / tf.cast(tf.size(absolute_error), tf.float32)

    def ssimse_loss(d1, d2):
        return 0.0

    def depth_loss(ground_truth_image, predicted_image):
        """
        Depth loss according to Liu et al. 2017 - depth aware neural style transfer
        """
        resized_predicted_image = resizing_layer(predicted_image)
        resized_ground_truth_image = resizing_layer(ground_truth_image)
        predicted_depth = midas_model(tf.transpose(resized_predicted_image, [0, 3, 1, 2]))
        ground_truth_depth = midas_model(tf.transpose(resized_ground_truth_image, [0, 3, 1, 2]))
        return mean_l2_loss_on_batch(ground_truth_depth - predicted_depth)

    return depth_loss


def mean_l2_loss_on_batch(tensor):
    axis = list(range(1, len(tensor.shape)))
    return tf.reduce_mean(0.5 * tensor ** 2, axis=axis)


def make_style_loss_function(loss_feature_extractor_model: StyleLossModelBase, output_shape, num_styles,
                             with_depth_loss=True):
    loss_feature_extractor_model.trainable = False
    root_inputs = {
        'prediction': tf.keras.Input(output_shape),
        'ground_truth': {'content': tf.keras.Input(output_shape),
                         'style': tf.keras.Input((num_styles,) + output_shape)}
    }

    inputs = root_inputs

    # perform feature extraction on the input content image and diff it against the output features
    input_style, input_prediction, input_ground_truth = (inputs['ground_truth']['style'],
                                                         inputs['prediction'],
                                                         inputs['ground_truth']['content'])

    assert len(input_style.shape) < 5 or input_style.shape[1] == 1, \
        f"Loss model does not support multiple styles. Found {input_style.shape[1]} in shape {input_style.shape}"

    single_input_style = tf.squeeze(input_style, axis=1, name="squeeze_style") if input_style.shape[1] == 1 else input_style

    loss_data_content = loss_feature_extractor_model(input_ground_truth)
    loss_data_style = loss_feature_extractor_model(single_input_style)
    loss_data_prediction = loss_feature_extractor_model(input_prediction)
    input_feature_values: tf.Tensor = loss_data_content['content']
    input_style_features: tf.Tensor = loss_data_style['style']
    output_feature_values: tf.Tensor = loss_data_prediction['content']
    output_style_features: tf.Tensor = loss_data_prediction['style']

    feature_loss = tf.reduce_mean([
        mean_l2_loss_on_batch(tf.cast(out_value, tf.float32) - tf.cast(in_value, tf.float32))
        for (input_layer, in_value), (out_layer, out_value)
        in zip(input_feature_values.items(), output_feature_values.items())
    ], axis=[0]) * loss_feature_extractor_model.content_loss_factor

    gram_matrix_model = get_gram_matrix_model((None, None, None))
    style_loss = tf.reduce_mean([
        mean_l2_loss_on_batch((gram_matrix_model(out_value) - gram_matrix_model(in_value)))
        for (input_layer, in_value), (out_layer, out_value)
        in zip(input_style_features.items(), output_style_features.items())
    ], axis=[0]) * loss_feature_extractor_model.style_loss_factor

    total_variation_loss = tf.image.total_variation(input_prediction, 'total_variation_loss') * \
                           loss_feature_extractor_model.total_variation_loss_factor

    if with_depth_loss:
        depth_loss = get_depth_loss_func(output_shape[:-1])(input_ground_truth, input_prediction) * \
                     loss_feature_extractor_model.depth_loss_factor

    total_loss = tf.cast(feature_loss, tf.float32) + \
                 tf.cast(style_loss, tf.float32) + \
                 tf.cast(total_variation_loss, tf.float32)

    if with_depth_loss:
        total_loss += depth_loss

    output = {
        "loss": total_loss,
        "feature_loss": feature_loss,
        "style_loss": style_loss,
        "total_variation_loss": total_variation_loss,
    }
    if with_depth_loss:
        output['depth_loss'] = depth_loss

    model = tf.keras.Model(root_inputs, output, name="StyleLoss")
    model.trainable = False

    def compute_loss(y_pred: tf.Tensor, y_true):
        return model({
            'prediction': y_pred,
            'ground_truth': y_true
        })

    return compute_loss, model
