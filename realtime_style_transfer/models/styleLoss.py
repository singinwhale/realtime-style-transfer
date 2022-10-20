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

        channels = input_shape[-1]
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
        self.content_loss_factor = 1e-3
        self.style_loss_factor = 0.5e-8
        self.total_variation_loss_factor = 1e-1

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
        self.content_loss_factor = 1e-4
        self.style_loss_factor = 0.5e-3
        self.total_variation_loss_factor = 1e-3

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

        self.feature_extractor = tf.keras.Model(inputs, [output1, output2], name="Dummy")
        self.feature_extractor.trainable = False

        self.num_style_layers = 1
        self.content_loss_factor = 1
        self.style_loss_factor = 1


def get_depth_loss_func(input_shape):
    import tensorflow_hub as hub

    midas_model = hub.KerasLayer("https://tfhub.dev/intel/midas/v2/2", tags=['serve'],
                                 signature='serving_default',
                                 input_shape=(3, 384, 384), output_shape=(384, 384))

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
        return tf.reduce_mean(tf.abs(ground_truth_depth - predicted_depth)**2)

    return depth_loss


def make_style_loss_function(loss_feature_extractor_model: StyleLossModelBase, input_shape, output_shape):
    loss_feature_extractor_model.trainable = False
    inputs = {
        'content': tf.keras.Input(input_shape['content']),
        'style': tf.keras.Input(input_shape['style']),
        'prediction': tf.keras.Input(output_shape),
    }
    # perform feature extraction on the input content image and diff it against the output features
    input_content, input_style, input_prediction = inputs['content'], inputs['style'], inputs['prediction']

    assert input_style.shape[1] == 1, \
        f"Loss model does not support multiple styles. Found {input_style.shape[1]} in shape {input_style.shape}"

    single_input_style = tf.squeeze(input_style, axis=1)

    ground_truth_final_image = input_content[..., :3]

    loss_data_content = loss_feature_extractor_model(ground_truth_final_image)
    loss_data_style = loss_feature_extractor_model(single_input_style)
    loss_data_prediction = loss_feature_extractor_model(input_prediction)
    input_feature_values: tf.Tensor = loss_data_content['content']
    input_style_features: tf.Tensor = loss_data_style['style']
    output_feature_values: tf.Tensor = loss_data_prediction['content']
    output_style_features: tf.Tensor = loss_data_prediction['style']

    feature_loss = tf.reduce_mean([
        tf.nn.l2_loss((tf.cast(out_value, tf.float32) - tf.cast(in_value, tf.float32))) *
        loss_feature_extractor_model.content_loss_factor
        for (input_layer, in_value), (out_layer, out_value)
        in zip(input_feature_values.items(), output_feature_values.items())
    ])

    gram_matrix_model = get_gram_matrix_model((None, None, None))
    style_loss = tf.reduce_mean([
        tf.nn.l2_loss((gram_matrix_model(out_value) - gram_matrix_model(in_value))) *
        loss_feature_extractor_model.style_loss_factor
        for (input_layer, in_value), (out_layer, out_value)
        in zip(input_style_features.items(), output_style_features.items())
    ])

    total_variation_loss = tf.image.total_variation(input_prediction, 'total_variation_loss') * \
                           loss_feature_extractor_model.total_variation_loss_factor

    depth_loss = get_depth_loss_func(output_shape[:-1])(ground_truth_final_image, input_prediction) * \
                 loss_feature_extractor_model.depth_loss_factor

    total_loss = tf.cast(feature_loss, tf.float32) + \
                 tf.cast(style_loss, tf.float32) + \
                 tf.cast(total_variation_loss, tf.float32) + \
                 depth_loss
    output = {
        "loss": total_loss,
        "feature_loss": feature_loss,
        "style_loss": style_loss,
        "total_variation_loss": total_variation_loss,
        "depth_loss": depth_loss,
    }
    model = tf.keras.Model(inputs, output, name="StyleLoss")
    model.trainable = False

    def compute_loss(x: tf.Tensor, y_pred: tf.Tensor):
        return model({
            'content': x['content'],
            'style': x['style'],
            'prediction': y_pred,
        })

    return compute_loss, model
