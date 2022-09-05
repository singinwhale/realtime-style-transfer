from tensorflow import keras
import tensorflow as tf

import logging

log = logging.getLogger(__name__)


def get_gram_matrix_model(input_spec):
    inputs = tf.keras.Input(input_spec)
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
    num_style_layers: int = None
    feature_extractor: tf.keras.Model

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainable = False

    def call(self, inputs, **kwargs):
        outputs = self.feature_extractor(inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


class StyleLossModelVGG(StyleLossModelBase):
    """## Extract style and content
    When called on an image, this model returns the gram matrix (style) of the `style_layers` and content of the `content_layers`
    Build a model that returns the style and content tensors.
    """

    def __init__(self):
        super().__init__(name='StyleLossModelVGG')

        content_layers = ['block5_pool']
        # style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        style_layers = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool']

        # Load our model. Load pretrained VGG, trained on ImageNet data
        vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in (content_layers + style_layers)]

        self.feature_extractor = tf.keras.Model([vgg.input], outputs)
        self.feature_extractor.trainable = False

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.content_loss_factor = 1e-4
        self.style_loss_factor = 1e-6

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
                                                                         input_shape=input_shape)
        efficientnet.trainable = False

        outputs = [efficientnet.get_layer(name).output for name in output_layer_names]

        self.feature_extractor = tf.keras.Model([efficientnet.input], outputs)
        self.feature_extractor.trainable = False

        self.style_layers = style_layer_names
        self.content_layers = content_layer_names
        self.num_style_layers = len(style_layer_names)

    def call(self, inputs, **kwargs):
        """Expects float input in [0,1]
        :param **kwargs:
        """
        inputs = inputs * 255.0
        return super().call(inputs, **kwargs)


class StyleLossModelMobileNet(StyleLossModelBase):

    def __init__(self, input_shape, name="StyleLossModelMobileNet"):
        super().__init__(name=name)

        style_layer_names = [
            'expanded_conv_2/Add',
            'expanded_conv_4/Add',
            'expanded_conv_5/Add',
            'expanded_conv_7/Add',
            'expanded_conv_9/Add',
        ]
        content_layer_names = [
            'expanded_conv_10/Add',
        ]
        output_layer_names = style_layer_names + content_layer_names

        mobile_net = tf.keras.applications.MobileNetV3Small(include_top=False, input_shape=input_shape)
        mobile_net.trainable = False

        outputs = [mobile_net.get_layer(name).output for name in output_layer_names]

        self.feature_extractor = tf.keras.Model([mobile_net.input], outputs)
        self.feature_extractor.trainable = False

        self.style_layers = style_layer_names
        self.content_layers = content_layer_names
        self.num_style_layers = len(style_layer_names)
        self.content_loss_factor = 1e-2
        self.style_loss_factor = 5e-1

    def call(self, inputs, **kwargs):
        """Expects float input in [0,1]
        """
        inputs = inputs * 255.0
        return super().call(inputs)


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

        self.feature_extractor = tf.keras.Model(inputs, [output1, output2])
        self.feature_extractor.trainable = False

        self.num_style_layers = 1
        self.content_loss_factor = 1
        self.style_loss_factor = 1


def make_style_loss_function(loss_model: keras.Model, input_shape, output_shape):
    loss_model.trainable = False
    inputs = {
        'content': tf.keras.Input(input_shape['content']),
        'style': tf.keras.Input(input_shape['style']),
        'prediction': tf.keras.Input(output_shape),
    }
    # perform feature extraction on the input content image and diff it against the output features
    input_content, input_style = inputs['content'], inputs['style']
    loss_data_content = loss_model(input_content)
    loss_data_style = loss_model(input_style)
    loss_data_prediction = loss_model(inputs['prediction'])
    input_feature_values: tf.Tensor = loss_data_content['content']
    input_style_features: tf.Tensor = loss_data_style['style']
    output_feature_values: tf.Tensor = loss_data_prediction['content']
    output_style_features: tf.Tensor = loss_data_prediction['style']

    feature_loss = tf.reduce_mean([tf.nn.l2_loss((out_value - in_value) * loss_model.content_loss_factor) for (input_layer, in_value), (out_layer, out_value) in
                                   zip(input_feature_values.items(), output_feature_values.items())])

    gram_matrix_model = get_gram_matrix_model((None, None, None))
    style_loss = tf.reduce_mean([tf.nn.l2_loss((gram_matrix_model(out_value) - gram_matrix_model(in_value)) * loss_model.style_loss_factor) for
                                 (input_layer, in_value), (out_layer, out_value) in
                                 zip(input_style_features.items(), output_style_features.items())])

    output = {
        "loss": feature_loss + style_loss,
        "feature_loss": feature_loss,
        "style_loss": style_loss,
    }
    model = tf.keras.Model(inputs, output, name="StyleLoss")
    model.trainable = False

    def compute_loss(x: tf.Tensor, y_pred: tf.Tensor):
        return model({
            'content': x['content'],
            'style': x['style'],
            'prediction': y_pred,
        })

    return compute_loss
