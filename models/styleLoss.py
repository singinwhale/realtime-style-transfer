from tensorflow import keras
import tensorflow as tf

import logging
log = logging.getLogger(__name__)

def vgg_layers(layer_names):
    """#### Intermediate layers for style and content

    So why do these intermediate outputs within our pretrained image classification network allow us to define style and content representations?

    At a high level, in order for a network to perform image classification (which this network has been trained to do),
     it must understand the image. This requires taking the raw image as input pixels and building an internal representation
     that converts the raw image pixels into a complex understanding of the features present within the image.

    This is also a reason why convolutional neural networks are able to generalize well:
    they’re able to capture the invariances and defining features within classes (e.g. cats vs. dogs)
    that are agnostic to background noise and other nuisances.
    Thus, somewhere between where the raw image is fed into the model and the output classification label,
    the model serves as a complex feature extractor. By accessing intermediate layers of the model,
    you're able to describe the content and style of input images.

    ## Build the model

    The networks in `tf.keras.applications` are designed so you can easily extract the intermediate layer values using the Keras functional API.

    To define a model using the functional API, specify the inputs and outputs:

    `model = Model(inputs, outputs)`

    This following function builds a VGG19 model that returns a list of intermediate layer outputs:
    Creates a VGG model that returns a list of intermediate output values.
    """
    # Load our model. Load pretrained VGG, trained on ImageNet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


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


def style_content_loss(outputs, num_style_layers, style_targets, content_targets):
    """To optimize this, use a weighted combination of the two losses to get the total loss"""
    style_weight = 1e4
    content_weight = 1e-2

    style_outputs = outputs['style']
    content_outputs = outputs['content']
    per_layer_style_losses = [tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in
                              style_outputs.keys()]

    style_loss = tf.add_n(per_layer_style_losses)
    style_loss *= style_weight / num_style_layers

    per_layer_content_losses = [tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in
                                content_outputs.keys()]

    content_loss = tf.add_n(per_layer_content_losses)
    content_loss *= content_weight / num_style_layers
    loss = style_loss + content_loss
    return loss


class StyleLossModelBase(tf.keras.models.Model):
    num_style_layers: int = None

    def __init__(self, *args, **kwargs):
        super(StyleLossModelBase, self).__init__(*args, **kwargs)
        self.trainable = False

    def call(self, inputs):
        outputs = self.feature_extractor(inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_outputs, 'style': style_outputs}


class StyleLossModelVGG(StyleLossModelBase):
    """## Extract style and content
    When called on an image, this model returns the gram matrix (style) of the `style_layers` and content of the `content_layers`
    Build a model that returns the style and content tensors.
    """

    def __init__(self):
        super(StyleLossModelVGG, self).__init__(name='StyleLossModelVGG')

        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

        self.vgg = vgg_layers(style_layers + content_layers)
        self.vgg.trainable = False

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        return super(StyleLossModelVGG, self).call(preprocessed_input)


class StyleLossModelEfficientNet(StyleLossModelBase):
    """## Extract style and content
    When called on an image, this model returns the gram matrix (style) of the `style_layers` and content of the `content_layers`
    Build a model that returns the style and content tensors.
    """

    def __init__(self, input_shape, name="StyleLossModelEfficientNet"):
        super(StyleLossModelEfficientNet, self).__init__(name=name)

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
                                                                         input_shape=input_shape[-3:])
        efficientnet.trainable = False

        outputs = [efficientnet.get_layer(name).output for name in output_layer_names]

        self.feature_extractor = tf.keras.Model([efficientnet.input], outputs)
        self.feature_extractor.trainable = False

        self.style_layers = style_layer_names
        self.content_layers = content_layer_names
        self.num_style_layers = len(style_layer_names)

    def call(self, inputs):
        """Expects float input in [0,1]"""
        inputs = inputs * 255.0
        return super(StyleLossModelEfficientNet, self).call(inputs)


class StyleLossModelMobileNet(StyleLossModelBase):

    def __init__(self, input_shape, name="StyleLossModelMobileNet"):
        super(StyleLossModelMobileNet, self).__init__(name=name)

        style_layer_names = [
            'expanded_conv_2/Add',
            'expanded_conv_4/Add',
            'expanded_conv_5/Add',
        ]
        content_layer_names = [
            'expanded_conv_7/Add',
            'expanded_conv_9/Add',
            'expanded_conv_10/Add',
        ]
        output_layer_names = style_layer_names + content_layer_names

        # Load our model. Load pretrained VGG, trained on ImageNet data
        mobile_net = tf.keras.applications.MobileNetV3Small(include_top=False, input_shape=input_shape[-3:])
        mobile_net.trainable = False

        outputs = [mobile_net.get_layer(name).output for name in output_layer_names]

        self.feature_extractor = tf.keras.Model([mobile_net.input], outputs)
        self.feature_extractor.trainable = False

        self.style_layers = style_layer_names
        self.content_layers = content_layer_names
        self.num_style_layers = len(style_layer_names)

    def call(self, inputs):
        """Expects float input in [0,1]"""
        inputs = inputs * 255.0
        return super(StyleLossModelMobileNet, self).call(inputs)


def style_loss(loss_model: keras.Model, x: tf.Tensor, y_pred: tf.Tensor):
    # perform feature extraction on the input content image and diff it against the output features
    input_content, input_style = x['content'], x['style']
    input_feature_values: tf.Tensor = loss_model(input_content)['content']
    output_feature_values: tf.Tensor = loss_model(y_pred)['content']

    feature_loss = sum([tf.nn.l2_loss(out_value - in_value) for out_value, in_value in
                        zip(input_feature_values, output_feature_values)])

    input_style_gram_value = gram_matrix(input_style)
    output_gram_value = gram_matrix(y_pred)
    gram_loss = tf.nn.l2_loss(output_gram_value - input_style_gram_value)

    return feature_loss + gram_loss
