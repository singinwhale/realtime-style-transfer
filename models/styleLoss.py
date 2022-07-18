import tensorflow as tf


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


class StyleLossModelVGG(tf.keras.models.Model):
    """## Extract style and content
    When called on an image, this model returns the gram matrix (style) of the `style_layers` and content of the `content_layers`
    Build a model that returns the style and content tensors.
    """

    def __init__(self):
        super(StyleLossModelVGG, self).__init__()

        content_layers = ['block5_conv2']
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


class StyleLossModelEfficientNet(tf.keras.models.Model):
    """## Extract style and content
    When called on an image, this model returns the gram matrix (style) of the `style_layers` and content of the `content_layers`
    Build a model that returns the style and content tensors.
    """

    def __init__(self):
        super(StyleLossModelEfficientNet, self).__init__()

        style_layer_names = ['block2c_add',
                             'block3c_add',
                             'block4e_add',
                             ]
        content_layer_names = ['block6f_add',
                               'block7b_add',
                               'block5e_add']
        output_layer_names = style_layer_names + content_layer_names

        # Load our model. Load pretrained VGG, trained on ImageNet data
        efficientnet = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False)
        efficientnet.trainable = False

        outputs = [efficientnet.get_layer(name).output for name in output_layer_names]

        self.efficientNet = tf.keras.Model([efficientnet.input], outputs)
        self.style_layers = style_layer_names
        self.content_layers = content_layer_names
        self.num_style_layers = len(style_layer_names)
        self.efficientNet.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0

        outputs = self.efficientNet(inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}
