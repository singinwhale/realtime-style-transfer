from realtime_style_transfer.models import stylePrediction

class ShapeConfig:
    def __init__(self, num_styles=1, hdr=True):
        resolution_divider = 2
        self.num_styles = num_styles
        self.channels = [
            ("FinalImage", 3),
            ("BaseColor", 3),
            # ("ShadowMask", 1),
            ("AmbientOcclusion", 1),
            ("Metallic", 1),
            ("Specular", 1),
            ("Roughness", 1),
            ("ViewNormal", 3),
            ("SceneDepth", 1),
            ("LightingModel", 3),
        ]
        num_channels = sum(map(lambda c: c[1], self.channels))
        self.num_channels = num_channels
        image_dimensions = (960 // resolution_divider, 1920 // resolution_divider)
        image_shape = image_dimensions + (3,)
        self.image_shape = image_shape
        self.sdr_input_shape = {'content': image_shape,
                                'style_weights': image_dimensions + (num_styles - 1,),
                                'style': (num_styles,) + image_shape}
        self.hdr_input_shape = {'content': image_dimensions + (num_channels,),
                                'style_weights': image_dimensions + (num_styles - 1,),
                                'style': (num_styles,) + image_shape}

        self.output_shape = (960 // resolution_divider, 1920 // resolution_divider, 3)
        self.input_shape = self.hdr_input_shape if hdr else self.sdr_input_shape
        self.style_feature_extractor_type = stylePrediction.StyleFeatureExtractor.MOBILE_NET

    def __str__(self):
        import json
        return json.dumps(self.__dict__, indent=4)
