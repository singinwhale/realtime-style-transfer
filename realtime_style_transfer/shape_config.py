from realtime_style_transfer.models import stylePrediction

class ShapeConfig:
    def __init__(self, num_styles=1, hdr=True):
        self.bottleneck_res_y = 32
        self.bottleneck_num_filters = 128
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
        input_dimensions = (960 // resolution_divider, 1920 // resolution_divider)
        output_dimensions = (960, 1920)
        self.output_shape = output_dimensions + (3,)
        image_shape = input_dimensions + (3,)
        self.image_shape = image_shape
        self.sdr_input_shape = {'content': image_shape,
                                'style_weights': output_dimensions + (num_styles - 1,),
                                'style': (num_styles,) + self.output_shape}
        self.hdr_input_shape = {'content': input_dimensions + (num_channels,),
                                'style_weights': output_dimensions + (num_styles - 1,),
                                'style': (num_styles,) + self.output_shape}

        self.input_shape = self.hdr_input_shape if hdr else self.sdr_input_shape
        self.style_feature_extractor_type = stylePrediction.StyleFeatureExtractor.MOBILE_NET

    def __str__(self):
        import json
        return json.dumps(self.__dict__, indent=4)
