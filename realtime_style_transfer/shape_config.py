from realtime_style_transfer.models import stylePrediction


class ShapeConfig:
    def __init__(self, num_styles=1, hdr=True, bottleneck_res_y=120, bottleneck_num_filters=32, resolution_divider=2,
                 num_channels=17):
        self.bottleneck_res_y = bottleneck_res_y
        self.bottleneck_num_filters = bottleneck_num_filters
        self.num_styles = num_styles
        self.channels = self._get_channels_from_number(num_channels)

        num_channels = sum(map(lambda c: c[1], self.channels))
        self.num_channels = num_channels
        input_dimensions = (960 // resolution_divider, 1920 // resolution_divider)
        output_dimensions = (960 // resolution_divider, 1920 // resolution_divider)
        self.output_shape = output_dimensions + (3,)
        image_shape = input_dimensions + (3,)
        self.image_shape = image_shape
        sdr_input_shape = {'content': image_shape,
                           'style': (num_styles,) + self.output_shape}
        hdr_input_shape = {'content': input_dimensions + (num_channels,),
                           'style': (num_styles,) + self.output_shape}

        self.input_shape = hdr_input_shape if hdr else sdr_input_shape

        if num_styles > 1:
            self.input_shape['style_weights'] = output_dimensions + (num_styles - 1,),

        self.style_feature_extractor_type = stylePrediction.StyleFeatureExtractor.MOBILE_NET
        self.with_depth_loss = True

    @staticmethod
    def from_spec(spec: str, num_styles=1, hdr=True):
        """

        :param spec: something like rst-960-120-128-17
        :return: None
        """
        parts = spec.split('-')

        prefix = parts[0]
        res_x = int(parts[1])
        bottleneck_res_y = int(parts[2])
        bottleneck_num_filters = int(parts[3])
        num_channels = int(parts[4])

        return ShapeConfig(num_styles, hdr, bottleneck_res_y, bottleneck_num_filters, 1920 // res_x,
                           num_channels)

    def __str__(self):
        import json
        return json.dumps(self.__dict__, indent=4)

    def _get_channels_from_number(self, num_channels):
        channels = [
            ("FinalImage", 3),
        ]
        if num_channels > 3:
            channels += [("BaseColor", 3)]
        if num_channels >= 18:
            channels += [("ShadowMask", 1)]
        if num_channels >= 17:
            channels += [
                ("AmbientOcclusion", 1),
                ("Metallic", 1),
                ("Specular", 1),
                ("Roughness", 1),
                ("ViewNormal", 3),
                ("SceneDepth", 1),
                ("LightingModel", 3),
            ]

        return channels
