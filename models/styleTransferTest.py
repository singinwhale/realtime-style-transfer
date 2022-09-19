import numpy.testing

from .styleTransfer import *
from . import styleTransfer
import numpy as np
import unittest


def _generate_vertical_gradient_tensor(min_max, shape):
    #
    assert len(shape) == 4, "shape needs to be of kind 'byxc'"
    gradient = list()
    for b in range(shape[0]):
        for i in range(shape[1]):
            y = list()
            for j in range(shape[2]):
                x = min_max[0] + (i / shape[0]) * (min_max[1] - min_max[0])
                y.append(x)
            gradient.append(y)

    return tf.constant(gradient, tf.float32, shape, 'Gradient')


class StyleTransferTests(unittest.TestCase):
    def test_apply_style_weights(self):
        style_weights = _generate_vertical_gradient_tensor((0, 1), (2, 10, 20, 1))
        style_params = tf.constant([
            [[[10, 20, 30, 40, 50, 60], [70, 80, 90, 100, 110, 120]]],
            [[[10, 20, 30, 40, 50, 60], [70, 80, 90, 100, 110, 120]]],
        ], dtype=tf.float32, shape=(2, 1, 2, 6))
        weighted_style_params = styleTransfer._apply_style_weights(style_weights, style_params).numpy()
        expected_shape = (2, 10, 20, 6)
        self.assertTupleEqual(weighted_style_params.shape, expected_shape)

        expected_style_params = np.zeros(expected_shape)
        for b in range(expected_shape[0]):
            for x in range(expected_shape[1]):
                for y in range(expected_shape[2]):
                    for c in range(expected_shape[3]):
                        expected_style_params[b, x, y, c] = style_weights[b, x, y] * style_params[b, 0, 0, c] + \
                                                            (1.0 - style_weights[b, x, y]) * style_params[b, 0, 1, c]

        numpy.testing.assert_almost_equal(weighted_style_params, expected_style_params)
