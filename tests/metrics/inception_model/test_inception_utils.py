import numpy as np
import tensorflow as tf

from torch_mimicry.metrics.inception_model import inception_utils


class TestInceptionUtils:
    def test_get_activations(self):
        for inception_path in ['./metrics/inception_model', None]:
            inception_utils.create_inception_graph(inception_path)

            images = np.ones((4, 32, 32, 3))
            with tf.compat.v1.Session() as sess:
                feat = inception_utils.get_activations(images=images,
                                                       sess=sess)

                assert feat.shape == (4, 2048)


if __name__ == "__main__":
    test = TestInceptionUtils()
    test.test_get_activations()
