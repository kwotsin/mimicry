import pytest
import numpy as np
import tensorflow as tf

from torch_mimicry.metrics.fid import fid_utils
from torch_mimicry.metrics.inception_model import inception_utils


class TestFID:
    def setup(self):
        self.images = np.ones((4, 32, 32, 3))
        self.sess = tf.compat.v1.Session()

    def test_calculate_activation_statistics(self):
        inception_path = './metrics/inception_model'
        inception_utils.create_inception_graph(inception_path)

        mu, sigma = fid_utils.calculate_activation_statistics(
            images=self.images, sess=self.sess)

        assert mu.shape == (2048, )
        assert sigma.shape == (2048, 2048)

    def test_calculate_frechet_distance(self):
        mu1, sigma1 = np.ones((16, )), np.ones((16, 16))
        mu2, sigma2 = mu1 * 2, sigma1 * 2

        score = fid_utils.calculate_frechet_distance(mu1=mu1,
                                                     mu2=mu2,
                                                     sigma1=sigma1,
                                                     sigma2=sigma2)

        assert type(score) == np.float64

        # Inputs check
        bad_mu2, bad_sigma2 = np.ones((15, 15)), np.ones((15, 15))
        with pytest.raises(ValueError):
            fid_utils.calculate_frechet_distance(mu1=mu1,
                                                 mu2=bad_mu2,
                                                 sigma1=sigma1,
                                                 sigma2=bad_sigma2)

    def teardown(self):
        del self.images
        self.sess.close()


if __name__ == "__main__":
    test = TestFID()
    test.setup()
    test.test_calculate_activation_statistics()
    test.test_calculate_frechet_distance()
    test.teardown()
