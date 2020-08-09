import pytest
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel

from torch_mimicry.metrics.kid import kid_utils


class TestKID:
    def setup(self):
        self.codes_g = np.ones((4, 16))
        self.codes_r = np.ones((4, 16))

    def test_polynomial_mmd(self):
        score = kid_utils.polynomial_mmd(codes_g=self.codes_g,
                                         codes_r=self.codes_r)

        assert type(score) == np.float64
        assert score < 1e-5

    def test_polynomial_mmd_averages(self):

        scores = kid_utils.polynomial_mmd_averages(codes_g=self.codes_g,
                                                   codes_r=self.codes_r,
                                                   n_subsets=4,
                                                   subset_size=1)

        assert len(scores) == 4
        assert type(scores[0]) == np.float64

    def test_compute_mmd2(self):
        X = self.codes_g
        Y = self.codes_r
        K_XX = polynomial_kernel(X)
        K_YY = polynomial_kernel(Y)
        K_XY = polynomial_kernel(X, Y)

        mmd_est_args = ['u-statistic', 'unbiased']

        for mmd_est in mmd_est_args:
            for unit_diagonal in [True, False]:
                mmd2_score = kid_utils._compute_mmd2(
                    K_XX=K_XX,
                    K_YY=K_YY,
                    K_XY=K_XY,
                    mmd_est=mmd_est,
                    unit_diagonal=unit_diagonal)

                assert type(mmd2_score) == np.float64

        # Input checks
        with pytest.raises(ValueError):
            kid_utils._compute_mmd2(K_XX=K_XX,
                                    K_YY=K_YY,
                                    K_XY=K_XY,
                                    mmd_est='invalid_option',
                                    unit_diagonal=unit_diagonal)

        m = K_XX.shape[0]
        with pytest.raises(ValueError):
            bad_K_XX = np.ones((m + 1, m + 1))
            kid_utils._compute_mmd2(K_XX=bad_K_XX,
                                    K_YY=K_YY,
                                    K_XY=K_XY,
                                    mmd_est='unbiased',
                                    unit_diagonal=unit_diagonal)

        with pytest.raises(ValueError):
            bad_K_YY = np.ones((m + 1, m + 1))
            kid_utils._compute_mmd2(K_XX=K_XX,
                                    K_YY=bad_K_YY,
                                    K_XY=K_XY,
                                    mmd_est='unbiased',
                                    unit_diagonal=unit_diagonal)

        with pytest.raises(ValueError):
            bad_K_XY = np.ones((m + 1, m + 1))
            kid_utils._compute_mmd2(K_XX=K_XX,
                                    K_YY=K_YY,
                                    K_XY=bad_K_XY,
                                    mmd_est='unbiased',
                                    unit_diagonal=unit_diagonal)

    def teardown(self):
        del self.codes_g
        del self.codes_r


if __name__ == "__main__":
    test = TestKID()
    test.setup()
    test.test_polynomial_mmd()
    test.test_polynomial_mmd_averages()
    test.test_compute_mmd2()
    test.teardown()
