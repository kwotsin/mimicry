import pytest
import numpy as np
import torch

from torch_mimicry.metrics.inception_model import inception_utils
from torch_mimicry.metrics.inception_score import inception_score_utils


class TestInceptionScore:
    def test_get_predictions(self):
        inception_utils.create_inception_graph('./metrics/inception_model')

        images = np.ones((4, 32, 32, 3))
        preds = inception_score_utils.get_predictions(images)
        assert preds.shape == (4, 1008)

        preds = inception_score_utils.get_predictions(
            images, device=torch.device('cpu'))
        assert preds.shape == (4, 1008)

    def test_get_inception_score(self):
        images = np.ones((4, 32, 32, 3))
        mean, std = inception_score_utils.get_inception_score(images)

        assert type(mean) == float
        assert type(std) == float

        with pytest.raises(ValueError):
            images *= -1
            inception_score_utils.get_inception_score(images)


if __name__ == "__main__":
    test = TestInceptionScore()
    test.test_get_predictions()
    test.test_get_inception_score()
