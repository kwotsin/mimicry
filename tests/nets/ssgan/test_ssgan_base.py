"""
Test for SSGAN specific functions at the discriminator.
"""
import pytest
import torch

from torch_mimicry.nets.ssgan.ssgan_base import SSGANBaseDiscriminator
from torch_mimicry.utils import common


class ExampleDiscriminator(SSGANBaseDiscriminator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.ones(x.shape[0])


class TestSSGANBase:
    def setup(self):
        self.netD = ExampleDiscriminator(ndf=16)

    def test_rot_tensor(self):
        # Load image and model
        image, _ = common.load_images(1, size=32)

        # For any rotation, after performing the same action 4 times,
        # you should return to the same pixel value
        for deg in [0, 90, 180, 270]:
            x = image.clone()
            for _ in range(4):
                x = self.netD._rot_tensor(x, deg)

            assert torch.sum((x - image)**2) < 1e-5

    def test_rotate_batch(self):
        # Load image and model
        images, _ = common.load_images(8, size=32)

        check = images.clone()
        check, labels = self.netD._rotate_batch(check)
        degrees = [0, 90, 180, 270]

        # Rotate 3 more times to get back to original.
        for i in range(check.shape[0]):
            for _ in range(3):
                check[i] = self.netD._rot_tensor(check[i], degrees[labels[i]])

        assert torch.sum((images - check)**2) < 1e-5

        with pytest.raises(ValueError):
            self.netD._rot_tensor(check[i], 9999)

    def teardown(self):
        del self.netD


if __name__ == "__main__":
    test = TestSSGANBase()
    test.setup()
    test.test_rot_tensor()
    test.test_rotate_batch()
    test.teardown()
