"""
Test for SSGAN specific functions at the discriminator.
"""
import pytest
import math
import torch

from torch_mimicry.nets.infomax_gan.infomax_gan_base import BaseDiscriminator


class ExampleDiscriminator(BaseDiscriminator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return


class TestInfoMaxGANBase:
    def setup(self):
        self.ndf = 16
        self.nrkhs = 32
        self.N = 4
        self.netD = ExampleDiscriminator(ndf=self.ndf, nrkhs=self.nrkhs)

    def test_infonce_loss(self):
        l = torch.ones(self.N, self.nrkhs, 1)
        m = torch.ones(self.N, self.nrkhs, 1)

        loss = self.netD.infonce_loss(l=l, m=m)
        prob = math.exp(-1 * loss.item())

        assert type(loss.item()) == float

        # 1/4 probability
        assert abs(prob - 0.25) < 1e-2

    def test_compute_infomax_loss(self):
        with pytest.raises(ValueError):
            local_feat = torch.ones(self.N, self.ndf, 4, 4)
            global_feat = torch.ones(self.N, self.nrkhs)
            scale = 0.2
            self.netD.compute_infomax_loss(local_feat, global_feat, scale)

    def teardown(self):
        del self.netD


if __name__ == "__main__":
    test = TestInfoMaxGANBase()
    test.setup()
    test.test_infonce_loss()
    test.test_compute_infomax_loss()
    test.teardown()
