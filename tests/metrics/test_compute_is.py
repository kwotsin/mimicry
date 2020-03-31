import torch

from torch_mimicry.metrics import compute_is
from torch_mimicry.nets.gan import gan


class ExampleGen(gan.BaseGenerator):
    def __init__(self,
                 bottom_width=4,
                 nz=4,
                 ngf=256,
                 loss_type='gan',
                 *args,
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         *args,
                         **kwargs)

    def forward(self, x):
        output = torch.ones(x.shape[0], 3, 32, 32)

        return output


class TestComputeIS:
    def setup(self):
        self.device = torch.device('cpu')
        self.netG = ExampleGen()

    def test_compute_inception_score(self):
        mean, std = compute_is.inception_score(netG=self.netG,
                                               device=self.device,
                                               num_samples=10,
                                               batch_size=10)

        assert type(mean) == float
        assert type(std) == float

    def teardown(self):
        del self.netG


if __name__ == "__main__":
    test = TestComputeIS()
    test.setup()
    test.test_compute_inception_score()
