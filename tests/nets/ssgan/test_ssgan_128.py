"""
Test functions for SSGAN for image size 128.
"""
import torch
import torch.optim as optim

from torch_mimicry.nets.ssgan.ssgan_128 import SSGANGenerator128, SSGANDiscriminator128
from torch_mimicry.training import metric_log
from torch_mimicry.utils import common


class TestSSGAN128:
    def setup(self):
        self.nz = 128
        self.N, self.C, self.H, self.W = (8, 3, 128, 128)
        self.ngf = 16
        self.ndf = 16
        self.device = 'cpu'

        self.netG = SSGANGenerator128(ngf=self.ngf)
        self.netD = SSGANDiscriminator128(ndf=self.ndf)

    def test_SSGANGenerator128(self):
        noise = torch.ones(self.N, self.nz)
        output = self.netG(noise)

        assert output.shape == (self.N, self.C, self.H, self.W)

    def test_SSGANDiscriminator128(self):
        images = torch.ones(self.N, self.C, self.H, self.W)
        output, labels = self.netD(images)

        assert output.shape == (self.N, 1)
        assert labels.shape == (self.N, 4)

    def test_train_steps(self):
        real_batch = common.load_images(self.N, size=self.H)

        # Setup optimizers
        optD = optim.Adam(self.netD.parameters(), 2e-4, betas=(0.0, 0.9))
        optG = optim.Adam(self.netG.parameters(), 2e-4, betas=(0.0, 0.9))

        # Log statistics to check
        log_data = metric_log.MetricLog()

        # Test D train step
        log_data = self.netD.train_step(real_batch=real_batch,
                                        netG=self.netG,
                                        optD=optD,
                                        device=self.device,
                                        log_data=log_data)

        log_data = self.netG.train_step(real_batch=real_batch,
                                        netD=self.netD,
                                        optG=optG,
                                        log_data=log_data,
                                        device=self.device)

        for name, metric_dict in log_data.items():
            assert type(name) == str
            assert type(metric_dict['value']) == float

    def teardown(self):
        del self.netG
        del self.netD


if __name__ == "__main__":
    test = TestSSGAN128()
    test.setup()
    test.test_SSGANGenerator128()
    test.test_SSGANDiscriminator128()
    test.test_train_steps()
    test.teardown()
