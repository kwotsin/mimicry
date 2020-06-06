"""
Test functions for SSGAN for image size 48.
"""
import torch
import torch.optim as optim

from torch_mimicry.nets.ssgan.ssgan_48 import SSGANGenerator48, SSGANDiscriminator48
from torch_mimicry.training import metric_log
from torch_mimicry.utils import common


class TestSSGAN48:
    def setup(self):
        self.nz = 128
        self.N, self.C, self.H, self.W = (8, 3, 48, 48)
        self.ngf = 16
        self.ndf = 16

        self.netG = SSGANGenerator48(ngf=self.ngf)
        self.netD = SSGANDiscriminator48(ndf=self.ndf)

    def test_SSGANGenerator48(self):
        noise = torch.ones(self.N, self.nz)
        output = self.netG(noise)

        assert output.shape == (self.N, self.C, self.H, self.W)

    def test_SSGANDiscriminator48(self):
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
                                        device='cpu',
                                        log_data=log_data)

        log_data = self.netG.train_step(real_batch=real_batch,
                                        netD=self.netD,
                                        optG=optG,
                                        log_data=log_data,
                                        device='cpu')

        for name, metric_dict in log_data.items():
            assert type(name) == str
            assert type(metric_dict['value']) == float

    def teardown(self):
        del self.netG
        del self.netD


if __name__ == "__main__":
    test = TestSSGAN48()
    test.setup()
    test.test_SSGANGenerator48()
    test.test_SSGANDiscriminator48()
    test.test_train_steps()
    test.teardown()
