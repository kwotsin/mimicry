"""
Test functions for cGAN-PD for image size 128.
"""
import torch
import torch.optim as optim

from torch_mimicry.nets.cgan_pd.cgan_pd_128 import CGANPDGenerator128, CGANPDDiscriminator128
from torch_mimicry.training import metric_log
from torch_mimicry.utils import common


class TestCGANPD128:
    def setup(self):
        self.num_classes = 10
        self.nz = 128
        self.N, self.C, self.H, self.W = (4, 3, 128, 128)

        self.noise = torch.ones(self.N, self.nz)
        self.images = torch.ones(self.N, self.C, self.H, self.W)
        self.Y = torch.randint(low=0, high=self.num_classes, size=(self.N, ))

        self.ngf = 16
        self.ndf = 16

        self.netG = CGANPDGenerator128(num_classes=self.num_classes,
                                       ngf=self.ngf)
        self.netD = CGANPDDiscriminator128(num_classes=self.num_classes,
                                           ndf=self.ndf)

    def test_CGANPDGenerator128(self):
        images = self.netG(self.noise, self.Y)
        assert images.shape == (self.N, self.C, self.H, self.W)

        images = self.netG(self.noise, None)
        assert images.shape == (self.N, self.C, self.H, self.W)

    def test_CGANPDDiscriminator128(self):
        output = self.netD(self.images, self.Y)

        assert output.shape == (self.N, 1)

    def test_train_steps(self):
        real_batch = common.load_images(self.N)

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
        del self.noise
        del self.images
        del self.Y
        del self.netG
        del self.netD


if __name__ == "__main__":
    test = TestCGANPD128()
    test.setup()
    test.test_CGANPDGenerator128()
    test.test_CGANPDDiscriminator128()
    test.test_train_steps()
    test.teardown()
