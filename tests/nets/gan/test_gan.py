import pytest
import torch
import torch.nn as nn

from torch_mimicry.nets.gan.gan import BaseGenerator, BaseDiscriminator


class ExampleGenerator(BaseGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(1, 1)

    def forward(self, x):
        return torch.ones(x.shape[0], 3, 32, 32)


class ExampleDiscriminator(BaseDiscriminator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(1, 1)

    def forward(self, x):
        return torch.ones(x.shape[0])


class TestBaseGAN:
    def setup(self):
        self.N = 1
        self.device = "cpu"
        self.real_label_val = 1.0
        self.fake_label_val = 0.0

        self.nz = 16
        self.ngf = 16
        self.ndf = 16
        self.bottom_width = 4
        self.loss_type = 'gan'

        self.netG = ExampleGenerator(ngf=self.ngf,
                                     bottom_width=self.bottom_width,
                                     nz=self.nz,
                                     loss_type=self.loss_type)
        self.netD = ExampleDiscriminator(ndf=self.ndf,
                                         loss_type=self.loss_type)
        self.output_fake = torch.ones(self.N, 1)
        self.output_real = torch.ones(self.N, 1)

    def test_generate_images(self):
        images = self.netG.generate_images(10)

        assert images.shape == (10, 3, 32, 32)
        assert images.device == self.netG.device

    def test_compute_GAN_loss(self):
        losses = ['gan', 'ns', 'hinge', 'wasserstein']

        for loss_type in losses:
            self.netG.loss_type = loss_type
            self.netD.loss_type = loss_type

            errG = self.netG.compute_gan_loss(output=self.output_fake)
            errD = self.netD.compute_gan_loss(output_real=self.output_real,
                                              output_fake=self.output_fake)

            assert type(errG.item()) == float
            assert type(errD.item()) == float

        with pytest.raises(ValueError):
            self.netG.loss_type = 'invalid'
            self.netG.compute_gan_loss(output=self.output_fake)

        with pytest.raises(ValueError):
            self.netD.loss_type = 'invalid'
            self.netD.compute_gan_loss(output_real=self.output_real,
                                       output_fake=self.output_fake)

    def teardown(self):
        del self.netG
        del self.netD
        del self.output_real
        del self.output_fake


if __name__ == "__main__":
    test = TestBaseGAN()
    test.setup()
    test.test_generate_images()
    test.test_compute_GAN_loss()
    test.teardown()
