import pytest
import torch
import torch.nn as nn

from torch_mimicry.nets.gan.cgan import BaseConditionalGenerator, BaseConditionalDiscriminator


class ExampleGenerator(BaseConditionalGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(1, 1)

    def forward(self, x, y):
        return torch.ones(x.shape[0], 3, 32, 32)


class ExampleDiscriminator(BaseConditionalDiscriminator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l1 = nn.Linear(1, 1)

    def forward(self, x, y):
        return torch.ones(x.shape[0])


class TestBaseGAN:
    def setup(self):
        self.N = 1
        self.device = "cpu"

        self.nz = 16
        self.ngf = 16
        self.ndf = 16
        self.bottom_width = 4
        self.num_classes = 10
        self.loss_type = 'gan'

        self.netG = ExampleGenerator(num_classes=self.num_classes,
                                     ngf=self.ngf,
                                     bottom_width=self.bottom_width,
                                     nz=self.nz,
                                     loss_type=self.loss_type)
        self.netD = ExampleDiscriminator(num_classes=self.num_classes,
                                         ndf=self.ndf,
                                         loss_type=self.loss_type)
        self.output_fake = torch.ones(self.N, 1)
        self.output_real = torch.ones(self.N, 1)

    def test_generate_images(self):
        with pytest.raises(ValueError):
            images = self.netG.generate_images(10, c=self.num_classes + 1)

        images = self.netG.generate_images(10)
        assert images.shape == (10, 3, 32, 32)
        assert images.device == self.netG.device

        images = self.netG.generate_images(10, c=0)
        assert images.shape == (10, 3, 32, 32)
        assert images.device == self.netG.device

    def test_generate_images_with_labels(self):
        with pytest.raises(ValueError):
            images, labels = self.netG.generate_images_with_labels(
                10, c=self.num_classes + 1)

        images, labels = self.netG.generate_images_with_labels(10)
        assert images.shape == (10, 3, 32, 32)
        assert images.device == self.netG.device
        assert labels.shape == (10, )
        assert labels.device == self.netG.device

        images, labels = self.netG.generate_images_with_labels(10, c=0)
        assert images.shape == (10, 3, 32, 32)
        assert images.device == self.netG.device
        assert labels.shape == (10, )
        assert labels.device == self.netG.device

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

    def teardown(self):
        del self.netG
        del self.netD
        del self.output_real
        del self.output_fake


if __name__ == "__main__":
    test = TestBaseGAN()
    test.setup()
    test.test_generate_images()
    test.test_generate_images_with_labels()
    test.test_compute_GAN_loss()
    test.teardown()
