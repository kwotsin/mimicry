"""
Implementation of DCGAN based on Kurach et al. specifically for CIFAR-10. 
The main difference with dcgan_32 is in using sigmoid 
as the final activation for the generator instead of tanh.

To reproduce scores, CIFAR-10 images should not be normalized from -1 to 1, and should
instead have values from 0 to 1, which is the default when loading images as np arrays.
"""
import torch
import torch.nn as nn

from torch_mimicry.nets.dcgan import dcgan_base
from torch_mimicry.modules.resblocks import DBlockOptimized, DBlock, GBlock


class DCGANGeneratorCIFAR(dcgan_base.DCGANBaseGenerator):
    r"""
    ResNet backbone generator for ResNet DCGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self, nz=128, ngf=256, bottom_width=4, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        # Build the layers
        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block3 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block4 = GBlock(self.ngf, self.ngf, upsample=True)
        self.b5 = nn.BatchNorm2d(self.ngf)
        self.c5 = nn.Conv2d(self.ngf, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.

        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).

        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = torch.sigmoid(self.c5(h))

        return h


class DCGANDiscriminatorCIFAR(dcgan_base.DCGANBaseDiscriminator):
    r"""
    ResNet backbone discriminator for ResNet DCGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self, ndf=128, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        # Build layers
        self.block1 = DBlockOptimized(3, self.ndf, spectral_norm=False)
        self.block2 = DBlock(self.ndf,
                             self.ndf,
                             downsample=True,
                             spectral_norm=False)
        self.block3 = DBlock(self.ndf,
                             self.ndf,
                             downsample=False,
                             spectral_norm=False)
        self.block4 = DBlock(self.ndf,
                             self.ndf,
                             downsample=False,
                             spectral_norm=False)
        self.l5 = nn.Linear(self.ndf, 1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)

        # Global mean pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l5(h)

        return output
