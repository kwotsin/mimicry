"""
Implementation of SSGAN for image size 128.
"""
import torch
import torch.nn as nn

from torch_mimicry.modules.layers import SNLinear
from torch_mimicry.modules.resblocks import DBlockOptimized, DBlock, GBlock
from torch_mimicry.nets.ssgan import ssgan_base


class SSGANGenerator128(ssgan_base.SSGANBaseGenerator):
    r"""
    ResNet backbone generator for SSGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
        ss_loss_scale (float): Self-supervised loss scale for generator.
    """
    def __init__(self, nz=128, ngf=1024, bottom_width=4, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        # Build the layers
        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block3 = GBlock(self.ngf, self.ngf >> 1, upsample=True)
        self.block4 = GBlock(self.ngf >> 1, self.ngf >> 2, upsample=True)
        self.block5 = GBlock(self.ngf >> 2, self.ngf >> 3, upsample=True)
        self.block6 = GBlock(self.ngf >> 3, self.ngf >> 4, upsample=True)
        self.b7 = nn.BatchNorm2d(self.ngf >> 4)
        self.c7 = nn.Conv2d(self.ngf >> 4, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c7.weight.data, 1.0)

    def forward(self, x):
        """
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
        h = self.block5(h)
        h = self.block6(h)
        h = self.b7(h)
        h = self.activation(h)
        h = torch.tanh(self.c7(h))

        return h


class SSGANDiscriminator128(ssgan_base.SSGANBaseDiscriminator):
    r"""
    ResNet backbone discriminator for SSGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.        
        ss_loss_scale (float): Self-supervised loss scale for discriminator.        
    """
    def __init__(self, ndf=1024, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        # Build layers
        self.block1 = DBlockOptimized(3, self.ndf >> 4)
        self.block2 = DBlock(self.ndf >> 4, self.ndf >> 3, downsample=True)
        self.block3 = DBlock(self.ndf >> 3, self.ndf >> 2, downsample=True)
        self.block4 = DBlock(self.ndf >> 2, self.ndf >> 1, downsample=True)
        self.block5 = DBlock(self.ndf >> 1, self.ndf, downsample=True)
        self.block6 = DBlock(self.ndf, self.ndf, downsample=False)
        self.l7 = SNLinear(self.ndf, 1)
        self.activation = nn.ReLU(True)

        # Produce label vector from trained embedding
        self.l_y = SNLinear(self.ndf, self.num_classes)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l7.weight.data, 1.0)
        nn.init.xavier_uniform_(self.l_y.weight.data, 1.0)

        self.activation = nn.ReLU(True)

    def forward(self, x):
        """
        Feedforwards a batch of real/fake images and produces a batch of GAN logits,
        and rotation classes.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
            Tensor: A batch of predicted classes of shape (N, num_classes).
        """
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)

        # Global sum pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)

        # Produce the class output logits
        output_classes = self.l_y(h)

        return output, output_classes
