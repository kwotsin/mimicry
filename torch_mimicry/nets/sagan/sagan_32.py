"""
Implementation of SAGAN for image size 32.

Note: The original SAGAN did not have architecture designs for datasets with
resolutions other than 128x128. Thus, the current placement of the attention
block follows the version for 128x128 but may well not be optimal for this
particular resolution. Feel free to explore if it works for you.
"""

import torch
import torch.nn as nn

from torch_mimicry.nets.sagan import sagan_base
from torch_mimicry.modules.layers import SNLinear, SNConv2d, SNEmbedding, SelfAttention
from torch_mimicry.modules.resblocks import DBlockOptimized, DBlock, GBlock


class SAGANGenerator32(sagan_base.SAGANBaseGenerator):
    r"""
    ResNet backbone generator for SAGAN,

    Attributes:
        num_classes (int): Number of classes, more than 0 for conditional GANs.    
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self, num_classes, bottom_width=4, nz=128, ngf=256, **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         num_classes=num_classes,
                         **kwargs)

        # Build the layers
        self.l1 = SNLinear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf,
                             self.ngf,
                             upsample=True,
                             num_classes=self.num_classes,
                             spectral_norm=True)
        self.block3 = GBlock(self.ngf,
                             self.ngf,
                             upsample=True,
                             num_classes=self.num_classes,
                             spectral_norm=True)
        self.block4 = GBlock(self.ngf,
                             self.ngf,
                             upsample=True,
                             num_classes=self.num_classes,
                             spectral_norm=True)
        self.b5 = nn.BatchNorm2d(self.ngf)
        self.c5 = SNConv2d(self.ngf, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # SA block
        self.attn_block = SelfAttention(self.ngf, spectral_norm=True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def forward(self, x, y=None):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images, also
        conditioning the batch norm with labels of the images to be produced.

        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).
            y (Tensor): A batch of labels of shape (N,) for conditional batch norm.

        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        if y is None:
            y = torch.randint(low=0,
                              high=self.num_classes,
                              size=(x.shape[0], ),
                              device=x.device)

        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.attn_block(h)
        h = self.block4(h, y)
        h = self.b5(h)
        h = self.activation(h)
        h = torch.tanh(self.c5(h))

        return h


class SAGANDiscriminator32(sagan_base.SAGANBaseDiscriminator):
    r"""
    ResNet backbone discriminator for SAGAN.

    Attributes:
        num_classes (int): Number of classes, more than 0 for conditional GANs.        
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.                
    """
    def __init__(self, num_classes, ndf=128, **kwargs):
        super().__init__(ndf=ndf, num_classes=num_classes, **kwargs)

        # Build layers
        self.block1 = DBlockOptimized(3, self.ndf)
        self.block2 = DBlock(self.ndf, self.ndf, downsample=True)
        self.block3 = DBlock(self.ndf, self.ndf, downsample=False)
        self.block4 = DBlock(self.ndf, self.ndf, downsample=False)
        self.l5 = SNLinear(self.ndf, 1)

        # Produce label vector from trained embedding
        self.l_y = SNEmbedding(num_embeddings=self.num_classes,
                               embedding_dim=self.ndf)

        # SA block
        self.attn_block = SelfAttention(self.ndf, spectral_norm=True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)
        nn.init.xavier_uniform_(self.l_y.weight.data, 1.0)

        self.activation = nn.ReLU(True)

    def forward(self, x, y=None):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.
        Further projects labels to condition on the output logit score.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).
            y (Tensor): A batch of labels of shape (N,).

        Returns:
            output (Tensor): A batch of GAN logits of shape (N, 1).
        """
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.attn_block(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)

        # Global sum pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l5(h)

        # Add the projection loss
        w_y = self.l_y(y)
        output += torch.sum((w_y * h), dim=1, keepdim=True)

        return output
