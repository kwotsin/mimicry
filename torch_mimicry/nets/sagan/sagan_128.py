"""
Implementation of SAGAN for image size 128.
"""

import torch
import torch.nn as nn

from torch_mimicry.nets.sagan import sagan_base
from torch_mimicry.modules.layers import SNLinear, SNConv2d, SNEmbedding, SelfAttention
from torch_mimicry.modules.resblocks import DBlockOptimized, DBlock, GBlock


class SAGANGenerator128(sagan_base.SAGANBaseGenerator):
    r"""
    ResNet backbone generator for SAGAN,

    Attributes:
        num_classes (int): Number of classes, more than 0 for conditional GANs.    
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self,
                 num_classes,
                 nz=128,
                 ngf=1024,
                 bottom_width=4,
                 **kwargs):
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
                             self.ngf >> 1,
                             upsample=True,
                             num_classes=self.num_classes,
                             spectral_norm=True)
        self.block4 = GBlock(self.ngf >> 1,
                             self.ngf >> 2,
                             upsample=True,
                             num_classes=self.num_classes,
                             spectral_norm=True)
        self.block5 = GBlock(self.ngf >> 2,
                             self.ngf >> 3,
                             upsample=True,
                             num_classes=self.num_classes,
                             spectral_norm=True)
        self.block6 = GBlock(self.ngf >> 3,
                             self.ngf >> 4,
                             upsample=True,
                             num_classes=self.num_classes,
                             spectral_norm=True)
        self.b7 = nn.BatchNorm2d(self.ngf >> 4)
        self.c7 = SNConv2d(self.ngf >> 4, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # SA block
        self.attn_block = SelfAttention(self.ngf >> 2, spectral_norm=True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c7.weight.data, 1.0)

    def forward(self, x, y=None):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images, also
        conditioning the batch norm with labels of the images to be produced.

        Self attention is applied after 3rd residual block at G.
        https://github.com/brain-research/self-attention-gan/blob/master/generator.py#L208
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
        h = self.block4(h, y)
        h = self.attn_block(h)
        h = self.block5(h, y)
        h = self.block6(h, y)
        h = self.b7(h)
        h = self.activation(h)
        h = torch.tanh(self.c7(h))

        return h


class SAGANDiscriminator128(sagan_base.SAGANBaseDiscriminator):
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
        self.block1 = DBlockOptimized(3, self.ndf >> 4)
        self.block2 = DBlock(self.ndf >> 4, self.ndf >> 3, downsample=True)
        self.block3 = DBlock(self.ndf >> 3, self.ndf >> 2, downsample=True)
        self.block4 = DBlock(self.ndf >> 2, self.ndf >> 1, downsample=True)
        self.block5 = DBlock(self.ndf >> 1, self.ndf, downsample=True)
        self.block6 = DBlock(self.ndf, self.ndf, downsample=False)
        self.l7 = SNLinear(self.ndf, 1)
        self.activation = nn.ReLU(True)

        # Produce label vector from trained embedding
        self.l_y = SNEmbedding(num_embeddings=self.num_classes,
                               embedding_dim=self.ndf)

        # SA block
        self.attn_block = SelfAttention(self.ndf >> 3, spectral_norm=True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l7.weight.data, 1.0)
        nn.init.xavier_uniform_(self.l_y.weight.data, 1.0)

        self.activation = nn.ReLU(True)

    def forward(self, x, y=None):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.
        Further projects labels to condition on the output logit score.

        Self-attention is applied after 2nd resblock in D:
        https://github.com/brain-research/self-attention-gan/blob/master/discriminator.py#L191

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).
            y (Tensor): A batch of labels of shape (N,).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.attn_block(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)

        # Global sum pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)

        # Add the projection loss
        w_y = self.l_y(y)
        output += torch.sum((w_y * h), dim=1, keepdim=True)

        return output
