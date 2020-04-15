"""
Implementation of cGAN-PD for image size 128.
"""

import torch
import torch.nn as nn

from torch_mimicry.nets.cgan_pd import cgan_pd_base
from torch_mimicry.modules.layers import SNLinear, SNEmbedding
from torch_mimicry.modules.resblocks import DBlockOptimized, DBlock, GBlock


class CGANPDGenerator128(cgan_pd_base.CGANPDBaseGenerator):
    r"""
    ResNet backbone generator for cGAN-PD,

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
        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf,
                             self.ngf,
                             upsample=True,
                             num_classes=self.num_classes)
        self.block3 = GBlock(self.ngf,
                             self.ngf >> 1,
                             upsample=True,
                             num_classes=self.num_classes)
        self.block4 = GBlock(self.ngf >> 1,
                             self.ngf >> 2,
                             upsample=True,
                             num_classes=self.num_classes)
        self.block5 = GBlock(self.ngf >> 2,
                             self.ngf >> 3,
                             upsample=True,
                             num_classes=self.num_classes)
        self.block6 = GBlock(self.ngf >> 3,
                             self.ngf >> 4,
                             upsample=True,
                             num_classes=self.num_classes)
        self.b7 = nn.BatchNorm2d(self.ngf >> 4)
        self.c7 = nn.Conv2d(self.ngf >> 4, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c7.weight.data, 1.0)

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
        h = self.block4(h, y)
        h = self.block5(h, y)
        h = self.block6(h, y)
        h = self.b7(h)
        h = self.activation(h)
        h = torch.tanh(self.c7(h))

        return h


class CGANPDDiscriminator128(cgan_pd_base.CGANPDBaseDiscriminator):
    r"""
    ResNet backbone discriminator for cGAN-PD.

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

        # Initialise the weights
        nn.init.xavier_uniform_(self.l7.weight.data, 1.0)
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
            Tensor: A batch of GAN logits of shape (N, 1).
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

        # Add the projection loss
        w_y = self.l_y(y)
        output += torch.sum((w_y * h), dim=1, keepdim=True)

        return output
