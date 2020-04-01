"""
Implementation of InfoMax-GAN for image size 32.
"""
import torch
import torch.nn as nn

from torch_mimicry.nets.infomax_gan import infomax_gan_base
from torch_mimicry.modules.layers import SNConv2d, SNLinear
from torch_mimicry.modules.resblocks import DBlockOptimized, DBlock, GBlock


class InfoMaxGANGenerator32(infomax_gan_base.InfoMaxGANBaseGenerator):
    r"""
    ResNet backbone generator for InfoMax-GAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
        infomax_loss_scale (float): The alpha parameter used for scaling the generator infomax loss.
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
        h = torch.tanh(self.c5(h))

        return h


class InfoMaxGANDiscriminator32(infomax_gan_base.BaseDiscriminator):
    r"""
    ResNet backbone discriminator for InfoMax-GAN.

    Attributes:
        nrkhs (int): The RKHS dimension R to project the local and global features to.
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
        infomax_loss_scale (float): The beta parameter used for scaling the discriminator infomax loss.
    """
    def __init__(self, nrkhs=1024, ndf=128, **kwargs):
        super().__init__(nrkhs=nrkhs, ndf=ndf, **kwargs)

        # Define activation used
        self.activation = nn.ReLU(True)

        # ----------------
        #   GAN Layers
        # ----------------
        self.local_feat_blocks = nn.Sequential(
            DBlockOptimized(3, self.ndf),
            DBlock(self.ndf, self.ndf, downsample=True),
            DBlock(self.ndf, self.ndf, downsample=False))

        self.global_feat_blocks = nn.Sequential(
            DBlock(self.ndf, self.ndf, downsample=False))

        self.linear = SNLinear(self.ndf, 1)
        nn.init.xavier_uniform_(self.linear.weight.data, 1.0)

        # --------------------
        #   InfoMax Layers
        # --------------------
        # Critic network layers for local features
        self.local_nrkhs_a = SNConv2d(self.ndf, self.ndf, 1, 1, 0)
        self.local_nrkhs_b = SNConv2d(self.ndf, self.nrkhs, 1, 1, 0)
        self.local_nrkhs_sc = SNConv2d(self.ndf, self.nrkhs, 1, 1, 0)

        nn.init.xavier_uniform_(self.local_nrkhs_a.weight.data, 1.0)
        nn.init.xavier_uniform_(self.local_nrkhs_b.weight.data, 1.0)
        nn.init.xavier_uniform_(self.local_nrkhs_sc.weight.data, 1.0)

        # Critic network layers for global features
        self.global_nrkhs_a = SNLinear(self.ndf, self.ndf)
        self.global_nrkhs_b = SNLinear(self.ndf, self.nrkhs)
        self.global_nrkhs_sc = SNLinear(self.ndf, self.nrkhs)

        nn.init.xavier_uniform_(self.global_nrkhs_a.weight.data, 1.0)
        nn.init.xavier_uniform_(self.global_nrkhs_b.weight.data, 1.0)
        nn.init.xavier_uniform_(self.global_nrkhs_sc.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits,
        local features of the images, and global features of the images.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
            Tensor: A batch of local features of shape (N, ndf, H>>2, W>>2).
            Tensor: A batch of global features of shape (N, ndf)
        """
        h = x

        # Get features
        local_feat = self.local_feat_blocks(h)  # (N, C, H, W)
        global_feat = self.global_feat_blocks(local_feat)
        global_feat = self.activation(global_feat)
        global_feat = torch.sum(global_feat, dim=(2, 3))

        # GAN task output
        output = self.linear(global_feat)

        return output, local_feat, global_feat
