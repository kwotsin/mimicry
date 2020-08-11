"""
ResBlocks for WGAN-GP.
"""
import torch.nn as nn
import torch.functional as F

from torch_mimicry.modules import resblocks


class GBlock(resblocks.GBlock):
    r"""
    Residual block for generator. 
    Modifies original resblock definitions with small changes.

    Uses bilinear (rather than nearest) interpolation, and align_corners
    set to False. This is as per how torchvision does upsampling, as seen in:
    https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        upsample (bool): If True, upsamples the input feature map.
        num_classes (int): If more than 0, uses conditional batch norm instead.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 upsample=False,
                 num_classes=0,
                 spectral_norm=False,
                 **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         hidden_channels=hidden_channels,
                         upsample=upsample,
                         num_classes=num_classes,
                         spectral_norm=spectral_norm,
                         **kwargs)

        # Redefine shortcut layer without act.
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(self.in_channels,
                                  self.out_channels,
                                  1,
                                  1,
                                  padding=0)


class DBlock(resblocks.DBlock):
    r"""
    Residual block for discriminator.

    Modifies original resblock definition by including layer norm and removing
    act for shortcut. Convs are LN-ReLU-Conv. See official TF code:
    https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar_resnet.py#L105

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        downsample (bool): If True, downsamples the input feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.        
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 downsample=False,
                 spectral_norm=False,
                 **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         hidden_channels=hidden_channels,
                         downsample=downsample,
                         spectral_norm=spectral_norm,
                         **kwargs)

        # Redefine shortcut layer without act.
        # TODO: Maybe can encapsulate defining of learnable sc in a fn
        # then override it later? Might be cleaner.
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)

        self.norm1 = None
        self.norm2 = None

    # TODO: Verify again. Interestingly, LN has no effect on FID. Not using LN
    # has almost no difference in FID score.
    # def residual(self, x):
    #     r"""
    #     Helper function for feedforwarding through main layers.
    #     """
    #     if self.norm1 is None:
    #         self.norm1 = nn.LayerNorm(
    #             [self.in_channels, x.shape[2], x.shape[3]])

    #     h = x
    #     h = self.norm1(h)
    #     h = self.activation(h)
    #     h = self.c1(h)

    #     if self.norm2 is None:
    #         self.norm2 = nn.LayerNorm(
    #             [self.hidden_channels, h.shape[2], h.shape[3]])

    #     h = self.norm2(h)
    #     h = self.activation(h)
    #     h = self.c2(h)
    #     if self.downsample:
    #         h = F.avg_pool2d(h, 2)

    #     return h


class DBlockOptimized(resblocks.DBlockOptimized):
    r"""
    Optimized residual block for discriminator.

    Does not have any normalisation. See official TF Code:
    https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar_resnet.py#L139

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.        
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 spectral_norm=False,
                 **kwargs):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         spectral_norm=spectral_norm,
                         **kwargs)

        # Redefine shortcut layer
        self.c_sc = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
