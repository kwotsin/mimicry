"""
Base implementation of SNGAN with default variables.
"""
from torch_mimicry.nets.gan import gan


class SNGANBaseGenerator(gan.BaseGenerator):
    r"""
    ResNet backbone generator for SNGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz, ngf, bottom_width, loss_type='hinge', **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         **kwargs)


class SNGANBaseDiscriminator(gan.BaseDiscriminator):
    r"""
    ResNet backbone discriminator for SNGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
    """
    def __init__(self, ndf, loss_type='hinge', **kwargs):
        super().__init__(ndf=ndf, loss_type=loss_type, **kwargs)
