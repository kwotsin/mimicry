"""
Base class definition of DCGAN.
"""
from torch_mimicry.nets.gan import gan


class DCGANBaseGenerator(gan.BaseGenerator):
    r"""
    ResNet backbone generator for ResNet DCGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self, nz, ngf, bottom_width, loss_type='ns', **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         **kwargs)


class DCGANBaseDiscriminator(gan.BaseDiscriminator):
    r"""
    ResNet backbone discriminator for ResNet DCGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self, ndf, loss_type='ns', **kwargs):
        super().__init__(ndf=ndf, loss_type=loss_type, **kwargs)
