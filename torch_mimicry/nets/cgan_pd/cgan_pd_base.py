"""
Base class definition of cGAN-PD.
"""

from torch_mimicry.nets.gan import cgan


class CGANPDBaseGenerator(cgan.BaseConditionalGenerator):
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
                 bottom_width,
                 nz,
                 ngf,
                 loss_type='hinge',
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         num_classes=num_classes,
                         **kwargs)


class CGANPDBaseDiscriminator(cgan.BaseConditionalDiscriminator):
    r"""
    ResNet backbone discriminator for cGAN-PD.

    Attributes:
        num_classes (int): Number of classes, more than 0 for conditional GANs.        
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.                
    """
    def __init__(self, num_classes, ndf, loss_type='hinge', **kwargs):
        super().__init__(ndf=ndf,
                         loss_type=loss_type,
                         num_classes=num_classes,
                         **kwargs)
