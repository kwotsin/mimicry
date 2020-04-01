"""
Script for building specific layers needed by GAN architecture.
"""
import torch.nn as nn

from torch_mimicry.modules import spectral_norm


def SNConv2d(*args, **kwargs):
    r"""
    Wrapper for applying spectral norm on conv2d layer.
    """
    if kwargs.get('default', True):
        return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))

    else:
        return spectral_norm.SNConv2d(*args, **kwargs)


def SNLinear(*args, **kwargs):
    r"""
    Wrapper for applying spectral norm on linear layer.
    """
    if kwargs.get('default', True):
        return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))

    else:
        return spectral_norm.SNLinear(*args, **kwargs)


def SNEmbedding(*args, **kwargs):
    r"""
    Wrapper for applying spectral norm on embedding layer.
    """
    if kwargs.get('default', True):
        return nn.utils.spectral_norm(nn.Embedding(*args, **kwargs))

    else:
        return spectral_norm.SNEmbedding(*args, **kwargs)


class ConditionalBatchNorm2d(nn.Module):
    r"""
    Conditional Batch Norm as implemented in
    https://github.com/pytorch/pytorch/issues/8985

    Attributes:
        num_features (int): Size of feature map for batch norm.
        num_classes (int): Determines size of embedding layer to condition BN.
    """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:,
                               num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        r"""
        Feedforwards for conditional batch norm.

        Args:
            x (Tensor): Input feature map.
            y (Tensor): Input class labels for embedding.

        Returns:
            Tensor: Output feature map.
        """
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(
            2, 1)  # divide into 2 chunks, split from dim 1.
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1)

        return out
