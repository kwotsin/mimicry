"""
Script for building specific layers needed by GAN architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_mimicry.modules import spectral_norm


class SelfAttention(nn.Module):
    """
    Self-attention layer based on version used in BigGAN code:
    https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
    """
    def __init__(self, num_feat, spectral_norm=True):
        super().__init__()
        self.num_feat = num_feat

        if spectral_norm:
            self.f = SNConv2d(self.num_feat,
                              self.num_feat >> 3,
                              1,
                              1,
                              padding=0,
                              bias=False)
            self.g = SNConv2d(self.num_feat,
                              self.num_feat >> 3,
                              1,
                              1,
                              padding=0,
                              bias=False)
            self.h = SNConv2d(self.num_feat,
                              self.num_feat >> 1,
                              1,
                              1,
                              padding=0,
                              bias=False)
            self.o = SNConv2d(self.num_feat >> 1,
                              self.num_feat,
                              1,
                              1,
                              padding=0,
                              bias=False)

        else:
            self.f = nn.Conv2d(self.num_feat,
                               self.num_feat >> 3,
                               1,
                               1,
                               padding=0,
                               bias=False)
            self.g = nn.Conv2d(self.num_feat,
                               self.num_feat >> 3,
                               1,
                               1,
                               padding=0,
                               bias=False)
            self.h = nn.Conv2d(self.num_feat,
                               self.num_feat >> 1,
                               1,
                               1,
                               padding=0,
                               bias=False)
            self.o = nn.Conv2d(self.num_feat >> 1,
                               self.num_feat,
                               1,
                               1,
                               padding=0,
                               bias=False)

        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x):
        """
        Feedforward function. Implementation differs from actual SAGAN paper,
        see note from BigGAN:
        https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py#L142
        """
        # 1x1 convs to project input feature map
        f = self.f(x)
        g = F.max_pool2d(self.g(x), [2, 2])
        h = F.max_pool2d(self.h(x), [2, 2])

        # Reshape layers
        f = f.view(-1, self.num_feat >> 3, x.shape[2] * x.shape[3])
        g = g.view(-1, self.num_feat >> 3, x.shape[2] * x.shape[3] >> 2)
        h = h.view(-1, self.num_feat >> 1, x.shape[2] * x.shape[3] >> 2)

        # Compute attention map probabiltiies
        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), -1)

        # Weigh output features by attention map
        o = self.o(
            torch.bmm(h, beta.transpose(1, 2)).view(-1, self.num_feat >> 1,
                                                    x.shape[2], x.shape[3]))

        return self.gamma * o + x


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
