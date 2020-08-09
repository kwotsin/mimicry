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
        self.spectral_norm = spectral_norm

        if self.spectral_norm:
            self.theta = SNConv2d(self.num_feat,
                                  self.num_feat >> 3,
                                  1,
                                  1,
                                  padding=0,
                                  bias=False)
            self.phi = SNConv2d(self.num_feat,
                                self.num_feat >> 3,
                                1,
                                1,
                                padding=0,
                                bias=False)
            self.g = SNConv2d(self.num_feat,
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
            self.theta = nn.Conv2d(self.num_feat,
                                   self.num_feat >> 3,
                                   1,
                                   1,
                                   padding=0,
                                   bias=False)
            self.phi = nn.Conv2d(self.num_feat,
                                 self.num_feat >> 3,
                                 1,
                                 1,
                                 padding=0,
                                 bias=False)
            self.g = nn.Conv2d(self.num_feat,
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

        See official TF Implementation:
        https://github.com/brain-research/self-attention-gan/blob/master/non_local.py

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Feature map weighed with attention map.
        """
        N, C, H, W = x.shape
        location_num = H * W
        downsampled_num = location_num >> 2

        # Theta path
        theta = self.theta(x)
        theta = theta.view(N, C >> 3, location_num)  # (N, C>>3, H*W)

        # Phi path
        phi = self.phi(x)
        phi = F.max_pool2d(phi, [2, 2], stride=2)
        phi = phi.view(N, C >> 3, downsampled_num)  # (N, C>>3, H*W>>2)

        # Attention map
        attn = torch.bmm(theta.transpose(1, 2), phi)
        attn = F.softmax(attn, -1)  # (N, H*W, H*W>>2)
        # print(torch.sum(attn, axis=2)) # (N, H*W)

        # Conv value
        g = self.g(x)
        g = F.max_pool2d(g, [2, 2], stride=2)
        g = g.view(N, C >> 1, downsampled_num)  # (N, C>>1, H*W>>2)

        # Apply attention
        attn_g = torch.bmm(g, attn.transpose(1, 2))  # (N, C>>1, H*W)
        attn_g = attn_g.view(N, C >> 1, H, W)  # (N, C>>1, H, W)

        # Project back feature size
        attn_g = self.o(attn_g)

        # Weigh attention map
        output = x + self.gamma * attn_g

        return output


def SNConv2d(*args, default=True, **kwargs):
    r"""
    Wrapper for applying spectral norm on conv2d layer.
    """
    if default:
        return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))

    else:
        return spectral_norm.SNConv2d(*args, **kwargs)


def SNLinear(*args, default=True, **kwargs):
    r"""
    Wrapper for applying spectral norm on linear layer.
    """
    if default:
        return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))

    else:
        return spectral_norm.SNLinear(*args, **kwargs)


def SNEmbedding(*args, default=True, **kwargs):
    r"""
    Wrapper for applying spectral norm on embedding layer.
    """
    if default:
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
