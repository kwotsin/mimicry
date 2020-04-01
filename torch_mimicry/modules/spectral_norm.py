"""
Implementation of spectral normalization for GANs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNorm(object):
    r"""
    Spectral Normalization for GANs (Miyato 2018).

    Inheritable class for performing spectral normalization of weights,
    as approximated using power iteration.

    Details: See Algorithm 1 of Appendix A (Miyato 2018).

    Attributes:
        n_dim (int): Number of dimensions.
        num_iters (int): Number of iterations for power iter.
        eps (float): Epsilon for zero division tolerance when normalizing.
    """
    def __init__(self, n_dim, num_iters=1, eps=1e-12):
        self.num_iters = num_iters
        self.eps = eps

        # Register a singular vector for each sigma
        self.register_buffer('sn_u', torch.randn(1, n_dim))
        self.register_buffer('sn_sigma', torch.ones(1))

    @property
    def u(self):
        return getattr(self, 'sn_u')

    @property
    def sigma(self):
        return getattr(self, 'sn_sigma')

    def _power_iteration(self, W, u, num_iters, eps=1e-12):
        with torch.no_grad():
            for _ in range(num_iters):
                v = F.normalize(torch.matmul(u, W), eps=eps)
                u = F.normalize(torch.matmul(v, W.t()), eps=eps)

        # Note: must have gradients, otherwise weights do not get updated!
        sigma = torch.mm(u, torch.mm(W, v.t()))

        return sigma, u, v

    def sn_weights(self):
        r"""
        Spectrally normalize current weights of the layer.
        """
        W = self.weight.view(self.weight.shape[0], -1)

        # Power iteration
        sigma, u, v = self._power_iteration(W=W,
                                            u=self.u,
                                            num_iters=self.num_iters,
                                            eps=self.eps)

        # Update only during training
        if self.training:
            with torch.no_grad():
                self.sigma[:] = sigma
                self.u[:] = u

        return self.weight / sigma


class SNConv2d(nn.Conv2d, SpectralNorm):
    r"""
    Spectrally normalized layer for Conv2d.

    Attributes:
        in_channels (int): Input channel dimension.
        out_channels (int): Output channel dimensions.
    """
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, *args, **kwargs)

        SpectralNorm.__init__(self,
                              n_dim=out_channels,
                              num_iters=kwargs.get('num_iters', 1))

    def forward(self, x):
        return F.conv2d(input=x,
                        weight=self.sn_weights(),
                        bias=self.bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups)


class SNLinear(nn.Linear, SpectralNorm):
    r"""
    Spectrally normalized layer for Linear.

    Attributes:
        in_features (int): Input feature dimensions.
        out_features (int): Output feature dimensions.
    """
    def __init__(self, in_features, out_features, *args, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, *args, **kwargs)

        SpectralNorm.__init__(self,
                              n_dim=out_features,
                              num_iters=kwargs.get('num_iters', 1))

    def forward(self, x):
        return F.linear(input=x, weight=self.sn_weights(), bias=self.bias)


class SNEmbedding(nn.Embedding, SpectralNorm):
    r"""
    Spectrally normalized layer for Embedding.

    Attributes:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimensions of each embedding vector
    """
    def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, *args,
                              **kwargs)

        SpectralNorm.__init__(self, n_dim=num_embeddings)

    def forward(self, x):
        return F.embedding(input=x, weight=self.sn_weights())
