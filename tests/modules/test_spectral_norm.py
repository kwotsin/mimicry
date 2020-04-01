import torch
import torch.nn as nn

from torch_mimicry.modules import spectral_norm


class TestSpectralNorm:
    def setup(self):
        torch.manual_seed(0)
        self.N, self.C, self.H, self.W = (32, 16, 32, 32)
        self.n_in = self.C
        self.n_out = 32

    def test_SNConv2d(self):
        conv = spectral_norm.SNConv2d(self.n_in, self.n_out, 1, 1, 0)
        conv_def = nn.utils.spectral_norm(
            nn.Conv2d(self.n_in, self.n_out, 1, 1, 0))

        # Init with ones to test implementation without randomness.
        nn.init.ones_(conv.weight.data)
        nn.init.ones_(conv_def.weight.data)

        # Get outputs
        X = torch.ones(self.N, self.C, self.H, self.W)
        output = conv(X)
        output_def = conv_def(X)

        # Test valid shape
        assert output.shape == output_def.shape == (32, 32, 32, 32)

        # Test per element it is very close to default implementation
        # to preserve correctness even when user toggles b/w implementations
        assert abs(torch.mean(output_def) - torch.mean(output)) < 1

    def test_SNLinear(self):
        linear = spectral_norm.SNLinear(self.n_in, self.n_out)
        linear_def = nn.utils.spectral_norm(nn.Linear(self.n_in, self.n_out))

        nn.init.ones_(linear.weight.data)
        nn.init.ones_(linear_def.weight.data)

        X = torch.ones(self.N, self.n_in)
        output = linear(X)
        output_def = linear_def(X)

        assert output.shape == output_def.shape == (32, 32)
        assert abs(torch.mean(output_def) - torch.mean(output)) < 1

    def test_SNEmbedding(self):
        embedding = spectral_norm.SNEmbedding(self.N, self.n_out)
        embedding_def = nn.utils.spectral_norm(nn.Embedding(
            self.N, self.n_out))

        nn.init.ones_(embedding.weight.data)
        nn.init.ones_(embedding_def.weight.data)

        X = torch.ones(self.N, dtype=torch.int64)
        output = embedding(X)
        output_def = embedding_def(X)

        assert output.shape == output_def.shape == (32, 32)
        assert abs(torch.mean(output_def) - torch.mean(output)) < 1


if __name__ == "__main__":
    test = TestSpectralNorm()
    test.setup()
    test.test_SNConv2d()
    test.test_SNLinear()
    test.test_SNEmbedding()
