import torch

from torch_mimicry.modules import losses


class TestLosses:
    def setup(self):
        self.output_real = torch.ones(4, 1)
        self.output_fake = torch.ones(4, 1)
        self.device = torch.device('cpu')

    def test_minimax_loss(self):
        loss_gen = losses.minimax_loss_gen(output_fake=self.output_fake)

        loss_dis = losses.minimax_loss_dis(output_fake=self.output_fake,
                                           output_real=self.output_real)

        assert loss_gen.dtype == torch.float32
        assert loss_dis.dtype == torch.float32
        assert loss_gen.item() - 0.3133 < 1e-2
        assert loss_dis.item() - 1.6265 < 1e-2

    def test_ns_loss(self):
        loss_gen = losses.ns_loss_gen(self.output_fake)

        assert loss_gen.dtype == torch.float32
        assert loss_gen.item() - 0.3133 < 1e-2

    def test_wasserstein_loss(self):
        loss_gen = losses.wasserstein_loss_gen(self.output_fake)
        loss_dis = losses.wasserstein_loss_dis(output_real=self.output_real,
                                               output_fake=self.output_fake)

        assert loss_gen.dtype == torch.float32
        assert loss_dis.dtype == torch.float32
        assert loss_gen.item() + 1.0 < 1e-2
        assert loss_dis.item() < 1e-2

    def test_hinge_loss(self):
        loss_gen = losses.hinge_loss_gen(output_fake=self.output_fake)
        loss_dis = losses.hinge_loss_dis(output_fake=self.output_fake,
                                         output_real=self.output_real)

        assert loss_gen.dtype == torch.float32
        assert loss_dis.dtype == torch.float32
        assert loss_gen.item() + 1.0 < 1e-2
        assert loss_dis.item() - 2.0 < 1e-2

    def teardown(self):
        del self.output_real
        del self.output_fake
        del self.device


if __name__ == "__main__":
    test = TestLosses()
    test.setup()
    test.test_minimax_loss()
    test.test_ns_loss()
    test.test_wasserstein_loss()
    test.test_hinge_loss()
    test.teardown()
