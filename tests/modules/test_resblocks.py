from itertools import product

import torch

from torch_mimicry.modules import resblocks


class TestResBlocks:
    def setup(self):
        self.images = torch.ones(4, 3, 16, 16)

    def test_GBlock(self):
        # Arguments
        num_classes_list = [0, 10]
        spectral_norm_list = [True, False]
        in_channels = 3
        out_channels = 8
        args_comb = product(num_classes_list, spectral_norm_list)

        for args in args_comb:
            num_classes = args[0]
            spectral_norm = args[1]

            if num_classes > 0:
                y = torch.ones((4, ), dtype=torch.int64)
            else:
                y = None

            gen_block_up = resblocks.GBlock(in_channels=in_channels,
                                            out_channels=out_channels,
                                            upsample=True,
                                            num_classes=num_classes,
                                            spectral_norm=spectral_norm)

            gen_block = resblocks.GBlock(in_channels=in_channels,
                                         out_channels=out_channels,
                                         upsample=False,
                                         num_classes=num_classes,
                                         spectral_norm=spectral_norm)

            gen_block_no_sc = resblocks.GBlock(in_channels=in_channels,
                                               out_channels=in_channels,
                                               upsample=False,
                                               num_classes=num_classes,
                                               spectral_norm=spectral_norm)

            assert gen_block_up(self.images, y).shape == (4, 8, 32, 32)
            assert gen_block(self.images, y).shape == (4, 8, 16, 16)
            assert gen_block_no_sc(self.images, y).shape == (4, 3, 16, 16)

    def test_DBlocks(self):
        in_channels = 3
        out_channels = 8

        for spectral_norm in [True, False]:
            dis_block_down = resblocks.DBlock(in_channels=in_channels,
                                              out_channels=out_channels,
                                              downsample=True,
                                              spectral_norm=spectral_norm)

            dis_block = resblocks.DBlock(in_channels=in_channels,
                                         out_channels=out_channels,
                                         downsample=False,
                                         spectral_norm=spectral_norm)

            dis_block_opt = resblocks.DBlockOptimized(
                in_channels=in_channels,
                out_channels=out_channels,
                spectral_norm=spectral_norm)

            assert dis_block(self.images).shape == (4, out_channels, 16, 16)
            assert dis_block_down(self.images).shape == (4, out_channels, 8, 8)
            assert dis_block_opt(self.images).shape == (4, out_channels, 8, 8)

    def teardown(self):
        del self.images


if __name__ == "__main__":
    test = TestResBlocks()
    test.setup()
    test.test_GBlock()
    test.test_DBlocks()
    test.teardown()
