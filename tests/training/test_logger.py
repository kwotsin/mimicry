import os
import shutil

import torch
import torch.nn as nn

from torch_mimicry.nets.gan import gan, cgan
from torch_mimicry.training import logger, metric_log


class ExampleGen(gan.BaseGenerator):
    def __init__(self,
                 bottom_width=4,
                 nz=4,
                 ngf=16,
                 loss_type='gan',
                 *args,
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         *args,
                         **kwargs)
        self.linear = nn.Linear(self.nz, 3072)

    def forward(self, x):
        output = self.linear(x)
        output = output.view(x.shape[0], 3, 32, 32)

        return output


class ExampleConditionalGen(cgan.BaseConditionalGenerator):
    def __init__(self,
                 bottom_width=4,
                 nz=4,
                 ngf=16,
                 loss_type='gan',
                 num_classes=10,
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         num_classes=num_classes,
                         **kwargs)
        self.linear = nn.Linear(self.nz, 3072)

    def forward(self, x, y=None):
        output = self.linear(x)
        output = output.view(x.shape[0], 3, 32, 32)

        return output


class TestLogger:
    def setup(self):
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "test_log")

        self.logger = logger.Logger(log_dir=self.log_dir,
                                    num_steps=100,
                                    dataset_size=50000,
                                    flush_secs=30,
                                    device=torch.device('cpu'))

        self.scalars = [
            'errG',
            'errD',
            'D(x)',
            'D(G(z))',
            'img',
            'lr_D',
            'lr_G',
        ]

    def test_print_log(self):
        log_data = metric_log.MetricLog()
        global_step = 10

        # Populate log data with some value
        for scalar in self.scalars:
            if scalar == 'img':
                continue

            log_data.add_metric(scalar, 1.0)

        printed = self.logger.print_log(global_step=global_step,
                                        log_data=log_data,
                                        time_taken=10)

        assert printed == (
            'INFO: [Epoch 1/1][Global Step: 10/100] ' +
            '\n| D(G(z)): 1.0\n| D(x): 1.0\n| errD: 1.0\n| errG: 1.0' +
            '\n| lr_D: 1.0\n| lr_G: 1.0\n| (10.0000 sec/idx)')

    def test_vis_images(self):
        netG = ExampleGen()
        netG_conditional = ExampleConditionalGen()

        global_step = 10
        num_images = 64

        # Test unconditional
        self.logger.vis_images(netG, global_step, num_images)
        img_dir = os.path.join(self.log_dir, 'images')
        filenames = os.listdir(img_dir)
        assert 'fake_samples_step_10.png' in filenames
        assert 'fixed_fake_samples_step_10.png' in filenames

        # Remove images
        for file in filenames:
            os.remove(os.path.join(img_dir, file))

        # Test conditional
        self.logger.vis_images(netG_conditional, global_step, num_images)
        assert 'fake_samples_step_10.png' in filenames
        assert 'fixed_fake_samples_step_10.png' in filenames

    def teardown(self):
        shutil.rmtree(self.log_dir)


if __name__ == "__main__":
    test = TestLogger()
    test.setup()
    test.test_print_log()
    test.test_vis_images()
    test.teardown()
