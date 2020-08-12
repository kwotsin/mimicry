import os
import shutil

import pytest
import torch
import torch.nn as nn

from torch_mimicry.metrics import compute_metrics
from torch_mimicry.nets.gan import gan


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


class TestMetrics:
    def setup(self):
        # Default values
        self.device = torch.device('cpu')
        self.start_seed = 0
        self.evaluate_step = 100000
        self.dataset = 'fake_data'

        # Test directory
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "test_log")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Create example checkpoint
        self.netG = ExampleGen()
        self.netG.save_checkpoint(directory=os.path.join(
            self.log_dir, 'checkpoints', 'netG'),
                                  global_step=self.evaluate_step,
                                  optimizer=None)

    @pytest.mark.skipif(True, reason='Docker OOM.'
                        )  # TODO: Restore when CI plan is good.
    def test_evaluate_fid(self):
        kwargs = {
            'metric': 'fid',
            'log_dir': self.log_dir,
            'netG': self.netG,
            'dataset': self.dataset,
            'num_real_samples': 10,
            'num_fake_samples': 10,
            'evaluate_step': self.evaluate_step,
            'start_seed': self.start_seed,
            'device': self.device
        }

        scores = compute_metrics.evaluate(**kwargs)[self.evaluate_step]
        assert type(scores) == list
        assert all(map(lambda x: type(x) == float, scores))

    @pytest.mark.skipif(True, reason='Docker OOM.'
                        )  # TODO: Restore when CI plan is good.
    def test_evaluate_kid(self):
        kwargs = {
            'metric': 'kid',
            'log_dir': self.log_dir,
            'evaluate_step': self.evaluate_step,
            'num_subsets': 10,
            'num_samples': 100,
            'netG': self.netG,
            'device': self.device,
            'start_seed': self.start_seed,
            'dataset': self.dataset
        }

        scores = compute_metrics.evaluate(**kwargs)[self.evaluate_step]
        assert type(scores) == list
        assert all(map(lambda x: type(x) == float, scores))

    @pytest.mark.skipif(True, reason='Docker OOM.'
                        )  # TODO: Restore when CI plan is good.
    def test_evaluate_is(self):
        kwargs = {
            'metric': 'inception_score',
            'log_dir': self.log_dir,
            'netG': self.netG,
            'num_samples': 10,
            'num_runs': 3,
            'evaluate_step': self.evaluate_step,
            'device': self.device,
            'start_seed': self.start_seed,
        }

        scores = compute_metrics.evaluate(**kwargs)[self.evaluate_step]
        assert type(scores) == list
        assert all(map(lambda x: type(x) == float, scores))

    def test_arguments(self):
        for metric in ['fid', 'kid', 'inception_score']:
            with pytest.raises(ValueError):
                compute_metrics.evaluate(metric=metric,
                                         log_dir=self.log_dir,
                                         netG=self.netG,
                                         dataset=self.dataset,
                                         evaluate_step=self.evaluate_step,
                                         device=self.device)

        # Both evaluate step and evaluate range defined
        with pytest.raises(ValueError):
            compute_metrics.evaluate(metric=metric,
                                     log_dir=self.log_dir,
                                     netG=self.netG,
                                     dataset=self.dataset,
                                     evaluate_range=(1000, 100000, 1000),
                                     evaluate_step=self.evaluate_step,
                                     device=self.device)

        # Faulty evaluate range
        with pytest.raises(ValueError):
            compute_metrics.evaluate(metric=metric,
                                     log_dir=self.log_dir,
                                     netG=self.netG,
                                     dataset=self.dataset,
                                     evaluate_range=(1000),
                                     device=self.device)

        with pytest.raises(ValueError):
            compute_metrics.evaluate(metric=metric,
                                     log_dir=self.log_dir,
                                     netG=self.netG,
                                     dataset=self.dataset,
                                     evaluate_range=('a', 'b', 'c'),
                                     device=self.device)

        with pytest.raises(ValueError):
            compute_metrics.evaluate(metric=metric,
                                     log_dir=self.log_dir,
                                     netG=self.netG,
                                     dataset=self.dataset,
                                     evaluate_range=(100, 100, 100, 100),
                                     device=self.device)

        with pytest.raises(ValueError):
            compute_metrics.evaluate(metric=metric,
                                     log_dir=self.log_dir,
                                     netG=self.netG,
                                     dataset=self.dataset,
                                     evaluate_range=None,
                                     device=self.device)

        # Invalid ckpt dir
        with pytest.raises(ValueError):
            compute_metrics.evaluate(metric=metric,
                                     log_dir='does_not_exist',
                                     netG=self.netG,
                                     dataset=self.dataset,
                                     evaluate_step=self.evaluate_step,
                                     device=self.device)

    def test_wrong_metric(self):
        with pytest.raises(ValueError):
            compute_metrics.evaluate(metric='wrong_metric',
                                     log_dir=self.log_dir,
                                     netG=self.netG,
                                     dataset=self.dataset,
                                     evaluate_step=self.evaluate_step,
                                     device=self.device)

    def teardown(self):
        del self.netG
        shutil.rmtree(self.log_dir)


if __name__ == "__main__":
    test = TestMetrics()
    test.setup()
    test.test_arguments()
    test.test_wrong_metric()
    test.test_evaluate_fid()
    test.test_evaluate_kid()
    test.test_evaluate_is()
    test.teardown()
