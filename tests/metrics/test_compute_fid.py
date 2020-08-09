import os
import shutil
import pytest

import numpy as np
import tensorflow as tf
import torch

from torch_mimicry.metrics import compute_fid
from torch_mimicry.metrics.inception_model import inception_utils
from torch_mimicry.nets.gan import gan


class ExampleGen(gan.BaseGenerator):
    def __init__(self,
                 bottom_width=4,
                 nz=4,
                 ngf=256,
                 loss_type='gan',
                 *args,
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         *args,
                         **kwargs)

    def forward(self, x):
        output = torch.ones(x.shape[0], 3, 32, 32)

        return output


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, nchw=True):
        super().__init__()
        if nchw:
            self.data = torch.ones(30, 3, 32, 32)
        else:
            self.data = torch.ones(30, 32, 32, 3)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


class TestComputeFID:
    def setup(self):
        self.netG = ExampleGen()
        self.num_real_samples = 10
        self.num_fake_samples = 10
        self.batch_size = 10
        self.device = torch.device("cpu")

        # Create inception graph once.
        self.inception_path = './metrics/inception_model'
        if not os.path.exists(self.inception_path):
            os.makedirs(self.inception_path)

        inception_utils.create_inception_graph(self.inception_path)

        # Directory
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "test_log")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def _create_cached_file(self):
        m, s = np.ones((128, )), np.ones((128, 128))
        cached_file = os.path.join(self.log_dir, 'cached.npz')
        np.savez(cached_file, mu=m, sigma=s)

    def test_compute_gen_dist_stats(self):
        if self.device.index is not None:
            # Avoid unbounded memory usage
            gpu_options = tf.compat.v1.GPUOptions(
                allow_growth=True,
                per_process_gpu_memory_fraction=0.15,
                visible_device_list=str(self.device.index))
            config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

        else:
            config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})

        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            m_fake, s_fake = compute_fid.compute_gen_dist_stats(
                netG=self.netG,
                num_samples=self.num_fake_samples,
                sess=sess,
                device=self.device,
                seed=0,
                batch_size=self.batch_size,
                print_every=1)

            assert m_fake.shape == (2048, )
            assert s_fake.shape == (2048, 2048)

    def test_compute_real_dist_stats(self):
        if self.device.index is not None:
            # Avoid unbounded memory usage
            gpu_options = tf.GPUOptions(allow_growth=True,
                                        per_process_gpu_memory_fraction=0.15,
                                        visible_device_list=str(
                                            self.device.index))
            config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

        else:
            config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})

        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            m_real, s_real = compute_fid.compute_real_dist_stats(
                num_samples=self.num_real_samples,
                sess=sess,
                dataset='fake_data',
                batch_size=self.batch_size,
                stats_file=None,
                log_dir=self.log_dir,
                seed=0,
                verbose=True)

            assert m_real.shape == (2048, )
            assert s_real.shape == (2048, 2048)

            # Run from cached
            cached_file = os.path.join(self.log_dir, 'cached.npz')
            self._create_cached_file()
            m_real, s_real = compute_fid.compute_real_dist_stats(
                num_samples=self.num_real_samples,
                sess=sess,
                dataset='fake_data',
                batch_size=self.batch_size,
                stats_file=cached_file,
                log_dir=self.log_dir,
                seed=0,
                verbose=True)

            assert m_real.shape == (128, )
            assert s_real.shape == (128, 128)

    def test_fid_score(self):
        custom_dataset = CustomDataset()

        # Non default dataset
        with pytest.raises(ValueError):
            compute_fid.fid_score(num_real_samples=self.num_real_samples,
                                  num_fake_samples=self.num_fake_samples,
                                  netG=self.netG,
                                  device=None,
                                  seed=99,
                                  batch_size=self.batch_size,
                                  dataset='does_not_exist',
                                  log_dir=self.log_dir)

        # Custom dataset without stats file
        with pytest.raises(ValueError):
            compute_fid.fid_score(num_real_samples=self.num_real_samples,
                                  num_fake_samples=self.num_fake_samples,
                                  netG=self.netG,
                                  device=None,
                                  seed=99,
                                  batch_size=self.batch_size,
                                  dataset=custom_dataset,
                                  log_dir=self.log_dir)

        # Invalid dataset
        with pytest.raises(ValueError):
            compute_fid.fid_score(num_real_samples=self.num_real_samples,
                                  num_fake_samples=self.num_fake_samples,
                                  netG=self.netG,
                                  device=None,
                                  seed=99,
                                  batch_size=self.batch_size,
                                  dataset=None,
                                  log_dir=self.log_dir)

        # Test output
        score = compute_fid.fid_score(num_real_samples=self.num_real_samples,
                                      num_fake_samples=self.num_fake_samples,
                                      netG=self.netG,
                                      device=None,
                                      seed=99,
                                      batch_size=self.batch_size,
                                      dataset='fake_data',
                                      log_dir=self.log_dir)
        assert type(score) == float

        custom_score = compute_fid.fid_score(
            num_real_samples=self.num_real_samples,
            num_fake_samples=self.num_fake_samples,
            netG=self.netG,
            device=None,
            seed=99,
            batch_size=self.batch_size,
            dataset=custom_dataset,
            stats_file=os.path.join(self.log_dir, 'stats_file'),
            log_dir=self.log_dir)
        assert type(custom_score) == float

    def teardown(self):
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        del self.netG


if __name__ == "__main__":
    test = TestComputeFID()
    test.setup()
    test.test_compute_gen_dist_stats()
    test.test_compute_real_dist_stats()
    test.test_fid_score()
    test.teardown()
