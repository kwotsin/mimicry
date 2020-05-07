import os
import shutil

import tensorflow as tf
import torch

from torch_mimicry.metrics import compute_kid
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


class TestComputeKID:
    def setup(self):
        self.netG = ExampleGen()
        self.num_subsets = 10
        self.subset_size = 5
        self.num_samples = self.subset_size * self.num_subsets
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

    def test_compute_gen_dist_feat(self):
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

            fake_feat = compute_kid.compute_gen_dist_feat(
                netG=self.netG,
                num_samples=self.num_samples,
                sess=sess,
                seed=0,
                device=self.device,
                batch_size=10,
                print_every=1)

            assert fake_feat.shape == (self.num_samples, 2048)

    def test_compute_real_dist_feat(self):
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

            real_feat = compute_kid.compute_real_dist_feat(
                num_samples=self.num_samples,
                sess=sess,
                dataset_name='fake_data',
                batch_size=10,
                log_dir=self.log_dir)

            print(real_feat.shape)

            assert real_feat.shape == (self.num_samples, 2048)

    def test_kid_score(self):
        score, var = compute_kid.kid_score(num_subsets=self.num_subsets,
                                           subset_size=self.subset_size,
                                           netG=self.netG,
                                           device=self.device,
                                           dataset_name='fake_data',
                                           batch_size=10,
                                           log_dir=self.log_dir,
                                           seed=0)

        assert type(score) == float
        assert type(var) == float

    def teardown(self):
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        del self.netG


if __name__ == "__main__":
    test = TestComputeKID()
    test.setup()
    test.test_compute_gen_dist_feat()
    test.test_compute_real_dist_feat()
    test.test_kid_score()
    test.teardown()
