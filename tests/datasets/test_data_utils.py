import os
import shutil

import numpy as np
import pytest
import torch
from skimage.io import imsave

from torch_mimicry.datasets import data_utils


class TestDataUtils:
    def setup(self):
        self.dataset_dir = './datasets'
        self.test_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_dir")

        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def create_test_image(self, H, W, C):
        img = np.random.randn(H, W, C)
        return img

    def create_imagenet_images(self):
        # Root diectory
        imagenet_dir = os.path.join(self.test_dir, 'imagenet')
        if not os.path.exists(imagenet_dir):
            os.makedirs(imagenet_dir)

        # Meta file
        meta_file = os.path.join(imagenet_dir, 'meta.bin')
        shutil.copy('./tests/datasets/imagenet/test.bin', meta_file)
        _, classes = torch.load(meta_file)
        classes = list(set(classes))

        # Create images
        num_images = 1
        H, W, C = 32, 32, 3

        for i, class_id in enumerate(classes):
            class_dir = os.path.join(imagenet_dir, 'train', class_id)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            for j in range(num_images):
                img_name = os.path.join(class_dir,
                                        '{}_{}.JPEG'.format(class_id, j))

                img = self.create_test_image(H, W, C)
                imsave(img_name, img)

    def test_lsun_bedroom_load_error(self):
        with pytest.raises(ValueError):
            data_utils.load_lsun_bedroom_dataset(
                root=os.path.join(self.test_dir, 'does_not_exist'))

    def test_load_imagenet_dataset(self):
        self.create_imagenet_images()

        sizes = [32, 128]
        for size in sizes:
            for transform_data in [True, False]:
                dataset = data_utils.load_imagenet_dataset(
                    size=size,
                    root=self.test_dir,
                    download=False,
                    transform_data=transform_data)
                img = dataset[0][0]

                if transform_data:
                    assert img.shape == (3, size, size)
                else:
                    img = np.asarray(img)
                    assert img.shape == (32, 32, 3
                                         )  # no resizing done, default 32x32.

    def test_load_fake_dataset(self):
        H, W, C = 32, 32, 3

        for transform_data in [True, False]:
            dataset = data_utils.load_fake_dataset(
                root=self.dataset_dir,
                image_size=(C, H, W),
                transform_data=transform_data)

            img = dataset[0][0]

            if transform_data:
                assert img.shape == (C, H, W)
            else:
                img = np.asarray(img)
                assert img.shape == (32, 32, 3
                                     )  # no resizing done, default 32x32.

    def test_load_wrong_dataset(self):
        with pytest.raises(ValueError):
            data_utils.load_dataset('random', 'wrong_dataset')

    @pytest.mark.skipif(not os.path.exists('./datasets/celeba/celeba'),
                        reason='Requires download.')
    def test_load_celeba_dataset(self):
        sizes = [64, 128]

        for size in sizes:
            dataset = data_utils.load_celeba_dataset(size=size,
                                                     root=self.dataset_dir,
                                                     download=False)

            img = dataset[0][0]

            assert img.shape == (3, size, size)

    @pytest.mark.skipif(not os.path.exists('./datasets/stl10'),
                        reason='Requires download.')
    def test_load_stl10_dataset(self):
        if not os.path.exists('./datasets/stl10'):
            return

        dataset = data_utils.load_stl10_dataset(size=48,
                                                root=self.dataset_dir,
                                                download=True)

        img = dataset[0][0]

        assert img.shape == (3, 48, 48)

    @pytest.mark.skipif(not os.path.exists('./datasets/cifar10'),
                        reason='Requires download.')
    def test_load_cifar10_dataset(self):
        dataset = data_utils.load_cifar10_dataset(root=self.dataset_dir)

        img = dataset[0][0]

        assert img.shape == (3, 32, 32)

    @pytest.mark.skipif(not os.path.exists('./datasets/cifar100'),
                        reason='Requires download.')
    def test_load_cifar100_dataset(self):
        dataset = data_utils.load_cifar100_dataset(root=self.dataset_dir)

        img = dataset[0][0]

        assert img.shape == (3, 32, 32)

    @pytest.mark.skipif(
        not os.path.exists('./datasets/lsun/bedroom_train_lmdb'),
        reason='Requires download.')
    def test_load_lsun_bedroom_dataset(self):
        dataset = data_utils.load_lsun_bedroom_dataset(root=self.dataset_dir,
                                                       size=128)

        img = dataset[0][0]

        assert img.shape == (3, 128, 128)

    def teardown(self):
        shutil.rmtree(self.test_dir)


if __name__ == "__main__":
    test = TestDataUtils()
    test.setup()
    test.test_load_lsun_bedroom_dataset()
    test.test_load_celeba_dataset()
    test.test_load_imagenet_dataset()
    test.test_load_stl10_dataset()
    test.test_load_cifar10_dataset()
    test.test_load_cifar100_dataset()
    test.test_load_wrong_dataset()
    test.test_lsun_bedroom_load_error()
    test.teardown()
