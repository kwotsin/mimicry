import os
import shutil

import numpy as np
import pytest
import torch
from skimage.io import imsave

from torch_mimicry.datasets import image_loader


class TestImageLoader:
    def setup(self):
        self.dataset_dir = './datasets'
        self.test_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_dir")

        if not os.path.exists(os.path.join(self.test_dir)):
            os.makedirs(self.test_dir)

    def create_test_image(self, H, W, C):
        img = np.random.randn(H, W, C)
        return img

    def create_imagenet_images(self):
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
            image_loader.get_lsun_bedroom_images(num_samples=1,
                                                 root=os.path.join(
                                                     self.test_dir,
                                                     'does_not_exist'))

    def test_imagenet_small_sample_error(self):
        with pytest.raises(ValueError):
            image_loader.get_imagenet_images(1)

    def test_get_imagenet_images(self):
        self.create_imagenet_images()

        num_images = 1000
        sizes = [32, 128]

        for size in sizes:
            images = image_loader.get_imagenet_images(num_images,
                                                      size=size,
                                                      root=self.test_dir)

            assert images.shape == (num_images, size, size, 3)
            assert np.mean(images[0]) != np.mean(images[1])

    def test_get_fake_data_images(self):
        num_images = 10
        H, W, C = 32, 32, 3
        images = image_loader.get_fake_data_images(num_images,
                                                   root=self.test_dir)

        assert images.shape == (num_images, H, W, C)
        assert np.mean(images[0]) != np.mean(images[1])

    @pytest.mark.skipif(not os.path.exists('./datasets/stl10'),
                        reason='Requires download.')
    def test_get_stl10_images(self):
        num_images = 10
        H, W, C = 48, 48, 3
        images = image_loader.get_stl10_images(num_images,
                                               size=H,
                                               root=self.dataset_dir)

        assert images.shape == (num_images, H, W, C)
        assert np.mean(images[0]) != np.mean(images[1])

    @pytest.mark.skipif(not os.path.exists('./datasets/cifar10'),
                        reason='Requires download.')
    def test_get_cifar10_images(self):
        num_images = 10
        H, W, C = 32, 32, 3
        images = image_loader.get_cifar10_images(num_images,
                                                 root=self.dataset_dir)

        assert images.shape == (num_images, H, W, C)
        assert np.mean(images[0]) != np.mean(images[1])

    @pytest.mark.skipif(not os.path.exists('./datasets/cifar100'),
                        reason='Requires download.')
    def test_get_cifar100_images(self):
        num_images = 10
        H, W, C = 32, 32, 3
        images = image_loader.get_cifar100_images(num_images,
                                                  root=self.dataset_dir)

        assert images.shape == (num_images, H, W, C)
        assert np.mean(images[0]) != np.mean(images[1])

    @pytest.mark.skipif(
        not os.path.exists('./datasets/lsun/bedroom_train_lmdb'),
        reason='Requires download.')
    def test_get_lsun_bedroom_images(self):
        num_images = 10
        H, W, C = 128, 128, 3
        images = image_loader.get_lsun_bedroom_images(num_images,
                                                      size=H,
                                                      root=self.dataset_dir)

        assert images.shape == (num_images, H, W, C)
        assert np.mean(images[0]) != np.mean(images[1])

    @pytest.mark.skipif(not os.path.exists('./datasets/celeba/celeba'),
                        reason='Requires download.')
    def test_get_celeba_images(self):
        num_images = 10
        sizes = [64, 128]

        for size in sizes:
            images = image_loader.get_celeba_images(num_images,
                                                    size=size,
                                                    root=self.dataset_dir,
                                                    download=False)

            assert images.shape == (num_images, size, size, 3)
            assert np.mean(images[0]) != np.mean(images[1])

    def teardown(self):
        shutil.rmtree(self.test_dir)


if __name__ == "__main__":
    test = TestImageLoader()
    test.setup()
    test.test_get_lsun_bedroom_images()
    test.test_get_celeba_images()
    test.test_get_imagenet_images()
    test.test_get_stl10_images()
    test.test_get_cifar10_images()
    test.test_get_cifar100_images()
    test.test_get_fake_data_images()
    test.test_imagenet_small_sample_error()
    test.test_lsun_bedroom_load_error()
    test.teardown()
