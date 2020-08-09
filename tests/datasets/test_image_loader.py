import os
import shutil

import numpy as np
import pytest
import torch
from skimage.io import imsave

from torch_mimicry.datasets import image_loader, data_utils


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


class TestImageLoader:
    def setup(self):
        self.dataset_dir = './datasets'
        self.test_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "test_dir")

        if not os.path.exists(os.path.join(self.test_dir)):
            os.makedirs(self.test_dir)

        self.custom_dataset = CustomDataset()

    def create_test_image(self, H, W, C, img_name):
        if C == 1 or C == 3:
            img = np.random.randn(H, W, C)
            img_name += '.JPEG'

        elif C == 4:
            img = np.random.randn(H, W, C)
            img_name += '.PNG'

        imsave(img_name, img)

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
        H, W = 32, 32

        for i, class_id in enumerate(classes):
            class_dir = os.path.join(imagenet_dir, 'train', class_id)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            for j in range(num_images):
                img_name = os.path.join(class_dir, '{}_{}'.format(class_id, j))

                if os.path.exists(img_name):
                    continue
                else:
                    # Randomly choose 1, 3 or 4 channels
                    C = int(np.random.choice([1, 3, 4]))
                    self.create_test_image(H, W, C, img_name)

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

    def test_sample_dataset_images(self):
        # Test typical dataset
        test_dataset = data_utils.load_fake_dataset(root=self.test_dir,
                                                    size=30,
                                                    image_size=(3, 32, 32))
        images = image_loader.sample_dataset_images(test_dataset, 10)

        assert images.shape == (10, 3, 32, 32)

        # If indexing dataset returns non iterable
        images = image_loader.sample_dataset_images(self.custom_dataset, 10)

        assert images.shape == (10, 3, 32, 32)

    def test_get_dataset_images(self):
        # Check if can return properly formatted np array.
        datasets = [
            'imagenet_32',
            'imagenet_128',
            'celeba_64',
            'celeba_128',
            'stl10_48',
            'cifar10',
            'cifar100',
            'lsun_bedroom_128',
            'fake_data',
        ]

        for ds in datasets:
            try:
                if 'imagenet' in ds:
                    self.create_imagenet_images()
                    images = image_loader.get_dataset_images(
                        ds, num_samples=1000, root=self.test_dir)

                else:
                    try:
                        images = image_loader.get_dataset_images(
                            ds,
                            num_samples=10,
                            root=self.dataset_dir,
                            download=False)

                    except (TypeError, ValueError):
                        continue  # Download false option not available.

                assert isinstance(images, np.ndarray)
                assert images.shape[
                    3] == 3  # 3 channels for all default datasets.

            except RuntimeError:
                continue  # No dataset download.

        # Check for bad formats
        bad_format_dataset = CustomDataset(nchw=True)
        bad_format_dataset.data *= 256
        images = image_loader.get_dataset_images(bad_format_dataset, 10)

        assert images.shape == (10, 32, 32, 3)
        assert np.min(images) >= 0 and np.max(images) <= 255

        wrong_dataset = None
        with pytest.raises(ValueError):
            image_loader.get_dataset_images(wrong_dataset, 10)

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
    test.test_sample_dataset_images()
    test.test_get_dataset_images()
    test.teardown()
