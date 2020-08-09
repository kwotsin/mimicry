"""
Script for loading datasets.
"""
import os

import torchvision
from torchvision import transforms

from torch_mimicry.datasets.imagenet import imagenet


def load_dataset(root, name, **kwargs):
    """
    Loads different datasets specifically for GAN training. 
    By default, all images are normalized to values in the range [-1, 1].

    Args:
        root (str): Path to where datasets are stored.
        name (str): Name of dataset to load.

    Returns:
        Dataset: Torch Dataset object for a specific dataset.
    """
    if name == "cifar10":
        return load_cifar10_dataset(root, **kwargs)

    elif name == "cifar100":
        return load_cifar100_dataset(root, **kwargs)

    elif name == "imagenet_32":
        return load_imagenet_dataset(root, size=32, **kwargs)

    elif name == "imagenet_128":
        return load_imagenet_dataset(root, size=128, **kwargs)

    elif name == "stl10_48":
        return load_stl10_dataset(root, size=48, **kwargs)

    elif name == "celeba_64":
        return load_celeba_dataset(root, size=64, **kwargs)

    elif name == "celeba_128":
        return load_celeba_dataset(root, size=128, **kwargs)

    elif name == "lsun_bedroom_128":
        return load_lsun_bedroom_dataset(root, size=128, **kwargs)

    elif name == "fake_data":
        return load_fake_dataset(root, **kwargs)

    else:
        raise ValueError("Invalid dataset name {} selected.".format(name))


def load_fake_dataset(root,
                      transform_data=True,
                      convert_tensor=True,
                      image_size=(3, 32, 32),
                      **kwargs):
    """
    Loads fake dataset for testing.

    Args:
        root (str): Path to where datasets are stored.
        transform_data (bool): If True, preprocesses data.
        convert_tensor (bool): If True, converts image to tensor and preprocess 
            to range [-1, 1].

    Returns:
        Dataset: Torch Dataset object.
    """
    dataset_dir = os.path.join(root, 'fake_data')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if transform_data:
        transforms_list = []
        if convert_tensor:
            transforms_list += [
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ]

        transform = transforms.Compose(transforms_list)

    else:
        transform = None

    dataset = torchvision.datasets.FakeData(transform=transform,
                                            image_size=image_size,
                                            **kwargs)

    return dataset


def load_lsun_bedroom_dataset(root,
                              size=128,
                              transform_data=True,
                              convert_tensor=True,
                              **kwargs):
    """
    Loads LSUN-Bedroom dataset.

    Args:
        root (str): Path to where datasets are stored.
        size (int): Size to resize images to.
        transform_data (bool): If True, preprocesses data.
        convert_tensor (bool): If True, converts image to tensor and preprocess 
            to range [-1, 1].

    Returns:
        Dataset: Torch Dataset object.   
    """
    dataset_dir = os.path.join(root, 'lsun')
    if not os.path.exists(dataset_dir):
        raise ValueError(
            "Missing directory {}. Download the dataset to this directory.".
            format(dataset_dir))

    if transform_data:
        transforms_list = [transforms.CenterCrop(256), transforms.Resize(size)]
        if convert_tensor:
            transforms_list += [
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ]

        transform = transforms.Compose(transforms_list)

    else:
        transform = None

    dataset = torchvision.datasets.LSUN(root=dataset_dir,
                                        classes=['bedroom_train'],
                                        transform=transform,
                                        **kwargs)

    return dataset


def load_celeba_dataset(root,
                        transform_data=True,
                        convert_tensor=True,
                        download=True,
                        split='all',
                        size=64,
                        **kwargs):
    """
    Loads the CelebA dataset.

    Args:
        root (str): Path to where datasets are stored.
        size (int): Size to resize images to.
        transform_data (bool): If True, preprocesses data.
        split (str): The split of data to use.
        download (bool): If True, downloads the dataset.
        convert_tensor (bool): If True, converts image to tensor and preprocess 
            to range [-1, 1].

    Returns:
        Dataset: Torch Dataset object.   
    """
    dataset_dir = os.path.join(root, 'celeba')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if transform_data:
        # Build default transforms for scaling outputs to -1 to 1.
        transforms_list = [
            transforms.CenterCrop(
                178),  # Because each image is size (178, 218) spatially.
            transforms.Resize(size)
        ]
        if convert_tensor:
            transforms_list += [
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ]

        transform = transforms.Compose(transforms_list)

    else:
        transform = None

    if download:
        print("INFO: download is True. Downloading CelebA images...")

    dataset = torchvision.datasets.CelebA(root=dataset_dir,
                                          transform=transform,
                                          download=download,
                                          split=split,
                                          **kwargs)

    return dataset


def load_stl10_dataset(root,
                       size=48,
                       split='unlabeled',
                       download=True,
                       transform_data=True,
                       convert_tensor=True,
                       **kwargs):
    """
    Loads the STL10 dataset.

    Args:
        root (str): Path to where datasets are stored.
        size (int): Size to resize images to.
        transform_data (bool): If True, preprocesses data.
        split (str): The split of data to use.
        download (bool): If True, downloads the dataset.
        convert_tensor (bool): If True, converts image to tensor and preprocess 
            to range [-1, 1].

    Returns:
        Dataset: Torch Dataset object.   
    """
    dataset_dir = os.path.join(root, 'stl10')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if transform_data:
        transforms_list = [transforms.Resize(size)]
        if convert_tensor:
            transforms_list += [
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ]

        transform = transforms.Compose(transforms_list)

    else:
        transform = None

    dataset = torchvision.datasets.STL10(root=dataset_dir,
                                         split=split,
                                         transform=transform,
                                         download=download,
                                         **kwargs)

    return dataset


def load_imagenet_dataset(root,
                          size=32,
                          split='train',
                          download=True,
                          transform_data=True,
                          convert_tensor=True,
                          **kwargs):
    """
    Loads the ImageNet dataset.

    Args:
        root (str): Path to where datasets are stored.
        size (int): Size to resize images to.
        transform_data (bool): If True, preprocesses data.
        split (str): The split of data to use.
        download (bool): If True, downloads the dataset.
        convert_tensor (bool): If True, converts image to tensor and preprocess 
            to range [-1, 1].

    Returns:
        Dataset: Torch Dataset object.   
    """
    dataset_dir = os.path.join(root, 'imagenet')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if transform_data:
        transforms_list = [transforms.CenterCrop(224), transforms.Resize(size)]
        if convert_tensor:
            transforms_list += [
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ]

        transform = transforms.Compose(transforms_list)

    else:
        transform = None

    dataset = imagenet.ImageNet(root=dataset_dir,
                                split=split,
                                transform=transform,
                                download=download,
                                **kwargs)

    return dataset


def load_cifar100_dataset(root,
                          split='train',
                          download=True,
                          transform_data=True,
                          convert_tensor=True,
                          **kwargs):
    """
    Loads the CIFAR-100 dataset.

    Args:
        root (str): Path to where datasets are stored.
        transform_data (bool): If True, preprocesses data.
        split (str): The split of data to use.
        download (bool): If True, downloads the dataset.
        convert_tensor (bool): If True, converts image to tensor and preprocess 
            to range [-1, 1].

    Returns:
        Dataset: Torch Dataset object.   
    """
    dataset_dir = os.path.join(root, 'cifar100')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if transform_data:
        transforms_list = []
        if convert_tensor:
            transforms_list += [
                transforms.ToTensor(),
                transforms.Normalize((0.5, ), (0.5, ))
            ]

        transform = transforms.Compose(transforms_list)
    else:
        transform = None

    # Build datasets
    if split == "all":
        train_dataset = torchvision.datasets.CIFAR100(root=dataset_dir,
                                                      train=True,
                                                      transform=transform,
                                                      download=download,
                                                      **kwargs)

        test_dataset = torchvision.datasets.CIFAR100(root=dataset_dir,
                                                     train=False,
                                                     transform=transform,
                                                     download=download,
                                                     **kwargs)

        # Merge the datasets
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    elif split == "train":
        dataset = torchvision.datasets.CIFAR100(root=dataset_dir,
                                                train=True,
                                                transform=transform,
                                                download=download,
                                                **kwargs)

    elif split == "test":
        dataset = torchvision.datasets.CIFAR100(root=dataset_dir,
                                                train=False,
                                                transform=transform,
                                                download=download,
                                                **kwargs)

    else:
        raise ValueError("split argument must one of ['train', 'val', 'all']")

    return dataset


def load_cifar10_dataset(root,
                         split='train',
                         download=True,
                         transform_data=True,
                         **kwargs):
    """
    Loads the CIFAR-10 dataset.
    
    Args:
        root (str): Path to where datasets are stored.
        transform_data (bool): If True, preprocesses data.
        split (str): The split of data to use.
        download (bool): If True, downloads the dataset.
        convert_tensor (bool): If True, converts image to tensor and preprocess 
            to range [-1, 1].

    Returns:
        Dataset: Torch Dataset object.   
    """
    dataset_dir = os.path.join(root, 'cifar10')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if transform_data:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, ), (0.5, ))])
    else:
        transform = None

    # Build datasets
    if split == "all":
        train_dataset = torchvision.datasets.CIFAR10(root=dataset_dir,
                                                     train=True,
                                                     transform=transform,
                                                     download=download,
                                                     **kwargs)

        test_dataset = torchvision.datasets.CIFAR10(root=dataset_dir,
                                                    train=False,
                                                    transform=transform,
                                                    download=download,
                                                    **kwargs)

        # Merge the datasets
        dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    elif split == "train":
        dataset = torchvision.datasets.CIFAR10(root=dataset_dir,
                                               train=True,
                                               transform=transform,
                                               download=download,
                                               **kwargs)

    elif split == "test":
        dataset = torchvision.datasets.CIFAR10(root=dataset_dir,
                                               train=False,
                                               transform=transform,
                                               download=download,
                                               **kwargs)

    else:
        raise ValueError("split argument must one of ['train', 'val', 'all']")

    return dataset
