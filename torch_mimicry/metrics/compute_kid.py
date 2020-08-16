"""
PyTorch interface for computing KID.
"""
import os
import random
import time

import numpy as np
import tensorflow as tf
import torch

from torch_mimicry.datasets.image_loader import get_dataset_images
from torch_mimicry.metrics.inception_model import inception_utils
from torch_mimicry.metrics.kid import kid_utils


def compute_real_dist_feat(num_samples,
                           sess,
                           dataset,
                           batch_size,
                           seed=0,
                           verbose=True,
                           feat_file=None,
                           log_dir='./log'):
    """
    Reads the image data and compute the real image features.

    Args:
        num_samples (int): Number of real images to compute features.
        sess (Session): TensorFlow session to use.
        dataset (str/Dataset): Dataset to load.
        batch_size (int): The batch size to feedforward for inference.
        feat_file (str): The features file to load from if there is already one.
        verbose (bool): If True, prints progress of computation.
        log_dir (str): Directory where features can be stored.

    Returns:
        ndarray: Inception features of real images.
    """
    # Create custom feat file name
    if feat_file is None:
        feat_dir = os.path.join(log_dir, 'metrics', 'kid', 'features')
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)

        feat_file = os.path.join(
            feat_dir,
            "kid_feat_{}_{}k_run_{}.npz".format(dataset, num_samples // 1000,
                                                seed))

    if feat_file and os.path.exists(feat_file):
        print("INFO: Loading existing features for real images...")
        f = np.load(feat_file)
        real_feat = f['feat'][:]
        f.close()

    else:
        # Obtain the numpy format data
        print("INFO: Obtaining images...")
        images = get_dataset_images(dataset, num_samples=num_samples)

        # Compute the mean and cov
        print("INFO: Computing features for real images...")
        real_feat = inception_utils.get_activations(images=images,
                                                    sess=sess,
                                                    batch_size=batch_size,
                                                    verbose=verbose)

        print("INFO: Saving features for real images...")
        np.savez(feat_file, feat=real_feat)

    return real_feat


def _normalize_images(images):
    """
    Given a tensor of images, uses the torchvision
    normalization method to convert floating point data to integers. See reference
    at: https://pytorch.org/docs/stable/_modules/torchvision/utils.html#save_image

    The function uses the normalization from make_grid and save_image functions.

    Args:
        images (Tensor): Batch of images of shape (N, 3, H, W).

    Returns:
        ndarray: Batch of normalized images of shape (N, H, W, 3).
    """
    # Shift the image from [-1, 1] range to [0, 1] range.
    min_val = float(images.min())
    max_val = float(images.max())
    images.clamp_(min=min_val, max=max_val)
    images.add_(-min_val).div_(max_val - min_val + 1e-5)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    images = images.mul_(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to(
        'cpu', torch.uint8).numpy()

    return images


def compute_gen_dist_feat(netG,
                          num_samples,
                          sess,
                          device,
                          seed,
                          batch_size,
                          print_every=20,
                          verbose=True):
    """
    Directly produces the images and convert them into numpy format without
    saving the images on disk.

    Args:
        netG (Module): Torch Module object representing the generator model.
        num_samples (int): The number of fake images for computing features.
        sess (Session): TensorFlow session to use.
        device (str): Device identifier to use for computation.
        seed (int): The random seed to use.
        batch_size (int): The number of samples per batch for inference.
        print_every (int): Interval for printing log.
        verbose (bool): If True, prints progress.

    Returns:
        ndarray: Inception features of generated images.
    """
    batch_size = min(num_samples, batch_size)

    with torch.no_grad():
        # Set model to evaluation mode
        netG.eval()

        # Collect num_samples of fake images
        images = []

        # Collect all samples
        start_time = time.time()
        for idx in range(num_samples // batch_size):
            fake_images = netG.generate_images(num_images=batch_size,
                                               device=device).detach().cpu()

            # Collect fake image
            images.append(fake_images)

            # Print some statistics
            if (idx + 1) % print_every == 0:
                end_time = time.time()
                print(
                    "INFO: Generated image {}/{} [Random Seed {}] ({:.4f} sec/idx)"
                    .format(
                        (idx + 1) * batch_size, num_samples, seed,
                        (end_time - start_time) / (print_every * batch_size)))
                start_time = end_time

        # Produce images in the required (N, H, W, 3) format for kid computation
        images = torch.cat(images, 0)  # Gives (N, 3, H, W)
        images = _normalize_images(images)  # Gives (N, H, W, 3)

    # Compute the kid
    print("INFO: Computing features for fake images...")
    fake_feat = inception_utils.get_activations(images=images,
                                                sess=sess,
                                                batch_size=batch_size,
                                                verbose=verbose)

    return fake_feat


def kid_score(num_samples,
              netG,
              dataset,
              seed=0,
              device=None,
              num_subsets=10,
              batch_size=50,
              verbose=True,
              feat_file=None,
              log_dir='./log'):
    """
    Computes KID score.

    Args:
        num_samples (int): The number of real and fake images to use for KID.
        num_subsets (int): Number of subsets to compute average MMD.
        netG (Module): Torch Module object representing the generator model.
        device (str): Device identifier to use for computation.
        seed (int): The random seed to use.
        dataset (str/Dataset): The name of the dataset to load if known, or a custom Dataset object
        batch_size (int): The batch size to feedforward for inference.
        feat_file (str): The path to specific inception features for real images.
        log_dir (str): Directory where features can be stored.
        verbose (bool): If True, prints progress.

    Returns:
        tuple: Scalar mean and std of KID scores computed.
    """
    start_time = time.time()

    # Check inputs
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    if isinstance(dataset, str):
        default_datasets = {
            'cifar10',
            'cifar100',
            'stl10_48',
            'imagenet_32',
            'imagenet_128',
            'celeba_64',
            'celeba_128',
            'lsun_bedroom',
            'fake_data',
        }
        if dataset not in default_datasets:
            raise ValueError('For default datasets, must be one of {}'.format(
                default_datasets))

    elif issubclass(type(dataset), torch.utils.data.Dataset):
        if feat_file is None:
            raise ValueError(
                "feat_file to save/load from cannot be empty if using a custom dataset.")

        if not feat_file.endswith('.npz'):
            feat_file = feat_file + '.npz'

    else:
        raise ValueError(
            'dataset must be either a Dataset object or a string.')

    # Make sure the random seeds are fixed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Directories
    inception_path = os.path.join(log_dir, 'metrics', 'inception_model')

    # Setup the inception graph
    inception_utils.create_inception_graph(inception_path)

    # Start producing features for real and fake images
    # if device.index is not None:
    #     # Avoid unbounded memory usage
    #     gpu_options = tf.compat.v1.GPUOptions(allow_growth=True,
    #                                 per_process_gpu_memory_fraction=0.15,
    #                                 visible_device_list=str(device.index))
    #     config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

    # else:
    #     config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        real_feat = compute_real_dist_feat(num_samples=num_samples,
                                           sess=sess,
                                           dataset=dataset,
                                           batch_size=batch_size,
                                           verbose=verbose,
                                           feat_file=feat_file,
                                           log_dir=log_dir,
                                           seed=seed)

        fake_feat = compute_gen_dist_feat(netG=netG,
                                          num_samples=num_samples,
                                          sess=sess,
                                          device=device,
                                          seed=seed,
                                          batch_size=batch_size,
                                          verbose=verbose)

        # Compute the KID score
        subset_size = num_samples // num_subsets
        scores = kid_utils.polynomial_mmd_averages(real_feat,
                                                   fake_feat,
                                                   n_subsets=num_subsets,
                                                   subset_size=subset_size)

        mmd_score, mmd_std = float(np.mean(scores)), float(np.std(scores))

        print("INFO: KID: {:.4f} ± {:.4f} [Time Taken: {:.4f} secs]".format(
            mmd_score, mmd_std,
            time.time() - start_time))

        return mmd_score, mmd_std


# def kid_score(num_subsets,
#               netG,
#               device,
#               seed,
#               dataset,
#               batch_size=50,
#               verbose=True,
#               feat_file=None,
#               log_dir='./log'):
#     """
#     Computes KID score.

#     Args:
#         num_real_samples (int): The number of real images to use for FID.
#         num_fake_samples (int): The number of fake images to use for FID.
#         num_subsets (int): Number of subsets to compute average MMD.
#         netG (Module): Torch Module object representing the generator model.
#         device (str): Device identifier to use for computation.
#         seed (int): The random seed to use.
#         dataset (str): The name of the dataset to load.
#         batch_size (int): The batch size to feedforward for inference.
#         feat_file (str): The path to specific inception features for real images.
#         log_dir (str): Directory where features can be stored.
#         verbose (bool): If True, prints progress.

#     Returns:
#         tuple: Scalar mean and std of KID scores computed.
#     """
#     start_time = time.time()

#     # Make sure the random seeds are fixed
#     torch.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)

#     # Directories
#     inception_path = os.path.join(log_dir, 'metrics', 'inception_model')

#     # Setup the inception graph
#     inception_utils.create_inception_graph(inception_path)

#     # Decide sample size
#     num_samples = int(num_subsets * subset_size)

#     # Start producing features for real and fake images
#     # if device.index is not None:
#     #     # Avoid unbounded memory usage
#     #     gpu_options = tf.compat.v1.GPUOptions(allow_growth=True,
#     #                                 per_process_gpu_memory_fraction=0.15,
#     #                                 visible_device_list=str(device.index))
#     #     config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

#     # else:
#     #     config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})

#     config = tf.compat.v1.ConfigProto()
#     config.gpu_options.per_process_gpu_memory_fraction = 0.2
#     config.gpu_options.allow_growth = True

#     with tf.compat.v1.Session(config=config) as sess:
#         sess.run(tf.compat.v1.global_variables_initializer())

#         real_feat = compute_real_dist_feat(num_samples=num_samples,
#                                            sess=sess,
#                                            dataset=dataset,
#                                            batch_size=batch_size,
#                                            verbose=verbose,
#                                            feat_file=feat_file,
#                                            log_dir=log_dir,
#                                            seed=seed)

#         fake_feat = compute_gen_dist_feat(netG=netG,
#                                           num_samples=num_samples,
#                                           sess=sess,
#                                           device=device,
#                                           seed=seed,
#                                           batch_size=batch_size,
#                                           verbose=verbose)

#         # Compute the KID score
#         scores = kid_utils.polynomial_mmd_averages(real_feat,
#                                                    fake_feat,
#                                                    n_subsets=num_subsets,
#                                                    subset_size=subset_size)

#         mmd_score, mmd_std = float(np.mean(scores)), float(np.std(scores))

#         print("INFO: KID: {:.4f} ± {:.4f} [Time Taken: {:.4f} secs]".format(
#             mmd_score, mmd_std,
#             time.time() - start_time))

#         return mmd_score, mmd_std
