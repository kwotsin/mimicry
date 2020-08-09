"""
PyTorch interface for computing Inception Score.
"""
import os
import random
import time

import numpy as np
import torch

from torch_mimicry.metrics.inception_model import inception_utils
from torch_mimicry.metrics.inception_score import inception_score_utils as tf_inception_score


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


def inception_score(num_samples,
                    netG,
                    device=None,
                    batch_size=50,
                    splits=10,
                    log_dir='./log',
                    seed=0,
                    print_every=20):
    """
    Computes the inception score of generated images.

    Args:
        netG (Module): The generator model to use for generating images.
        device (str/torch.device): Device identifier to use for computation.
        num_samples (int): The number of samples to generate.
        batch_size (int): Batch size per feedforward step for inception model.
        splits (int): The number of splits to use for computing IS.
        log_dir (str): Path to store metric computation objects.
        seed (int): Random seed for generation.
    Returns:
        Mean and standard deviation of the inception score computed from using
        num_samples generated images.
    """
    start_time = time.time()

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # Make sure the random seeds are fixed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Build inception
    inception_path = os.path.join(log_dir, 'metrics/inception_model')
    inception_utils.create_inception_graph(inception_path)

    # Inference variables
    batch_size = min(batch_size, num_samples)
    num_batches = num_samples // batch_size

    # Get images
    images = []
    with torch.no_grad():
        start_time = time.time()
        for idx in range(num_batches):
            # noise = torch.randn((batch_size, netG.nz), device=device)
            # fake_images = netG(noise)

            fake_images = netG.generate_images(num_images=batch_size,
                                               device=device).detach().cpu()

            fake_images = _normalize_images(fake_images)
            images.append(fake_images)

            if (idx + 1) % min(print_every, num_batches) == 0:
                end_time = time.time()
                print(
                    "INFO: Generated image {}/{} [Random Seed {}] ({:.4f} sec/idx)"
                    .format(
                        (idx + 1) * batch_size, num_samples, seed,
                        (end_time - start_time) / (print_every * batch_size)))
                start_time = end_time

    images = np.concatenate(images, axis=0)

    is_mean, is_std = tf_inception_score.get_inception_score(images,
                                                             splits=splits,
                                                             device=device)

    print("INFO: Inception Score: {:.4f} ± {:.4f} [Time Taken: {:.4f} secs]".
          format(is_mean, is_std,
                 time.time() - start_time))

    return is_mean, is_std
