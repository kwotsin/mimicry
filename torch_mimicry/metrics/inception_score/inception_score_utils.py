"""
Helper functions for computing inception score, as based on:
https://github.com/openai/improved-gan/tree/master/inception_score
"""
import time

import numpy as np
import tensorflow as tf

from torch_mimicry.metrics.inception_model import inception_utils


def get_predictions(images, device=None, batch_size=50, print_every=20):
    """
    Get the output probabilities of images.

    Args:
        images (ndarray): Batch of images of shape (N, H, W, 3).
        device (Device): Torch device object.
        batch_size (int): Batch size for inference using inception model.
        print_every (int): Prints logging variable every n batch inferences.

    Returns:
        ndarray: Batch of probabilities of equal size as number of images input.
    """
    # if device and device.index is not None:
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

    # Inference variables
    batch_size = min(batch_size, images.shape[0])
    num_batches = images.shape[0] // batch_size

    # Get predictions
    preds = []
    with tf.compat.v1.Session(config=config) as sess:
        # Batch input preparation
        inception_utils._get_inception_layer(sess)

        # Define input/outputs of default graph.
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        # pool3 = sess.graph.get_tensor_by_name('inception_model/pool_3:0') # TODO: Remove when safe. TF2 syntax changes again.
        # w = sess.graph.get_operation_by_name(
        #     "inception_model/softmax/logits/MatMul").inputs[1] # TODO: Remove when safe. TF2 syntax changes again.
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
        softmax = tf.nn.softmax(logits)

        # Predict images
        start_time = time.time()
        for i in range(num_batches):
            batch = images[i * batch_size:(i + 1) * batch_size]

            pred = sess.run(softmax, {'ExpandDims:0': batch})
            # pred = sess.run(softmax, {'inception_model/ExpandDims:0': batch}) # TODO: Remove when safe. TF2 syntax changes again.
            preds.append(pred)

            if (i + 1) % min(print_every, num_batches) == 0:
                end_time = time.time()
                print("INFO: Processed image {}/{}...({:.4f} sec/idx)".format(
                    (i + 1) * batch_size, images.shape[0],
                    (end_time - start_time) / (print_every * batch_size)))
                start_time = end_time

    preds = np.concatenate(preds, 0)

    return preds


def get_inception_score(images, splits=10, device=None):
    """
    Computes inception score according to official OpenAI implementation.

    Args:
        images (ndarray): Batch of images of shape (N, H, W, 3), which should have values
            in the range [0, 255].
        splits (int): Number of splits to use for computing IS.
        device (Device): Torch device object to decide which GPU to use for TF session.

    Returns:
        tuple: Tuple of mean and standard deviation of the inception score computed.
    """
    if np.max(images[0] < 10) and np.max(images[0] < 0):
        raise ValueError("Images should have value ranging from 0 to 255.")

    # Load graph and get probabilities
    preds = get_predictions(images, device=device)

    # Compute scores
    N = preds.shape[0]
    scores = []
    for i in range(splits):
        part = preds[(i * N // splits):((i + 1) * N // splits), :]
        kl = part * (np.log(part) -
                     np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    return float(np.mean(scores)), float(np.std(scores))
