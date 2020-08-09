"""
Common inception utils for computing metrics, as based on the FID helper code:
https://github.com/kwotsin/dissertation/blob/master/eval/TTUR/fid.py
"""
import os
import pathlib
import tarfile
import time
from urllib import request

import numpy as np
import tensorflow as tf


def _check_or_download_inception(inception_path):
    """
    Checks if the path to the inception file is valid, or downloads
    the file if it is not present.

    Args:
        inception_path (str): Directory for storing the inception model.

    Returns:
        str: File path of the inception protobuf model.

    """
    # Build file path of model
    inception_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'

    # Download model if required
    if not model_file.exists():
        print("Downloading Inception model")
        fn, _ = request.urlretrieve(inception_url)

        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))

    return str(model_file)


def _get_inception_layer(sess):
    """
    Prepares inception net for batched usage and returns pool_3 layer.

    Args:
        sess (Session): TensorFlow Session object.

    Returns:
        TensorFlow graph node representing inception model pool3 layer output.

    """
    # Get the output node
    # layer_name = 'inception_model/pool_3:0' # TODO: Remove when safe. TF2 syntax changes again.
    layer_name = 'pool_3:0'

    pool3 = sess.graph.get_tensor_by_name(layer_name)

    # Reshape to be batch size agnostic
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if len(shape._dims) > 0:
                try:
                    shape = [s.value for s in shape]
                except AttributeError:  # TF 2 uses None shape directly. No conversion needed.
                    shape = shape

                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.__dict__['_shape_val'] = tf.TensorShape(new_shape)

    return pool3


def get_activations(images, sess, batch_size=50, verbose=True):
    """
    Calculates the activations of the pool_3 layer for all images.

    Args:
        images (ndarray): Numpy array of shape (N, C, H, W) with values ranging
            in the range [0, 255].
        sess (Session): TensorFlow Session object.
        batch_size (int): The batch size to use for inference.
        verbose (bool): If True, prints out logging data for batch inference.

    Returns:
        ndarray: Numpy array of shape (N, 2048) representing the pool3 features from the
        inception model.

    """
    # Get output layer.
    inception_layer = _get_inception_layer(sess)

    # Inference variables
    batch_size = min(batch_size, images.shape[0])
    num_batches = images.shape[0] // batch_size

    # Get features
    pred_arr = np.empty((images.shape[0], 2048))
    for i in range(num_batches):
        start_time = time.time()

        start = i * batch_size
        end = start + batch_size
        batch = images[start:end]
        pred = sess.run(inception_layer, {'ExpandDims:0': batch})
        # pred = sess.run(inception_layer,
        #                 {'inception_model/ExpandDims:0': batch}) # TODO: Remove when safe. TF2 syntax changes again.
        pred_arr[start:end] = pred.reshape(batch_size, -1)

        if verbose:
            print("\rINFO: Propagated batch %d/%d (%.4f sec/batch)" \
                % (i+1, num_batches, time.time()-start_time), end="", flush=True)

    return pred_arr


def create_inception_graph(inception_path):
    """
    Creates a graph from saved GraphDef file.

    Args:
        inception_path (str): Directory for storing the inception model.

    Returns:
        None
    """
    if inception_path is None:
        inception_path = '/tmp'

    if not os.path.exists(inception_path):
        os.makedirs(inception_path)

    # Get inception model file path
    model_file = _check_or_download_inception(inception_path)

    # Creates graph from saved graph_def.pb.
    with tf.io.gfile.GFile(model_file, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='inception_model')
