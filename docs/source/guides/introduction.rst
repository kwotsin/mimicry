Introduction
============

Installation
------------
For best performance, we recommend you to install the GPU versions of both TensorFlow and PyTorch, which are used in this library. 

First create a new environment with :code:`conda` using Python 3.6 or 3.7 (Python 3.8 is currently not supported by TensorFlow):

.. code-block:: none

    $ conda create -n mimicry python=3.6

Install TensorFlow (GPU):

.. code-block:: none

    $ conda install -c anaconda tensorflow-gpu

Install PyTorch (GPU):

.. code-block:: none

    $ conda install pytorch torchvision cudatoolkit=9.2 -c pytorch

For installing the CUDA version matching your drivers, see the `official instructions <https://pytorch.org/>`_.

Mimicry can be installed using :code:`pip` directly:

.. code-block:: none

    $ pip install torch-mimicry

.. To install the GPU versions of TensorFlow and PyTorch see the official `PyTorch <https://pytorch.org/>`_  and `TensorFlow <https://www.tensorflow.org/install>`_ pages based on your system. The library is currently only compatible with :code:`TensorFlow 1.x` to accommodate the original implementations of the GAN metrics.

Quick Start
-----------
We provide a sample training script for training the `Spectral Normalization GAN <https://arxiv.org/abs/1802.05957>`_ on the CIFAR-10 dataset, with the same training hyperparameters that reproduce results in the paper.


.. code-block:: python

    import torch
    import torch.optim as optim
    import torch_mimicry as mmc
    from torch_mimicry.nets import sngan


    # Data handling objects
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    dataset = mmc.datasets.load_dataset(root='./datasets', name='cifar10')
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=True, num_workers=4)

    # Define models and optimizers
    netG = sngan.SNGANGenerator32().to(device)
    netD = sngan.SNGANDiscriminator32().to(device)
    optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
    optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

    # Start training
    trainer = mmc.training.Trainer(
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        n_dis=5,
        num_steps=100000,
        lr_decay='linear',
        dataloader=dataloader,
        log_dir='./log/example',
        device=device)
    trainer.train()

To evaluate its FID, we can simply run the following:

.. code-block:: python

    # Evaluate fid
    mmc.metrics.evaluate(
        metric='fid',
        log_dir='./log/example',
        netG=netG,
        dataset='cifar10',
        num_real_samples=50000,
        num_fake_samples=50000,
        evaluate_step=100000,
        device=device)

Alternatively, one could evaluate FID progressively over an interval by swapping the `evaluate_step` argument for `evaluate_range`:

.. code-block:: python

    # Evaluate fid
    mmc.metrics.evaluate(
        metric='fid',
        log_dir='./log/example',
        netG=netG,
        dataset='cifar10',
        num_real_samples=50000,
        num_fake_samples=50000,
        evaluate_range=(5000, 100000, 5000),
        device=device)

We support other datasets and models See `datasets <https://mimicry.readthedocs.io/en/latest/modules/datasets.html>`_ and `nets <https://mimicry.readthedocs.io/en/latest/modules/nets.html>`_ for more information.

Visualizations
--------------
Mimicry provides TensorBoard support for visualizing the following:

- Loss and probability curves for monitoring GAN training
- Randomly generated images for checking diversity.
- Generated images from a fixed set of noise vectors.

.. code-block:: none

    $ tensorboard --logdir=./log/example
