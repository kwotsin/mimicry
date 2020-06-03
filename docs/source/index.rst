:github_url: https://github.com/kwotsin/mimicry

Mimicry Documentation
=====================

`Mimicry <https://github.com/kwotsin/mimicry>`_ is a lightweight PyTorch library aimed towards the reproducibility of GAN research.

Comparing GANs is often difficult - mild differences in implementations and evaluation methodologies can result in huge performance differences. Mimicry aims to resolve this by providing:

- Standardized implementations of popular GANs that closely reproduce reported scores

- Baseline scores of GANs trained and evaluated under the same conditions

- A framework for researchers to focus on implementation of GANs without rewriting most of GAN training boilerplate code, with support for multiple GAN evaluation metrics.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Guides

   guides/introduction
   guides/tutorial
   guides/baselines

.. toctree::
   :caption: API Reference
   :maxdepth: 1

   modules/nets
   modules/modules
   modules/training
   modules/metrics
   modules/datasets
   modules/utils
