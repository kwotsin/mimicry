:github_url: https://github.com/kwotsin/mimicry

Mimicry Documentation
===============================

[Mimicry](https://github.com/kwotsin/mimicry) is a lightweight PyTorch library aimed towards the reproducibility of GAN research.

Comparing GANs is often difficult - mild differences in implementations and evaluation methodologies can result in huge performance differences. Mimicry aims to resolve this by providing: (a) Standardized implementations of popular GANs that closely reproduce reported scores; (b) Baseline scores of GANs trained and evaluated under the same conditions; (c) A framework for researchers to focus on implementation of GANs without rewriting most of GAN training boilerplate code, with support for multiple GAN evaluation metrics.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Guides

   notes/installation
   notes/introduction
