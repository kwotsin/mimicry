





![alt text](https://github.com/kwotsin/mimicry/blob/master/docs/images/mimicry_logo.png)

-----
[![CircleCI](https://circleci.com/gh/kwotsin/mimicry.svg?style=shield)](https://circleci.com/gh/kwotsin/mimicry) [![codecov](https://codecov.io/gh/kwotsin/mimicry/branch/master/graph/badge.svg)](https://codecov.io/gh/kwotsin/mimicry) [![PyPI version](https://badge.fury.io/py/torch-mimicry.svg)](https://badge.fury.io/py/torch-mimicry) [![Documentation Status](https://readthedocs.org/projects/mimicry/badge/?version=latest)](https://mimicry.readthedocs.io/en/latest/?badge=latest)
 [![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)


[About](https://kwotsin.github.io/post/introducing-mimicry/) | [Documentation](https://mimicry.readthedocs.io/en/latest/index.html) | [Tutorial](https://mimicry.readthedocs.io/en/latest/guides/tutorial.html) | [Gallery](https://github.com/kwotsin/mimicry/tree/master/docs/gallery/README.md) | [Paper](https://arxiv.org/abs/2005.02494)

Mimicry is a lightweight PyTorch library aimed towards the reproducibility of GAN research.

Comparing GANs is often difficult - mild differences in implementations and evaluation methodologies can result in huge performance differences. Mimicry aims to resolve this by providing: (a) Standardized implementations of popular GANs that closely reproduce reported scores; (b) Baseline scores of GANs trained and evaluated under the *same conditions*; (c\) A framework for researchers to focus on *implementation* of GANs without rewriting most of GAN training boilerplate code, with support for multiple GAN evaluation metrics.

We provide a model zoo and set of [baselines](#baselines) to benchmark different GANs of the same model size trained under the same conditions, using multiple metrics. To ensure [reproducibility](#reproducibility),  we verify scores of our implemented models against reported scores in literature.

-----
## Installation
The library can be installed with:
```
pip install git+https://github.com/kwotsin/mimicry.git
```

See also [setup information](https://mimicry.readthedocs.io/en/latest/guides/introduction.html) for more.

## Example Usage
Training a popular GAN like [SNGAN](https://arxiv.org/abs/1802.05957) that *reproduces reported scores* can be done as simply as:
```python
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
```
Example outputs:
```
>>> INFO: [Epoch 1/127][Global Step: 10/100000]
| D(G(z)): 0.5941
| D(x): 0.9303
| errD: 1.4052
| errG: -0.6671
| lr_D: 0.0002
| lr_G: 0.0002
| (0.4550 sec/idx)
^CINFO: Saving checkpoints from keyboard interrupt...
INFO: Training Ended
```
Tensorboard visualizations:
```
tensorboard --logdir=./log/example
```
See further details in [example script](https://github.com/kwotsin/mimicry/blob/master/examples/sngan_example.py), as well as a detailed [tutorial](https://mimicry.readthedocs.io/en/latest/guides/tutorial.html) on implementing a custom GAN from scratch.

### Further Guides
- [Evaluating a pre-trained generator model](https://github.com/kwotsin/mimicry/blob/master/examples/eval_pretrained.py)
- [Evaluation using custom datasets](https://github.com/kwotsin/mimicry/blob/master/examples/eval_pretrained.py)
- [Implementing, training and evaluating a custom GAN](https://mimicry.readthedocs.io/en/latest/guides/tutorial.html)

<div id="baselines"></div>

## Baselines | Model Zoo

For a fair comparison, we train all models under the same training conditions for each dataset, each implemented using ResNet backbones of the same architectural capacity. We train our models with the Adam optimizer using the popular hyperparameters (β<sub>1</sub>, β<sub>2</sub>)  = (0.0, 0.9).  n<sub>dis</sub> represents the number of discriminator update steps per generator update step, and n<sub>iter</sub> is simply the number of training iterations.

#### Models
| Abbrev. | Name | Type* |
|:-----------:|:---------------------------------------------:|:-------------:|
| DCGAN | Deep Convolutional GAN | Unconditional |
| WGAN-GP | Wasserstein GAN with Gradient Penalty | Unconditional |
| SNGAN | Spectral Normalization GAN | Unconditional |
| cGAN-PD | Conditional GAN with Projection Discriminator | Conditional |
| SSGAN | Self-supervised GAN | Unconditional |
| InfoMax-GAN | Infomax-GAN | Unconditional |

**Conditional GAN scores are only reported for labelled datasets.*

#### Metrics
| Metric | Method |
|:--------------------------------:|:---------------------------------------:|
| [Inception Score (IS)*](https://arxiv.org/abs/1606.03498) | 50K samples at 10 splits|
| [Fréchet Inception Distance (FID)](https://arxiv.org/abs/1706.08500) | 50K real/generated samples |
| [Kernel Inception Distance (KID)](https://arxiv.org/abs/1801.01401) | 50K real/generated samples, averaged over 10 splits.|

**Inception Score can be a poor indicator of GAN performance, as it does not measure diversity and is not domain agnostic. This is why certain datasets with only a single class (e.g. CelebA and LSUN-Bedroom) will perform poorly when using this metric.*

#### Datasets
| Dataset | Split | Resolution |
|:------------:|:---------:|:----------:|
| CIFAR-10 | Train | 32 x 32 |
| CIFAR-100 | Train | 32 x 32 |
| ImageNet | Train | 32 x 32 |
| STL-10 | Unlabeled | 48 x 48 |
| CelebA | All | 64 x 64 |
| CelebA | All | 128 x 128 |
| LSUN-Bedroom | Train | 128 x 128 |
| ImageNet | Train | 128 x 128 |

------

### CelebA
[Paper](https://arxiv.org/abs/1411.7766) | [Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

#### Training Parameters
| Resolution | Batch Size | Learning Rate | β<sub>1</sub> | β<sub>2</sub> | Decay Policy | n<sub>dis</sub> | n<sub>iter</sub> |
|:----------:|:----------:|:-------------:|:-------------:|:-------------:|:------------:|:---------------:|------------------|
| 128 x 128 | 64 | 2e-4 | 0.0 | 0.9 | None | 2 | 100K |
| 64 x 64 | 64 | 2e-4 | 0.0 | 0.9 | Linear | 5 | 100K |

#### Results
| Resolution | Model | IS | FID | KID | Checkpoint | Code |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 128 x 128 | SNGAN | 2.72 ± 0.01 | 12.93 ± 0.04 | 0.0076 ± 0.0001 | [netG.pth](https://drive.google.com/open?id=1rYnv2tCADbzljYlnc8Ypy-JTTipJlRyN) | [sngan_128.py](torch_mimicry/nets/sngan/sngan_128.py) |
| 128 x 128 | SSGAN | 2.63 ± 0.01 | 15.18 ± 0.10 | 0.0101 ± 0.0001 | [netG.pth](https://drive.google.com/open?id=1Gn4ouslRAHq3D7AP_V-T2x8Wi1S1hTXJ) | [ssgan_128.py](torch_mimicry/nets/ssgan/ssgan_128.py) |
| 128 x 128 | InfoMax-GAN | 2.84 ± 0.01 | 9.50 ± 0.04 | 0.0063 ± 0.0001 | [netG.pth](https://drive.google.com/open?id=1C0jzZmx2uSUvBlQpAo3aRay1l8fB4McU) | [infomax_gan_128.py](torch_mimicry/nets/infomax_gan/infomax_gan_128.py) |
| 64 x 64 | SNGAN | 2.68 ± 0.01 | 5.71 ± 0.02 | 0.0033 ± 0.0001 | [netG.pth](https://drive.google.com/open?id=1d9pYKKW9Hgi-ylOKwU0UM1ZsmUyD8Bpj) | [sngan_64.py](torch_mimicry/nets/sngan/sngan_64.py) |
| 64 x 64 | SSGAN | 2.67 ± 0.01 | 6.03 ± 0.04 | 0.0036 ± 0.0001 | [netG.pth](https://drive.google.com/open?id=1wM23g566VrHTJzC0F7X5MgxgcAzzhbLr) | [ssgan_64.py](torch_mimicry/nets/ssgan/ssgan_64.py) |
| 64 x 64 | InfoMax-GAN |2.68 ± 0.01 | 5.71 ± 0.06 | 0.0033 ± 0.0001 | [netG.pth](https://drive.google.com/open?id=1bvRvkOgZAuN89sVxfdNS16j6kAyKwN8z) | [infomax_gan_64.py](torch_mimicry/nets/infomax_gan/infomax_gan_64.py) |

### LSUN-Bedroom
[Paper](https://arxiv.org/abs/1506.03365) | [Dataset](https://github.com/fyu/lsun)

#### Training Parameters
| Resolution | Batch Size | Learning Rate | β<sub>1</sub> | β<sub>2</sub> | Decay Policy | n<sub>dis</sub> | n<sub>iter</sub> |
|:----------:|:----------:|:-------------:|:-------------:|:-------------:|:------------:|:---------------:|------------------|
| 128 x 128 | 64 | 2e-4 | 0.0 | 0.9 | Linear | 2 | 100K |


#### Results
| Resolution | Model | IS | FID | KID | Checkpoint | Code |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 128 x 128 | SNGAN | 2.30 ± 0.01 | 25.87 ± 0.03 | 0.0141 ± 0.0001 | [netG.pth](https://drive.google.com/open?id=1SrvdLM-5mUVYwEztY-6TVAdTKlNPUA7k) | [sngan_128.py](torch_mimicry/nets/sngan/sngan_128.py) |
| 128 x 128 | SSGAN | 2.12 ± 0.01 | 12.02 ± 0.07 | 0.0077 ± 0.0001 | [netG.pth](https://drive.google.com/open?id=1uSQHr0PMeCdpWAFPwbn-6YgZX0w0TApu) | [ssgan_128.py](torch_mimicry/nets/ssgan/ssgan_128.py) |
| 128 x 128 | InfoMax-GAN |2.22 ± 0.01 | 12.13 ± 0.16 | 0.0080 ± 0.0001 | [netG.pth](https://drive.google.com/open?id=1PNaYmzLb66D-JXdzVimZ-EiNjI7qAwu_) | [infomax_gan_128.py](torch_mimicry/nets/infomax_gan/infomax_gan_128.py) |

### STL-10
[Paper](http://proceedings.mlr.press/v15/coates11a.html) | [Dataset](https://ai.stanford.edu/~acoates/stl10/)

#### Training Parameters
| Resolution | Batch Size | Learning Rate | β<sub>1</sub> | β<sub>2</sub> | Decay Policy | n<sub>dis</sub> | n<sub>iter</sub> |
|:----------:|:----------:|:-------------:|:-------------:|:-------------:|:------------:|:---------------:|------------------|
| 48 x 48 | 64 | 2e-4 | 0.0 | 0.9 | Linear | 5 | 100K |

#### Results
| Resolution | Model | IS | FID | KID | Checkpoint | Code |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 48 x 48 | WGAN-GP | 8.55 ± 0.02 | 43.01 ± 0.19 | 0.0440 ± 0.0003 | [netG.pth](https://drive.google.com/file/d/1-Fw_xzl_D7P05wL7HmSPGtPPiH3zTeG7/view?usp=sharing) | [wgan_gp_48.py](torch_mimicry/nets/wgan_gp/wgan_gp_48.py) |
| 48 x 48 | SNGAN | 8.04 ± 0.07 | 39.56 ± 0.10 | 0.0369 ± 0.0002 | [netG.pth](https://drive.google.com/open?id=1tVRXDHgUTpBwfGh0RJvdjw2EUOGu-m-A) | [sngan_48.py](torch_mimicry/nets/sngan/sngan_48.py) |
| 48 x 48 | SSGAN | 8.25 ± 0.06 | 37.06 ± 0.19 | 0.0332 ± 0.0004| [netG.pth](https://drive.google.com/open?id=1TBeAhxvxJr3ykKwu4Wgw715D2fAslM-9) | [ssgan_48.py](torch_mimicry/nets/ssgan/ssgan_48.py) |
| 48 x 48 | InfoMax-GAN | 8.54 ± 0.12 | 35.52 ± 0.10 | 0.0326 ± 0.0002 | [netG.pth](https://drive.google.com/open?id=1QHwyvaCqAKyXbkYYzHGNEsDWpBPn05oS) | [infomax_gan_48.py](torch_mimicry/nets/infomax_gan/infomax_gan_48.py) |

### ImageNet
[Paper](https://ieeexplore.ieee.org/document/5206848) | [Dataset](http://www.image-net.org/challenges/LSVRC/)

#### Training Parameters
| Resolution | Batch Size | Learning Rate | β<sub>1</sub> | β<sub>2</sub> | Decay Policy | n<sub>dis</sub> | n<sub>iter</sub> |
|:----------:|:----------:|:-------------:|:-------------:|:-------------:|:------------:|:---------------:|------------------|
| 32 x 32 | 64 | 2e-4 | 0.0 | 0.9 | Linear | 5 | 100K |
| 128 x 128 | 64 | 2e-4 | 0.0 | 0.9 | None | 5 | 450k |

#### Results
| Resolution | Model | IS | FID | KID | Checkpoint | Code |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 128 x 128 | SNGAN | 13.05 ± 0.05 |  65.74 ± 0.31  |  0.0663 ± 0.0004  | [netG.pth](https://drive.google.com/file/d/1ZmjV9iMFgmthCgOKpMp8vKCIVQh_XPiR/view?usp=sharing) | [sngan_128.py](torch_mimicry/nets/sngan/sngan_128.py) |
| 128 x 128 | SSGAN | 13.30 ± 0.03 | 62.48 ± 0.31 | 0.0616 ± 0.0004 | [netG.pth](https://drive.google.com/file/d/1LIqtrlRd2jOcrwLpzD_ztf7Dc2GZwH8g/view?usp=sharing) | [ssgan_128.py](torch_mimicry/nets/ssgan/ssgan_128.py) |
| 128 x 128 | InfoMax-GAN | 13.68 ± 0.06 | 58.91 ± 0.14 | 0.0579 ± 0.0004 | [netG.pth](https://drive.google.com/file/d/1kKfMddYe3gP0Y3xOhq0KD3GNjBwe_k9S/view?usp=sharing) | [infomax_gan_128.py](torch_mimicry/nets/infomax_gan/infomax_gan_128.py) |
| 32 x 32 | SNGAN | 8.97 ± 0.12 | 23.04 ± 0.06  | 0.0157 ± 0.0002 | [netG.pth](https://drive.google.com/open?id=1LF-tNfbVmHHw8onneOz3V5b-j2ff-Kh2) | [sngan_32.py](torch_mimicry/nets/sngan/sngan_32.py) |
| 32 x 32 | cGAN-PD | 9.08 ± 0.17 | 21.17 ± 0.05 | 0.0145 ± 0.0002 | [netG.pth](https://drive.google.com/open?id=1gHduZDIP9QOr-YgkB3_4p4vU-gjCPnRH) | [cgan_pd_32.py](torch_mimicry/nets/cgan_pd/cgan_pd_32.py) |
| 32 x 32 | SSGAN | 9.11 ± 0.12 | 21.79 ± 0.09 | 0.0152 ± 0.0002 | [netG.pth](https://drive.google.com/open?id=1PzdCYzwg4lZ9r9tPf6I7XItoIu5lHhwy) | [ssgan_32.py](torch_mimicry/nets/ssgan/ssgan_32.py) |
| 32 x 32 | InfoMax-GAN | 9.04 ± 0.10 | 20.68 ± 0.02 | 0.0149 ± 0.0001 | [netG.pth](https://drive.google.com/open?id=1aKtU9P9eccqSmPZrlMRy2eqmd0BJT0vo) | [infomax_gan_32.py](torch_mimicry/nets/infomax_gan/infomax_gan_32.py) |

### CIFAR-10
[Paper](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) | [Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

#### Training Parameters
| Resolution | Batch Size | Learning Rate | β<sub>1</sub> | β<sub>2</sub> | Decay Policy | n<sub>dis</sub> | n<sub>iter</sub> |
|:----------:|:----------:|:-------------:|:-------------:|:-------------:|:------------:|:---------------:|------------------|
| 32 x 32 | 64 | 2e-4 | 0.0 | 0.9 | Linear | 5 | 100K |

#### Results
| Resolution | Model | IS | FID | KID | Checkpoint | Code |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 32 x 32 | WGAN-GP | 7.33 ± 0.02 | 22.29 ± 0.06 | 0.0204± 0.0004 | [netG.pth](https://drive.google.com/file/d/1KL9uDpSZ1Fwzt-4Mke1o88w8ftX01OxR/view?usp=sharing) | [wgan_gp_32.py](torch_mimicry/nets/wgan_gp/wgan_gp_32.py) |
| 32 x 32 | SNGAN | 7.97 ± 0.06 | 16.77 ± 0.04 | 0.0125 ± 0.0001 | [netG.pth](https://drive.google.com/open?id=16TxezS7VBTiuPdQdjDUM0-_rHXqyPrvr) | [sngan_32.py](torch_mimicry/nets/sngan/sngan_32.py) |
| 32 x 32 | cGAN-PD | 8.25 ± 0.13 | 10.84 ± 0.03 | 0.0070 ± 0.0001 | [netG.pth](https://drive.google.com/open?id=12nGJZjJyi-RJQFj98MlrSqhA_uST2_1n) | [cgan_pd_32.py](torch_mimicry/nets/cgan_pd/cgan_pd_32.py) |
| 32 x 32 | SSGAN | 8.17 ± 0.06 | 14.65 ± 0.04 |  0.0101 ± 0.0002 | [netG.pth](https://drive.google.com/open?id=1sceT1tUPw2wRVqLcpg1EkWpz8Qa5j_fE) | [ssgan_32.py](torch_mimicry/nets/ssgan/ssgan_32.py) |
| 32 x 32 | InfoMax-GAN | 8.08± 0.08 | 15.12 ± 0.10 | 0.0112 ± 0.0001 | [netG.pth](https://drive.google.com/open?id=15k7HmwcKBDL0wjpV4XG9AGs-7yzxuFKV) | [infomax_gan_32.py](torch_mimicry/nets/infomax_gan/infomax_gan_32.py) |

### CIFAR-100
[Paper](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) | [Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

#### Training Parameters
| Resolution | Batch Size | Learning Rate | β<sub>1</sub> | β<sub>2</sub> | Decay Policy | n<sub>dis</sub> | n<sub>iter</sub> |
|:----------:|:----------:|:-------------:|:-------------:|:-------------:|:------------:|:---------------:|------------------|
| 32 x 32 | 64 | 2e-4 | 0.0 | 0.9 | Linear | 5 | 100K |

#### Results
| Resolution | Model | IS | FID | KID | Checkpoint | Code |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 32 x 32 | SNGAN | 7.57 ± 0.11 | 22.61 ± 0.06 | 0.0156 ± 0.0003 | [netG.pth](https://drive.google.com/open?id=16XtxZd6ARXjfTBuOZml_FpR0WVaWkToA) | [sngan_32.py](torch_mimicry/nets/sngan/sngan_32.py) |
| 32 x 32 | cGAN-PD | 8.92 ± 0.07 | 14.16 ± 0.01 | 0.0085 ± 0.0002 | [netG.pth](https://drive.google.com/open?id=1wQY5IbqD_CcuVwpGZXfMdWxh7OZHrQxM) | [cgan_pd_32.py](torch_mimicry/nets/cgan_pd/cgan_pd_32.py) |
| 32 x 32 | SSGAN | 7.56 ± 0.07 | 22.18 ± 0.10 | 0.0161 ± 0.0002 | [netG.pth](https://drive.google.com/open?id=1dkO0mf6S44tvrtqRYtTABmIlZQ0iCmNw) | [ssgan_32.py](torch_mimicry/nets/ssgan/ssgan_32.py) |
| 32 x 32 | InfoMax-GAN | 7.86 ± 0.10 | 18.94 ± 0.13 | 0.0135 ± 0.0004 | [netG.pth](https://drive.google.com/open?id=1gkjytt_fdWIKsGOTHruahTQK-SuBds42) | [infomax_gan_32.py](torch_mimicry/nets/infomax_gan/infomax_gan_32.py) |

-------

<div id="reproducibility"></div>

## Reproducibility
To verify our implementations, we reproduce reported scores in literature by re-implementing the models with the same architecture, training them under the same conditions and evaluate them on CIFAR-10 using the exact same methodology for computing FID.

As FID produces highly biased estimates (where using larger samples lead to a lower score), we reproduce the scores using the same sample sizes, where n<sub>real</sub> and n<sub>fake</sub> refers to the number of real and fake images used respectively for computing FID.

| Metric              | Model       | Score           | Reported Score | n<sub>real</sub>| n<sub>fake</sub>| Checkpoint |  Code |
|:-------------------:|:-----------:|:---------------:|:--------------:|:------------------:|:----:|:----:|:---:|
| FID                 | DCGAN       | 28.95 ± 0.42    | 28.12     [4]  | 10K  | 10K | [netG.pth](https://drive.google.com/open?id=1IOcgdkVFScASEpBftWLxhwhhAPRgwSDt) | [dcgan_cifar.py](torch_mimicry/nets/dcgan/dcgan_cifar.py)
| FID				  | WGAN-GP     | 26.08 ± 0.12    | 29.3 <sup>†</sup> [6] | 50K | 50K | [netG.pth](https://drive.google.com/open?id=1HVBlp2cK_hT0J_5cRqp_JOvysJjBPrrj) | [wgan_gp_32.py](torch_mimicry/nets/wgan_gp/wgan_gp_32.py)
| FID                 | SNGAN       | 23.90 ± 0.20    | 21.7 ± 0.21 [1]| 10K | 5K | [netG.pth](https://drive.google.com/open?id=1aLbds1EGFoWZZF2y1l8mvPCvnb7AGHh7) | [sngan_32.py](torch_mimicry/nets/sngan/sngan_32.py)
| FID                 | cGAN-PD     | 17.84 ± 0.17    | 17.5 [2]      | 10K | 5K | [netG.pth](https://drive.google.com/open?id=1v26E7mIk8PqIg5ldE-iQNMQ2AYQ36iGM) | [cgan_pd_32.py](torch_mimicry/nets/cgan_pd/cgan_pd_32.py)
| FID                 | SSGAN       |  17.61 ± 0.14  | 17.88 ± 0.64 [3]     | 10K | 10K | [netG.pth](https://drive.google.com/open?id=1lgOk4lC0-p4PoXvju9m5p0lOGoZDWELe) | [ssgan_32.py](torch_mimicry/nets/ssgan/ssgan_32.py)
| FID                 | InfoMax-GAN | 17.14 ± 0.20    | 17.14 ± 0.20 [5]   | 50K | 10K | [netG.pth](https://drive.google.com/open?id=1hgoN6Nx59j_JuaCup2rBPC4koih9qZ1y) | [infomax_gan_32.py](torch_mimicry/nets/infomax_gan/infomax_gan_32.py)


*<sup>†</sup> Best FID was reported at 53K steps, but we find our score can improve till 100K steps to achieve 23.13 ± 0.13.*

## Citation
If you have found this work useful, please consider citing [our work](https://arxiv.org/abs/2005.02494):
```
@article{lee2020mimicry,
    title={Mimicry: Towards the Reproducibility of GAN Research},
    author={Kwot Sin Lee and Christopher Town},
    booktitle={CVPR Workshop on AI for Content Creation},
    year={2020},
}
```
For citing [InfoMax-GAN](https://arxiv.org/abs/2007.04589):
```
@InProceedings{Lee_2021_WACV,
    author    = {Lee, Kwot Sin and Tran, Ngoc-Trung and Cheung, Ngai-Man},
    title     = {InfoMax-GAN: Improved Adversarial Image Generation via Information Maximization and Contrastive Learning},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {3942-3952}
}
```

## References
[[1] Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)

[[2] cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637)

[[3] Self-Supervised GANs via Auxiliary Rotation Loss](https://arxiv.org/abs/1811.11212)

[[4] A Large-Scale Study on Regularization and Normalization in GANs](https://arxiv.org/abs/1807.04720)

[[5] InfoMax-GAN: Improved Adversarial Image Generation via Information Maximization and Contrastive Learning](https://arxiv.org/abs/2007.04589)

[[6] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
