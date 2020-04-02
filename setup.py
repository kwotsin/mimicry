from setuptools import setup, find_packages

__version__ = '0.1.8'
url = 'https://github.com/kwotsin/mimicry'

# install_requires = [
#     'numpy>=1.15.4',
#     'scipy>=1.0.1',
#     'requests>=2.22.0',
#     'tensorflow>=1.14.0,<2.0',
#     'torchvision>=0.4.0',
#     'six>=1.12.0',
#     'matplotlib>=3.1.1',
#     'torch>=1.2.0',
#     'Pillow>=6.2.0',
#     'scikit-image>=0.15.0',
#     'pytest>=5.3.2',
#     'scikit-learn>=0.20.1',
#     'future>=0.17.1',
#     'pytest-cov>=2.8.1',
#     'pandas>=0.25.3',
#     'psutil>=5.7.0',
#     'yapf>=0.29.0',
# ]

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov', 'mock']

long_description = """
Mimicry is a lightweight PyTorch library aimed towards the reproducibility of GAN research.

Comparing GANs is often difficult - mild differences in implementations and evaluation methodologies can result in huge performance differences.
Mimicry aims to resolve this by providing:
    (a) Standardized implementations of popular GANs that closely reproduce reported scores;
    (b) Baseline scores of GANs trained and evaluated under the same conditions;
    (c) A framework for researchers to focus on implementation of GANs without rewriting most of GAN training boilerplate code, with support for multiple GAN evaluation metrics.

We provide a model zoo and set of baselines to benchmark different GANs of the same model size trained under the same conditions, using multiple metrics. To ensure reproducibility, we verify scores of our implemented models against reported scores in literature.
"""

setup(
    name='torch_mimicry',
    version=__version__,
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Mimicry: Towards the Reproducibility of GAN Research',
    author='Kwot Sin Lee',
    author_email='ksl36@cam.ac.uk',
    url=url,
    download_url='{}/archive/{}.tar.gz'.format(url, __version__),
    keywords=[
        'pytorch',
        'generative-adversarial-networks',
        'gans',
        'GAN',
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    packages=find_packages(),
)
