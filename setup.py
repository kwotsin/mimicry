from setuptools import setup, find_packages

__version__ = '0.1.16'
url = 'https://github.com/kwotsin/mimicry'

install_requires = [
    'numpy',
    'scipy',
    'requests',
    'torch',
    'tensorflow',
    'torchvision',
    'six',
    'matplotlib',
    'Pillow',
    'scikit-image',
    'pytest',
    'scikit-learn',
    'future',
    'pytest-cov',
    'pandas',
    'psutil',
    'yapf',
    'lmdb',
]

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
