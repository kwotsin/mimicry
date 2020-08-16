"""
Example script of evaluating a pretrained generator.
"""
import torch
import torch_mimicry as mmc
from torch_mimicry.nets import sngan

######################################################
#       Computing Metrics with Default Datasets
######################################################

# Download cifar10 checkpoint: https://drive.google.com/uc?id=1Gn4ouslRAHq3D7AP_V-T2x8Wi1S1hTXJ&export=download
ckpt_file = "./log/sngan_example/checkpoints/netG/netG_100000_steps.pth"

# Default variables
log_dir = './examples/example_log'
dataset = 'cifar10'
seed = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# Restore model
netG = sngan.SNGANGenerator32().to(device)
netG.restore_checkpoint(ckpt_file)

# Metrics with a known/popular dataset.
mmc.metrics.fid_score(num_real_samples=50000,
                      num_fake_samples=50000,
                      netG=netG,
                      seed=seed,
                      dataset=dataset,
                      log_dir=log_dir,
                      device=device)

mmc.metrics.kid_score(num_samples=50000,
                      netG=netG,
                      seed=seed,
                      dataset=dataset,
                      log_dir=log_dir,
                      device=device)

mmc.metrics.inception_score(num_samples=50000,
                            netG=netG,
                            seed=seed,
                            log_dir=log_dir,
                            device=device)

######################################################
#       Computing Metrics with Custom Datasets
######################################################
"""
Simply define a custom dataset as below to compute FID/KID, and define
a stats_file/feat_file to save the cached statistics since we don't know what
name to give your file.
"""


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.ones(1000, 3, 32, 32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


custom_dataset = CustomDataset()

# Metrics with a custom dataset.
mmc.metrics.fid_score(num_real_samples=1000,
                      num_fake_samples=1000,
                      netG=netG,
                      seed=seed,
                      dataset=custom_dataset,
                      log_dir=log_dir,
                      device=device,
                      stats_file='./examples/example_log/fid_stats.npz')

mmc.metrics.kid_score(num_samples=1000,
                      netG=netG,
                      seed=seed,
                      dataset=custom_dataset,
                      log_dir=log_dir,
                      device=device,
                      feat_file='./examples/example_log/kid_stats.npz')

# Using the evaluate API, which assumes a more fixed directory.
netG = sngan.SNGANGenerator32().to(device)
mmc.metrics.evaluate(metric='fid',
                     log_dir='./log/sngan_example/',
                     netG=netG,
                     dataset=custom_dataset,
                     num_real_samples=1000,
                     num_fake_samples=1000,
                     evaluate_step=100000,
                     stats_file='./examples/example_log/fid_stats.npz',
                     device=device)