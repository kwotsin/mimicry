"""
Typical usage example.
"""

import torch
import torch.optim as optim
import torch_mimicry as mmc
from torch_mimicry.nets import sngan

if __name__ == "__main__":
    # Data handling objects
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    dataset = mmc.datasets.load_dataset(root='./datasets', name='cifar10')
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=64,
                                             shuffle=True,
                                             num_workers=4)

    # Define models and optimizers
    netG = sngan.SNGANGenerator32().to(device)
    netD = sngan.SNGANDiscriminator32().to(device)
    optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
    optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

    # Start training
    trainer = mmc.training.Trainer(netD=netD,
                                   netG=netG,
                                   optD=optD,
                                   optG=optG,
                                   n_dis=5,
                                   num_steps=30,
                                   lr_decay='linear',
                                   dataloader=dataloader,
                                   log_dir='./log/example',
                                   device=device)
    trainer.train()

    # Evaluate fid
    mmc.metrics.evaluate(metric='fid',
                         log_dir='./log/example',
                         netG=netG,
                         dataset='cifar10',
                         num_real_samples=50000,
                         num_fake_samples=50000,
                         evaluate_step=30,
                         device=device)

    # Evaluate kid
    mmc.metrics.evaluate(metric='kid',
                         log_dir='./log/example',
                         netG=netG,
                         dataset='cifar10',
                         num_samples=50000,
                         evaluate_step=30,
                         device=device)

    # Evaluate inception score
    mmc.metrics.evaluate(metric='inception_score',
                         log_dir='./log/example',
                         netG=netG,
                         num_samples=50000,
                         evaluate_step=30,
                         device=device)
