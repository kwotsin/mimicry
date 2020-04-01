import torch
import torch.optim as optim
import torch_mimicry as mmc
from torch_mimicry.nets import sngan


if __name__ == "__main__":
    # Data handling objects
    dataset = mmc.datasets.load_dataset(root='./datasets', name='cifar10')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=4)

    # Define models and optimizers
    netG = sngan.SNGANGenerator32()
    netD = sngan.SNGANDiscriminator32()
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
        log_dir='./log/example')
    trainer.train()

    # Evaluate fid
    mmc.metrics.evaluate(
        metric='fid',
        log_dir='./log/example',
        netG=netG,
        dataset_name='cifar10',
        num_real_samples=50000,
        num_fake_samples=50000,
        evaluate_step=100000)
