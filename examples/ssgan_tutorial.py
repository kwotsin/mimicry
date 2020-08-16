"""
Tutorial of using SSGAN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import torch_mimicry as mmc
from torch_mimicry.nets import gan
from torch_mimicry.modules import SNLinear
from torch_mimicry.modules import GBlock, DBlock, DBlockOptimized


#######################
#        Models
#######################
class SSGANGenerator(gan.BaseGenerator):
    def __init__(self,
                 nz=128,
                 ngf=256,
                 bottom_width=4,
                 loss_type='hinge',
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         **kwargs)
        self.ss_loss_scale = 0.2

        # Build the layers
        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block3 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block4 = GBlock(self.ngf, self.ngf, upsample=True)
        self.b5 = nn.BatchNorm2d(self.ngf)
        self.c5 = nn.Conv2d(ngf, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def forward(self, x):
        """
        Feedforward function.
        """
        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = torch.tanh(self.c5(h))

        return h

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        """
        Train step function.
        """
        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        # Produce fake images and logits
        fake_images = self.generate_images(num_images=batch_size,
                                           device=device)
        output, _ = netD(fake_images)

        # Compute GAN loss, upright images only.
        errG = self.compute_gan_loss(output)

        # Compute SS loss, rotates the images. -- only for fake images!
        errG_SS = netD.compute_ss_loss(images=fake_images,
                                       scale=self.ss_loss_scale)

        # Backprop and update gradients
        errG_total = errG + errG_SS
        errG_total.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')
        log_data.add_metric('errG_SS', errG_SS, group='loss_SS')

        return log_data


class SSGANDiscriminator(gan.BaseDiscriminator):
    def __init__(self, ndf=128, loss_type='hinge', **kwargs):
        super().__init__(ndf=ndf, loss_type=loss_type, **kwargs)
        self.num_classes = 4
        self.ss_loss_scale = 1.0

        # Build layers
        self.block1 = DBlockOptimized(3, self.ndf)
        self.block2 = DBlock(self.ndf, self.ndf, downsample=True)
        self.block3 = DBlock(self.ndf, self.ndf, downsample=False)
        self.block4 = DBlock(self.ndf, self.ndf, downsample=False)
        self.l5 = SNLinear(self.ndf, 1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)

        # Rotation class prediction layer
        self.l_y = SNLinear(self.ndf, self.num_classes)
        nn.init.xavier_uniform_(self.l_y.weight.data, 1.0)

    def forward(self, x):
        """
        Feedforwards a batch of real/fake images and produces a batch of GAN logits,
        and rotation classes.
        """
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)

        # Global sum pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l5(h)

        # Produce the class output logits
        output_classes = self.l_y(h)

        return output, output_classes

    def _rot_tensor(self, image, deg):
        """
        Rotation for pytorch tensors using rotation matrix. Takes in a tensor of (C, H, W shape).
        """
        if deg == 90:
            return image.transpose(1, 2).flip(1)

        elif deg == 180:
            return image.flip(1).flip(2)

        elif deg == 270:
            return image.transpose(1, 2).flip(2)

        elif deg == 0:
            return image

        else:
            raise NotImplementedError(
                "Function only supports 90,180,270,0 degree rotation.")

    def _rotate_batch(self, images):
        """
        Rotate a quarter batch of images in each of 4 directions.
        """
        N, C, H, W = images.shape
        choices = [(i, i * 4 // N) for i in range(N)]

        # Collect rotated images and labels
        ret = []
        ret_labels = []
        degrees = [0, 90, 180, 270]
        for i in range(N):
            idx, rot_label = choices[i]

            # Rotate images
            image = self._rot_tensor(images[idx],
                                     deg=degrees[rot_label])  # (C, H, W) shape
            image = torch.unsqueeze(image, 0)  # (1, C, H, W) shape

            # Get labels accordingly
            label = torch.from_numpy(np.array(rot_label))  # Zero dimension
            label = torch.unsqueeze(label, 0)

            ret.append(image)
            ret_labels.append(label)

        # Concatenate images and labels to (N, C, H, W) and (N, ) shape respectively.
        ret = torch.cat(ret, dim=0)
        ret_labels = torch.cat(ret_labels, dim=0).to(ret.device)

        return ret, ret_labels

    def compute_ss_loss(self, images, scale):
        """
        Function to compute SS loss.
        """
        # Rotate images and produce labels here.
        images_rot, class_labels = self._rotate_batch(images=images)

        # Compute SS loss
        _, output_classes = self.forward(images_rot)

        err_SS = F.cross_entropy(input=output_classes, target=class_labels)

        # Scale SS loss
        err_SS = scale * err_SS

        return err_SS

    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        """
        Train step function for discirminator.
        """
        self.zero_grad()

        # Produce real images
        real_images, _ = real_batch
        batch_size = real_images.shape[0]  # Match batch sizes for last iter

        # Produce fake images
        fake_images = netG.generate_images(num_images=batch_size,
                                           device=device).detach()

        # Compute real and fake logits for gan loss
        output_real, _ = self.forward(real_images)
        output_fake, _ = self.forward(fake_images)

        # Compute GAN loss, upright images only.
        errD = self.compute_gan_loss(output_real=output_real,
                                     output_fake=output_fake)

        # Compute SS loss, rotates the images. -- only for real images!
        errD_SS = self.compute_ss_loss(images=real_images,
                                       scale=self.ss_loss_scale)

        # Backprop and update gradients
        errD_total = errD + errD_SS
        errD_total.backward()
        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=output_real,
                                       output_fake=output_fake)

        # Log statistics for D once out of loop
        log_data.add_metric('errD', errD, group='loss')
        log_data.add_metric('errD_SS', errD_SS, group='loss_SS')
        log_data.add_metric('D(x)', D_x, group='prob')
        log_data.add_metric('D(G(z))', D_Gz, group='prob')

        return log_data


#########################
#        Training
#########################
# Directories
dataset_dir = './datasets'
log_dir = './log/ssgan'

# Data handling objects
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
dataset = mmc.datasets.load_dataset(root='./datasets', name='cifar10')
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=64,
                                         shuffle=True,
                                         num_workers=4)

# Define models and optimizers
netG = SSGANGenerator().to(device)
netD = SSGANDiscriminator().to(device)
optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

# Start training
trainer = mmc.training.Trainer(netD=netD,
                               netG=netG,
                               optD=optD,
                               optG=optG,
                               n_dis=2,
                               num_steps=100000,
                               dataloader=dataloader,
                               log_dir=log_dir,
                               device=device)
trainer.train()

##########################
#       Evaluation
##########################
# Evaluate fid
mmc.metrics.evaluate(metric='fid',
                     log_dir=log_dir,
                     netG=netG,
                     dataset='cifar10',
                     num_real_samples=10000,
                     num_fake_samples=10000,
                     evaluate_step=100000,
                     device=device)
