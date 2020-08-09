"""
Implementation of Base SSGAN models.
"""
import numpy as np
import torch
import torch.nn.functional as F

from torch_mimicry.nets.gan import gan


class SSGANBaseGenerator(gan.BaseGenerator):
    r"""
    ResNet backbone generator for SSGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
        ss_loss_scale (float): Self-supervised loss scale for generator.
    """
    def __init__(self,
                 nz,
                 ngf,
                 bottom_width,
                 loss_type='hinge',
                 ss_loss_scale=0.2,
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         **kwargs)
        self.ss_loss_scale = ss_loss_scale

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (MetricLog): An object to add custom metrics for visualisations.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

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

        # Compute SS loss, rotates the images.
        errG_SS, _ = netD.compute_ss_loss(images=fake_images,
                                          scale=self.ss_loss_scale)

        # Backprop and update gradients
        errG_total = errG + errG_SS
        errG_total.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')
        log_data.add_metric('errG_SS', errG_SS, group='loss_SS')

        return log_data


class SSGANBaseDiscriminator(gan.BaseDiscriminator):
    r"""
    ResNet backbone discriminator for SSGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.        
        ss_loss_scale (float): Self-supervised loss scale for discriminator.        
    """
    def __init__(self, ndf, loss_type='hinge', ss_loss_scale=1.0, **kwargs):
        super().__init__(ndf=ndf, loss_type=loss_type, **kwargs)
        self.num_classes = 4
        self.ss_loss_scale = ss_loss_scale

    def _rot_tensor(self, image, deg):
        r"""
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
            raise ValueError(
                "Function only supports 90,180,270,0 degree rotation.")

    def _rotate_batch(self, images):
        r"""
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
        r"""
        Function to compute SS loss.

        Args:
            images (Tensor): A batch of non-rotated, upright images.
            scale (float): The parameter to scale SS loss by.

        Returns:
            Tensor: Scalar tensor representing the SS loss.
        """
        # Rotate images and produce labels here.
        images_rot, class_labels = self._rotate_batch(images=images)

        # Compute SS loss
        _, output_classes = self.forward(images_rot)

        err_SS = F.cross_entropy(input=output_classes, target=class_labels)

        # Scale SS loss
        err_SS = scale * err_SS

        return err_SS, class_labels

    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for D.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
            netG (nn.Module): Generator model for obtaining fake images.
            optD (Optimizer): Optimizer for updating discriminator's parameters.
            device (torch.device): Device to use for running the model.
            log_data (MetricLog): An object to add custom metrics for visualisations.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

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

        # Compute SS loss, rotates the images.
        errD_SS, _ = self.compute_ss_loss(images=real_images,
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
