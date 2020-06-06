"""
Implementation of InfoMax-GAN base model.
"""
import torch
import torch.nn.functional as F

from torch_mimicry.nets.gan import gan


class InfoMaxGANBaseGenerator(gan.BaseGenerator):
    r"""
    ResNet backbone generator for InfoMax-GAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
        infomax_loss_scale (float): The alpha parameter used for scaling the generator infomax loss.
    """
    def __init__(self,
                 nz,
                 ngf,
                 bottom_width,
                 loss_type='hinge',
                 infomax_loss_scale=0.2,
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         **kwargs)
        self.infomax_loss_scale = infomax_loss_scale

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
        # Zero gradient every step.
        self.zero_grad()

        # Get only batch size from real batch
        real_images, _ = real_batch
        batch_size = real_images.shape[0]

        # Produce fake images
        fake_images = self.generate_images(num_images=batch_size,
                                           device=device)

        # Get logits and projected features
        output_fake, local_feat_fake, global_feat_fake = netD(fake_images)

        local_feat_fake, global_feat_fake = netD.project_features(
            local_feat=local_feat_fake, global_feat=global_feat_fake)

        # Compute losses
        errG = self.compute_gan_loss(output_fake)

        errG_IM = netD.compute_infomax_loss(local_feat=local_feat_fake,
                                            global_feat=global_feat_fake,
                                            scale=self.infomax_loss_scale)

        # Backprop and update gradients
        errG_total = errG + errG_IM

        errG_total.backward()
        optG.step()

        # Log statistics
        log_data.add_metric('errG', errG, group='loss')
        log_data.add_metric('errG_IM', errG_IM, group='loss_IM')

        return log_data


class BaseDiscriminator(gan.BaseDiscriminator):
    r"""
    ResNet backbone discriminator for SNGAN-Infomax.

    Attributes:
        nrkhs (int): The RKHS dimension R to project the local and global features to.
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.
        infomax_loss_scale (float): The beta parameter used for scaling the discriminator infomax loss.
    """
    def __init__(self,
                 nrkhs,
                 ndf,
                 loss_type='hinge',
                 infomax_loss_scale=0.2,
                 **kwargs):
        super().__init__(ndf=ndf, loss_type=loss_type, **kwargs)
        self.nrkhs = nrkhs
        self.infomax_loss_scale = infomax_loss_scale

    def _project_local(self, local_feat):
        r"""
        Helper function for projecting local features to RKHS.
        """
        local_feat_sc = self.local_nrkhs_sc(local_feat)

        local_feat = self.local_nrkhs_a(local_feat)
        local_feat = self.activation(local_feat)
        local_feat = self.local_nrkhs_b(local_feat)
        local_feat += local_feat_sc

        return local_feat

    def _project_global(self, global_feat):
        r"""
        Helper function for projecting global features to RKHS.
        """
        global_feat_sc = self.global_nrkhs_sc(global_feat)

        global_feat = self.global_nrkhs_a(global_feat)
        global_feat = self.activation(global_feat)
        global_feat = self.global_nrkhs_b(global_feat)
        global_feat += global_feat_sc

        return global_feat

    def project_features(self, local_feat, global_feat):
        r"""
        Projects local and global features.
        """
        local_feat = self._project_local(
            local_feat)  # (N, C, H, W) --> (N, nrkhs, H, W)
        global_feat = self._project_global(
            global_feat)  # (N, C) --> (N, nrkhs)

        return local_feat, global_feat

    def infonce_loss(self, l, m):
        r"""
        InfoNCE loss for local and global feature maps as used in DIM: 
        https://github.com/rdevon/DIM/blob/master/cortex_DIM/functions/dim_losses.py

        Args:
            l (Tensor): Local feature map of shape (N, ndf, H*W).
            m (Tensor): Global feature vector of shape (N, ndf, 1).
        Returns:
            Tensor: Scalar loss Tensor.
        """
        N, units, n_locals = l.size()
        _, _, n_multis = m.size()

        # First we make the input tensors the right shape.
        l_p = l.permute(0, 2, 1)
        m_p = m.permute(0, 2, 1)

        l_n = l_p.reshape(-1, units)
        m_n = m_p.reshape(-1, units)

        # Inner product for positive samples. Outer product for negative. We need to do it this way
        # for the multiclass loss. For the outer product, we want a N x N x n_local x n_multi tensor.
        u_p = torch.matmul(l_p, m).unsqueeze(2)
        u_n = torch.mm(m_n, l_n.t())
        u_n = u_n.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

        # We need to mask the diagonal part of the negative tensor.
        mask = torch.eye(N)[:, :, None, None].to(l.device)
        n_mask = 1 - mask

        # Masking is done by shifting the diagonal before exp.
        u_n = (n_mask * u_n) - (10. * (1 - n_mask))  # mask out "self" examples
        u_n = u_n.reshape(N, N * n_locals, n_multis).unsqueeze(dim=1).expand(
            -1, n_locals, -1, -1)

        # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
        pred_lgt = torch.cat(
            [u_p, u_n], dim=2
        )  # So the first of each "row" is positive, and we have N+1 elements
        pred_log = F.log_softmax(pred_lgt, dim=2)

        # The positive score is the first element of the log softmax.
        loss = -pred_log[:, :, 0].mean()

        return loss

    def compute_infomax_loss(self, local_feat, global_feat, scale):
        r"""
        Given local and global features of a real or fake image, produce the average
        dot product score between each local and global features, which is then used
        to obtain infoNCE loss.

        Args
            local_feat (Tensor): A batch of local features.
            global_feat (Tensor): A batch of global features.
            scale (float): The scaling hyperparameter for the infomax loss.

        Returns:
            Tensor: Scalar Tensor representing the scaled infomax loss.
        """
        if local_feat.shape[1] != self.nrkhs:
            raise ValueError(
                "Features have not been projected. Expected {} dim but got {} instead"
                .format(self.nrkhs, local_feat.shape[1]))

        # Prepare shapes for local and global features.
        local_feat = torch.flatten(local_feat, start_dim=2,
                                   end_dim=3)  # (N, C, H, W) --> (N, C, H*W)
        global_feat = torch.unsqueeze(global_feat, 2)  # (N, C) --> (N, C, 1)

        # Compute infomax loss and scale
        loss = self.infonce_loss(l=local_feat, m=global_feat)

        loss = scale * loss

        return loss

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
        output_real, local_feat_real, global_feat_real = self.forward(
            real_images)
        output_fake, _, _ = self.forward(fake_images)

        # Project the features
        local_feat_real, global_feat_real = self.project_features(
            local_feat=local_feat_real, global_feat=global_feat_real)

        # Compute losses
        errD = self.compute_gan_loss(output_real=output_real,
                                     output_fake=output_fake)

        errD_IM = self.compute_infomax_loss(local_feat=local_feat_real,
                                            global_feat=global_feat_real,
                                            scale=self.infomax_loss_scale)

        # Backprop and update gradients
        errD_total = errD + errD_IM
        errD_total.backward()
        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=output_real,
                                       output_fake=output_fake)

        # Log statistics for D
        log_data.add_metric('errD', errD, group='loss')
        log_data.add_metric('errD_IM', errD_IM, group='loss_IM')
        log_data.add_metric('D(x)', D_x, group='prob')
        log_data.add_metric('D(G(z))', D_Gz, group='prob')

        return log_data
