"""
Loss functions definitions.
"""
import torch
import torch.nn.functional as F


def _bce_loss_with_logits(output, labels, **kwargs):
    r"""
    Wrapper for BCE loss with logits.
    """
    return F.binary_cross_entropy_with_logits(output, labels, **kwargs)


def minimax_loss_gen(output_fake, real_label_val=1.0, **kwargs):
    r"""
    Standard minimax loss for GANs through the BCE Loss with logits fn.

    Args:
        output (Tensor): Discriminator output logits.
        labels (Tensor): Labels for computing cross entropy.

    Returns:
        Tensor: A scalar tensor loss output.      
    """
    # Produce real labels so G is rewarded if D is fooled
    real_labels = torch.full((output_fake.shape[0], 1),
                             real_label_val,
                             device=output_fake.device)

    loss = _bce_loss_with_logits(output_fake, real_labels, **kwargs)

    return loss


def minimax_loss_dis(output_fake,
                     output_real,
                     real_label_val=1.0,
                     fake_label_val=0.0,
                     **kwargs):
    r"""
    Standard minimax loss for GANs through the BCE Loss with logits fn.

    Args:
        output_fake (Tensor): Discriminator output logits for fake images.    
        output_real (Tensor): Discriminator output logits for real images.
        real_label_val (int): Label for real images.
        fake_label_val (int): Label for fake images.
        device (torch.device): Torch device object for sending created data.

    Returns:
        Tensor: A scalar tensor loss output.      
    """
    # Produce real and fake labels.
    fake_labels = torch.full((output_fake.shape[0], 1),
                             fake_label_val,
                             device=output_fake.device)
    real_labels = torch.full((output_real.shape[0], 1),
                             real_label_val,
                             device=output_real.device)

    # FF, compute loss and backprop D
    errD_fake = _bce_loss_with_logits(output=output_fake,
                                      labels=fake_labels,
                                      **kwargs)

    errD_real = _bce_loss_with_logits(output=output_real,
                                      labels=real_labels,
                                      **kwargs)

    # Compute cumulative error
    loss = errD_real + errD_fake

    return loss


def ns_loss_gen(output_fake):
    r"""
    Non-saturating loss for generator.

    Args:
        output_fake (Tensor): Discriminator output logits for fake images.

    Returns:
        Tensor: A scalar tensor loss output.    
    """
    output_fake = torch.sigmoid(output_fake)

    return -torch.mean(torch.log(output_fake + 1e-8))


def wasserstein_loss_dis(output_real, output_fake):
    r"""
    Computes the wasserstein loss for the discriminator.

    Args:
        output_real (Tensor): Discriminator output logits for real images.
        output_fake (Tensor): Discriminator output logits for fake images.

    Returns:
        Tensor: A scalar tensor loss output.        
    """
    loss = -1.0 * output_real.mean() + output_fake.mean()

    return loss


def wasserstein_loss_gen(output_fake):
    r"""
    Computes the wasserstein loss for generator.

    Args:
        output_fake (Tensor): Discriminator output logits for fake images.

    Returns:
        Tensor: A scalar tensor loss output.
    """
    loss = -output_fake.mean()

    return loss


def hinge_loss_dis(output_fake, output_real):
    r"""
    Hinge loss for discriminator.

    Args:
        output_fake (Tensor): Discriminator output logits for fake images.
        output_real (Tensor): Discriminator output logits for real images.

    Returns:
        Tensor: A scalar tensor loss output.        
    """
    loss = F.relu(1.0 - output_real).mean() + \
           F.relu(1.0 + output_fake).mean()

    return loss


def hinge_loss_gen(output_fake):
    r"""
    Hinge loss for generator.

    Args:
        output_fake (Tensor): Discriminator output logits for fake images.

    Returns:
        Tensor: A scalar tensor loss output.      
    """
    loss = -output_fake.mean()

    return loss
