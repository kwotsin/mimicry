"""
Implementation of a specific learning rate scheduler for GANs.
"""


class LRScheduler:
    """
    Learning rate scheduler for training GANs. Supports GAN specific LR scheduling
    policies, such as the linear decay policy using in SN-GAN paper as based on the
    original chainer implementation. However, one could safely ignore this class
    and instead use the official PyTorch scheduler wrappers around a optimizer
    for other scheduling policies.

    Attributes:
        lr_decay (str): The learning rate decay policy to use.
        optD (Optimizer): Torch optimizer object for discriminator.
        optG (Optimizer): Torch optimizer object for generator.
        num_steps (int): The number of training iterations.
        lr_D (float): The initial learning rate of optD.
        lr_G (float): The initial learning rate of optG.
    """
    def __init__(self,
                 lr_decay,
                 optD,
                 optG,
                 num_steps,
                 start_step=0,
                 **kwargs):
        if lr_decay not in [None, 'None', 'linear']:
            raise NotImplementedError(
                "lr_decay {} is not currently supported.")

        self.lr_decay = lr_decay
        self.optD = optD
        self.optG = optG
        self.num_steps = num_steps
        self.start_step = start_step

        # Cache the initial learning rate for uses later
        self.lr_D = optD.param_groups[0]['lr']
        self.lr_G = optG.param_groups[0]['lr']

    def linear_decay(self, optimizer, global_step, lr_value_range,
                     lr_step_range):
        """
        Performs linear decay of the optimizer learning rate based on the number of global
        steps taken. Follows SNGAN's chainer implementation of linear decay, as seen in the
        chainer references:
        https://docs.chainer.org/en/stable/reference/generated/chainer.training.extensions.LinearShift.html
        https://github.com/chainer/chainer/blob/v6.2.0/chainer/training/extensions/linear_shift.py#L66

        Note: assumes that the optimizer has only one parameter group to update!

        Args:
            optimizer (Optimizer): Torch optimizer object to update learning rate.
            global_step (int): The current global step of the training.
            lr_value_range (tuple): A tuple of floats (x,y) to decrease from x to y.
            lr_step_range (tuple): A tuple of ints (i, j) to start decreasing 
                when global_step > i, and until j.

        Returns:
            float: Float representing the new updated learning rate.
        """
        # Compute the new learning rate
        v1, v2 = lr_value_range
        s1, s2 = lr_step_range

        if global_step <= s1:
            updated_lr = v1

        elif global_step >= s2:
            updated_lr = v2

        else:
            scale_factor = (global_step - s1) / (s2 - s1)
            updated_lr = v1 + scale_factor * (v2 - v1)

        # Update the learning rate
        optimizer.param_groups[0]['lr'] = updated_lr

        return updated_lr

    def step(self, log_data, global_step):
        """
        Takes a step for updating learning rate and updates the input log_data
        with the current status.

        Args:
            log_data (MetricLog): Object for logging the updated learning rate metric.
            global_step (int): The current global step of the training.

        Returns:
            MetricLog: MetricLog object containing the updated learning rate at the current global step.
        """
        if self.lr_decay == "linear":
            lr_D = self.linear_decay(optimizer=self.optD,
                                     global_step=global_step,
                                     lr_value_range=(self.lr_D, 0.0),
                                     lr_step_range=(self.start_step,
                                                    self.num_steps))

            lr_G = self.linear_decay(optimizer=self.optG,
                                     global_step=global_step,
                                     lr_value_range=(self.lr_G, 0.0),
                                     lr_step_range=(self.start_step,
                                                    self.num_steps))

        elif self.lr_decay in [None, "None"]:
            lr_D = self.lr_D
            lr_G = self.lr_G

        # Update metrics log
        log_data.add_metric('lr_D', lr_D, group='lr', precision=6)
        log_data.add_metric('lr_G', lr_G, group='lr', precision=6)

        return log_data
