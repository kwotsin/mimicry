import torch.nn as nn
import torch.optim as optim

from torch_mimicry.training import scheduler, metric_log


class TestLRScheduler:
    def setup(self):
        self.netD = nn.Linear(10, 10)
        self.netG = nn.Linear(10, 10)

        self.num_steps = 10
        self.lr_D = 2e-4
        self.lr_G = 2e-4

    def get_lr(self, optimizer):
        return optimizer.param_groups[0]['lr']

    def test_linear_decay(self):
        optD = optim.Adam(self.netD.parameters(), self.lr_D, betas=(0.0, 0.9))
        optG = optim.Adam(self.netG.parameters(), self.lr_G, betas=(0.0, 0.9))

        lr_scheduler = scheduler.LRScheduler(lr_decay='linear',
                                             optD=optD,
                                             optG=optG,
                                             num_steps=self.num_steps)

        log_data = metric_log.MetricLog()
        for step in range(1, self.num_steps + 1):
            lr_scheduler.step(log_data, step)

            curr_lr = ((1 - step / self.num_steps) * self.lr_D)

            assert (curr_lr - self.get_lr(optD) < 1e-5)
            assert (curr_lr - self.get_lr(optG) < 1e-5)

    def test_no_decay(self):
        optD = optim.Adam(self.netD.parameters(), self.lr_D, betas=(0.0, 0.9))
        optG = optim.Adam(self.netG.parameters(), self.lr_G, betas=(0.0, 0.9))

        lr_scheduler = scheduler.LRScheduler(lr_decay='None',
                                             optD=optD,
                                             optG=optG,
                                             num_steps=self.num_steps)

        log_data = metric_log.MetricLog()
        for step in range(1, self.num_steps + 1):
            lr_scheduler.step(log_data, step)

            assert (self.lr_D == self.get_lr(optD))
            assert (self.lr_G == self.get_lr(optG))


if __name__ == "__main__":
    test = TestLRScheduler()
    test.setup()
    test.test_linear_decay()
    test.test_no_decay()
