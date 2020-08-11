import os
import shutil

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from torch_mimicry.nets.gan import gan
from torch_mimicry.training.trainer import Trainer
from torch_mimicry.utils import common


class ExampleDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.data = np.random.randn(10, 32, 32, 3).astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx] * np.random.randn()
        target = -1

        return img, target


class ExampleGen(gan.BaseGenerator):
    def __init__(self,
                 bottom_width=4,
                 nz=4,
                 ngf=16,
                 loss_type='gan',
                 *args,
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         *args,
                         **kwargs)
        self.linear = nn.Linear(self.nz, 3072)

    def forward(self, x):
        output = self.linear(x)
        output = output.view(x.shape[0], 3, 32, 32)

        return output


class ExampleDis(gan.BaseDiscriminator):
    def __init__(self, ndf=16, loss_type='gan', *args, **kwargs):
        super().__init__(ndf=ndf, loss_type=loss_type, *args, **kwargs)
        self.linear = nn.Linear(3072, 1)

    def forward(self, x):
        output = x.view(x.shape[0], -1)
        output = self.linear(output)

        return output


class TestTrainer:
    def setup(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        self.dataset = ExampleDataset()
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                      batch_size=50,
                                                      shuffle=True,
                                                      num_workers=8)

        # Define models
        self.netG = ExampleGen()
        self.netD = ExampleDis()

        # Build optimizers
        self.optD = optim.Adam(self.netD.parameters(), 2e-4, betas=(0.0, 0.9))
        self.optG = optim.Adam(self.netG.parameters(), 2e-4, betas=(0.0, 0.9))

        # Build directories
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "test_log")

        # Build test object
        self.trainer = Trainer(netD=self.netD,
                               netG=self.netG,
                               optD=self.optD,
                               optG=self.optG,
                               log_dir=self.log_dir,
                               dataloader=self.dataloader,
                               num_steps=2,
                               device='cpu',
                               save_steps=1,
                               log_steps=1,
                               vis_steps=1,
                               lr_decay='linear')

    def test_attributes(self):
        # Check parameters
        default_params = {
            'log_dir': self.log_dir,
            'n_dis': 1,
            'num_steps': 2,
            'batch_size': 50,
            'lr_decay': 'linear',
            'optD': self.optD.__repr__(),
            'optG': self.optG.__repr__(),
            'save_steps': 1,
        }
        for name, param in default_params.items():
            print(name, self.trainer.params[name], param)
            assert self.trainer.params[name] == param

        # Test optional parameters not covered
        netG_ckpt_file = os.path.join(self.log_dir, 'netG.pth')
        netD_ckpt_file = os.path.join(self.log_dir, 'netD.pth')
        device = None

        extra_trainer = Trainer(netD=self.netD,
                                netG=self.netG,
                                optD=self.optD,
                                optG=self.optG,
                                netG_ckpt_file=netG_ckpt_file,
                                netD_ckpt_file=netD_ckpt_file,
                                log_dir=os.path.join(self.log_dir, 'extra'),
                                dataloader=self.dataloader,
                                num_steps=2,
                                device=device,
                                save_steps=float('inf'),
                                log_steps=float('inf'),
                                vis_steps=float('inf'),
                                lr_decay='linear')

        assert extra_trainer.netG_ckpt_file == netG_ckpt_file
        assert extra_trainer.netG_ckpt_file == netG_ckpt_file
        assert extra_trainer.device == torch.device(
            'cuda:0' if torch.cuda.is_available() else "cpu")

        with pytest.raises(ValueError):
            bad_trainer = Trainer(netD=self.netD,
                                  netG=self.netG,
                                  optD=self.optD,
                                  optG=self.optG,
                                  netG_ckpt_file=netG_ckpt_file,
                                  netD_ckpt_file=netD_ckpt_file,
                                  log_dir=os.path.join(self.log_dir, 'extra'),
                                  dataloader=self.dataloader,
                                  num_steps=-1000,
                                  device=device,
                                  save_steps=float('inf'),
                                  log_steps=float('inf'),
                                  vis_steps=float('inf'),
                                  lr_decay='linear')

    def test_get_latest_checkpoint(self):
        ckpt_files = [
            'netG_1000_steps.pth', 'netG_10000_steps.pth', 'netG_*.pth',
            'asdasd.pth'
        ]

        test_dir = os.path.join(self.log_dir, 'test_checkpoint')

        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        for file in ckpt_files:
            with open(os.path.join(test_dir, file), 'w') as f:
                pass

        chosen_file = self.trainer._get_latest_checkpoint(test_dir)

        assert os.path.basename(chosen_file) == "netG_10000_steps.pth"

        # Case where no checkpoint files:
        empty_dir = os.path.join(self.log_dir, 'empty')
        if not os.path.exists(empty_dir):
            os.makedirs(empty_dir)

        assert self.trainer._get_latest_checkpoint(empty_dir) is None

    def test_fetch_data(self):
        iter_dataloader = iter(self.dataloader)

        batches = []
        for i in range(2):
            iter_dataloader, real_batch = self.trainer._fetch_data(
                iter_dataloader=iter_dataloader)
            batches.append(real_batch)

        # Test different images and labels at each fetch.
        for i in range(1, len(batches)):
            images_1, labels_1 = batches[i - 1]
            images_2, labels_2 = batches[i]

            assert torch.sum((images_1 - images_2)**2) > 1

    def test_train(self):
        def get_params_sum(net):
            total = 0
            for param in net.parameters():
                total = total + torch.sum(param)

            return total

        def get_lr(optimizer):
            return optimizer.param_groups[0]['lr']

        # Test parameters updated
        netD_params_before = get_params_sum(self.trainer.netD)
        netG_params_before = get_params_sum(self.trainer.netG)

        self.trainer.train()

        netD_params_after = get_params_sum(self.trainer.netD)
        netG_params_after = get_params_sum(self.trainer.netG)

        assert abs(netD_params_before.item() - netD_params_after.item()) > 0
        assert abs(netG_params_before.item() - netG_params_after.item()) > 0

        # Test LR
        assert get_lr(self.trainer.optD) == get_lr(self.trainer.scheduler.optD)
        assert get_lr(self.trainer.optG) == get_lr(self.trainer.scheduler.optG)

        # Test saving functions
        netG_files = os.listdir(self.trainer.netG_ckpt_dir)
        netD_files = os.listdir(self.trainer.netD_ckpt_dir)
        assert set(netD_files) == {'netD_2_steps.pth', 'netD_1_steps.pth'}
        assert set(netG_files) == {'netG_2_steps.pth', 'netG_1_steps.pth'}

        # Test tensorboard writing summaries
        tb_dir = os.path.join(self.log_dir, 'data')
        for root, dirs, files in os.walk(tb_dir):
            for file in files:
                assert file.startswith('events')

        # Test visualisations
        img_dir = os.path.join(self.log_dir, 'images')
        img_files = os.listdir(img_dir)
        check = set([
            'fake_samples_step_1.png', 'fixed_fake_samples_step_2.png',
            'fake_samples_step_2.png', 'fixed_fake_samples_step_1.png'
        ])
        assert set(img_files) == check

    def test_log_params(self):
        new_params = {
            'n_dis': 1,
            'lr_decay': 'None',
            'print_steps': 1,
            'vis_steps': 500,
            'flush_secs': 30,
            'log_steps': 50,
            'save_steps': 5000,
        }

        with pytest.raises(ValueError):
            self.trainer._log_params(new_params)

        params_file = os.path.join(self.log_dir, 'params.json')
        assert os.path.exists(params_file)

        params_check = common.load_from_json(params_file)
        assert self.trainer.params == params_check

    def test_restore_models_and_step(self):
        # Test mismatched steps
        self.netG.save_checkpoint(directory=self.trainer.netG_ckpt_dir,
                                  global_step=99,
                                  optimizer=self.optG)
        self.netD.save_checkpoint(directory=self.trainer.netD_ckpt_dir,
                                  global_step=999,
                                  optimizer=self.optD)

        # Cache
        tmp_netG_ckpt_file = self.trainer.netG_ckpt_file
        tmp_netD_ckpt_file = self.trainer.netD_ckpt_file
        self.trainer.netG_ckpt_file = os.path.join(self.trainer.netG_ckpt_dir,
                                                   'netG_99_steps.pth')
        self.trainer.netD_ckpt_file = os.path.join(self.trainer.netD_ckpt_dir,
                                                   'netD_999_steps.pth')

        with pytest.raises(ValueError):
            self.trainer._restore_models_and_step()

        # Remove and restore
        os.remove(self.trainer.netG_ckpt_file)
        os.remove(self.trainer.netD_ckpt_file)
        self.trainer.netG_ckpt_file = tmp_netG_ckpt_file
        self.trainer.netD_ckpt_file = tmp_netD_ckpt_file

    def teardown(self):
        shutil.rmtree(self.log_dir)

        del self.dataset,
        del self.dataloader,
        del self.netG,
        del self.netD,
        del self.optG,
        del self.optD,
        del self.trainer


if __name__ == "__main__":
    test = TestTrainer()
    test.setup()
    test.test_attributes()
    test.test_get_latest_checkpoint()
    test.test_restore_models_and_step()
    test.test_fetch_data()
    test.test_train()
    test.test_log_params()
    test.teardown()
