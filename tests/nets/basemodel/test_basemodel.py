import os
import shutil
import pytest

import torch
import torch.nn as nn
import torch.optim as optim

from torch_mimicry.nets.basemodel.basemodel import BaseModel


class ExampleModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(1, 4)
        nn.init.xavier_uniform_(self.linear.weight.data)

    def forward(self, x):
        return


class TestBaseModel:
    def setup(self):
        self.model = ExampleModel()
        self.opt = optim.Adam(self.model.parameters(), 2e-4, betas=(0.0, 0.9))
        self.global_step = 0

        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "test_log")

    def test_save_and_restore_checkpoint(self):
        ckpt_dir = os.path.join(self.log_dir, 'checkpoints/model')
        ckpt_file = os.path.join(ckpt_dir,
                                 "model_{}_steps.pth".format(self.global_step))

        self.model.save_checkpoint(directory=ckpt_dir,
                                   optimizer=self.opt,
                                   global_step=self.global_step)

        restored_model = ExampleModel()
        restored_opt = optim.Adam(self.model.parameters(),
                                  2e-4,
                                  betas=(0.0, 0.9))

        restored_model.restore_checkpoint(ckpt_file=ckpt_file,
                                          optimizer=self.opt)

        # Check weights are preserved
        assert all(
            (restored_model.linear.weight == self.model.linear.weight) == 1)

        with pytest.raises(ValueError):
            restored_model.restore_checkpoint(ckpt_file=None,
                                              optimizer=self.opt)

        # Check optimizers have same state dict
        assert self.opt.state_dict() == restored_opt.state_dict()

    def test_count_params(self):
        num_total_params, num_trainable_params = self.model.count_params()

        assert num_trainable_params == num_total_params == 8

    def test_get_device(self):
        assert type(self.model.device) == torch.device

    def teardown(self):
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

        del self.model
        del self.opt


if __name__ == "__main__":
    test = TestBaseModel()
    test.setup()
    test.test_save_and_restore_checkpoint()
    test.test_count_params()
    test.test_get_device()
    test.teardown()
