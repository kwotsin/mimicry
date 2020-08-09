import os
import shutil

import numpy as np
import torch
from PIL import Image

from torch_mimicry.utils import common


class TestCommon:
    def setup(self):
        # Build directories
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "test_log")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def test_json_write_and_load(self):
        dict_to_write = dict(a=1, b=2, c=3)
        output_file = os.path.join(self.log_dir, 'output.json')
        common.write_to_json(dict_to_write, output_file)
        check = common.load_from_json(output_file)

        assert dict_to_write == check

    def test_load_and_save_image(self):
        image, label = common.load_images(n=1)

        image = torch.squeeze(image, dim=0)
        output_file = os.path.join(self.log_dir, 'images', 'test_img.png')
        common.save_tensor_image(image, output_file=output_file)

        check = np.array(Image.open(output_file))

        assert check.shape == (32, 32, 3)
        assert label.shape == (1, )

    def teardown(self):
        shutil.rmtree(self.log_dir)


if __name__ == "__main__":
    test = TestCommon()
    test.setup()
    test.test_json_write_and_load()
    test.test_load_and_save_image()
    test.teardown()
