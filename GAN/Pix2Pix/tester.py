import os
import torch
from torch.autograd import Variable

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

import torchvision.utils as vutils

import imageio
import random

from models.pix2pix import Generator
from config import get_config
from dataloader import get_loader

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester(object):
    def __init__(self, config, val_dataloader, netG_path):
        self.config = config
        self.val_dataloader = val_dataloader
        self.netG_path = netG_path
        self.load_net()

    def load_net(self):
        netG = Generator(self.config.inch, self.config.outch, self.config.ngf)
        netG.load_state_dict(torch.load(self.netG_path))
        self.netG = netG.to(device)
        self.netG.eval() # switch to eval mode

        # check train mode of model
        print("self.netG.training:", self.netG.training)

    def generate_result(self):
        # prepare test input, target image
        val_iter = iter(self.val_dataloader)
        data_val = val_iter.next()

        if self.config.mode == "B2A":
            val_target, val_input = data_val
        elif self.config.mode == "A2B":
            val_input, val_target = data_val

        val_input, val_target = val_input.to(device), val_target.to(device)

        os.makedirs("test", exist_ok=True)
        vutils.save_image(val_input, "test\\test_input.png", normalize=True)
        vutils.save_image(val_target, "test\\test_target.png", normalize=True)

        test_output = torch.zeros(val_input.size())
        for idx in range(val_input.size(0)):
            img = val_input[idx, :, :, :].unsqueeze(0)
            input_img = Variable(img, volatile=True).to(device)
            x_hat_test = self.netG(input_img)
            test_output[idx, :, :, :].copy_(x_hat_test.data[0])

        vutils.save_image(test_output, "test\\test_output.png", normalize=True)

if __name__ == "__main__":
    config = get_config()

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)

    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed_all(config.manual_seed)

    cudnn.benchmark = True
    val_dataloader = get_loader(config.dataset, config.val_dataroot,
                                config.image_size, config.image_size,
                                config.val_batch_size, config.workers,
                                split='val', shuffle=False, seed=config.manual_seed)

    netG_path = "samples\\ver2\\netG_final.pth"
    tester = Tester(config, val_dataloader, netG_path)
    tester.generate_result()
