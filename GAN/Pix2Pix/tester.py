import torch

import imageio
from models.pix2pix import Generator

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester(object):
    def __init__(self, config, netG_path):
        self.config = config
        self.netG_path = netG_path
        self.load_net()

    def load_net(self):
        netG = Generator(self.config.inch, self.config.outch, self.config.ngf)
        netG.load_state_dict(torch.load(self.netG_path))
        self.netG = netG.to(device)

    def test(self):
        pass