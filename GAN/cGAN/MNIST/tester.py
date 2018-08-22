import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import itertools
import os

from model.cgan import Generator
from config import get_config

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester(object):
    # initializer
    def __init__(self, config, weight_path, out_path):
        self.config = config

        self.nz = config.nz
        self.ngf = config.ngf
        self.ndf = config.ndf

        self.image_size = config.image_size
        self.ncls = config.n_classes

        self.g_path = weight_path + "generator.pth"

        self.out_path = out_path

        self.load_net()

    # load trained network
    def load_net(self):
        self.g = Generator(self.nz, self.ngf, self.ncls, self.image_size)
        self.g.load_state_dict(torch.load(self.g_path, map_location=lambda storage, loc: storage))
        self.g = self.g.to(device)
        self.g.eval() # fix parameters

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    # generate test result images
    def test(self, nsamples):
        # fixed noise & label
        temp_z = torch.randn(nsamples, self.nz)
        fixed_z = temp_z
        fixed_y = torch.zeros(nsamples, 1)

        for i in range(self.ncls-1):
            fixed_z = torch.cat([fixed_z, temp_z], 0)
            temp_y = torch.ones(nsamples, 1) + i
            fixed_y = torch.cat([fixed_y, temp_y], 0)

        fixed_y_label = torch.zeros(nsamples*self.ncls, self.ncls)
        fixed_y_label.scatter_(1, fixed_y.type(torch.LongTensor), 1)

        # set fixed variable
        fixed_z = fixed_z.to(device)
        fixed_y_label = fixed_y_label.to(device)

        # generate result images
        result_imgs = self.g(fixed_z, fixed_y_label)
        result_imgs = result_imgs.view(-1, 1, self.image_size, self.image_size)
        result_imgs = self.denorm(result_imgs)

        # process image and save
        fig, ax = plt.subplots(self.ncls, nsamples, figsize=(5, 5))
        for i, j in itertools.product(range(self.ncls), range(nsamples)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for k in range(nsamples * self.ncls):
            i = k // nsamples
            j = k % nsamples
            ax[i, j].cla()
            ax[i, j].imshow(result_imgs[k, 0].cpu().data.numpy(), cmap='gray')

        label = 'Result image'
        fig.text(0.5, 0.04, label, ha='center')
        plt.savefig(self.out_path)

if __name__ == "__main__":
    config = get_config()

    weight_path = "samples\weights\\"

    os.makedirs('test', exist_ok=True)

    tester = Tester(config, weight_path, 'test\out.png')
    tester.test(nsamples=15)