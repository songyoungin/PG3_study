import torch

import matplotlib.pyplot as plt
import itertools
import os
import imageio

from model.cdcgan import Generator
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
        self.nch = config.nch
        self.ncls = config.n_classes

        self.g_path = weight_path + "generator.pth"

        self.out_path = out_path

        self.load_net()

    # load trained network
    def load_net(self):
        self.g = Generator(self.nz, self.ngf, self.nch, self.ncls)
        self.g.load_state_dict(torch.load(self.g_path, map_location=lambda storage, loc: storage))
        self.g = self.g.to(device)
        self.g.eval() # fix parameters

    # generate test result images
    def test(self):
        # prepare one-hot label
        onehot = torch.zeros(self.ncls, self.ncls)
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2, 1, 1)
        self.onehot = onehot

        # fixed noise & label
        temp_z0_ = torch.randn(4, 100)
        temp_z0_ = torch.cat([temp_z0_, temp_z0_], 0)
        temp_z1_ = torch.randn(4, 100)
        temp_z1_ = torch.cat([temp_z1_, temp_z1_], 0)

        fixed_z_ = torch.cat([temp_z0_, temp_z1_], 0)
        fixed_y_ = torch.cat([torch.zeros(4), torch.ones(4), torch.zeros(4), torch.ones(4)], 0).type(
            torch.LongTensor).squeeze()

        fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
        fixed_y_label_ = self.onehot[fixed_y_]

        fixed_z = fixed_z_.to(device)
        fixed_y_label = fixed_y_label_.to(device)

        fixed_z = fixed_z.to(device)
        fixed_y_label = fixed_y_label.to(device)

        # generate result images
        result_img = self.g(fixed_z, fixed_y_label)

        size_figure_grid = 4
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for k in range(size_figure_grid * size_figure_grid):
            i = k // size_figure_grid
            j = k % size_figure_grid
            ax[i, j].cla()
            ax[i, j].imshow((result_img[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

        label = 'Result image'
        fig.text(0.5, 0.04, label, ha='center')

        plt.savefig(self.out_path)
        plt.close()

    # generate GIF
    def generate_gif(self, sample_path):
        images = []
        for idx in range(18):
            image_name = "samples/epoch%02d.png" % idx
            images.append(imageio.imread(image_name))
        imageio.mimsave("samples/results.gif", images, fps=5)


if __name__ == "__main__":
    config = get_config()

    weight_path = "D:\Deep_learning\Weights\PG3-study\GAN\Cond. DCGAN\CelebA\\"

    os.makedirs('test', exist_ok=True)

    tester = Tester(config, weight_path, "test/out.png")
    tester.test()
    tester.generate_gif(sample_path="samples")