import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import pickle
import os

from model.cdcgan import Generator, Discriminator
from vis_tool import Visualizer

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer(object):
    # initializer
    def __init__(self, config, dataloader):
        self.config = config

        # prepare input, label
        self.dataloader = dataloader
        with open('../../data/resized_celebA/gender_label.pkl', 'rb') as fp:
            gender_label = pickle.load(fp)

        self.gender_label = torch.LongTensor(gender_label).squeeze()

        self.nz = config.nz
        self.ngf = config.ngf
        self.ndf = config.ndf
        self.nch = config.nch
        self.n_classes = config.n_classes

        self.batch_size = config.batch_size
        self.image_size = config.image_size

        self.lr = config.lr
        self.beta1 = config.beta1

        self.n_epochs = config.n_epochs
        self.out_folder = config.out_folder

        self.vis = Visualizer()

        self.build_net()
        self.setup_label()
        self.setup_fix_vectors()

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    # build network
    def build_net(self):
        self.g = Generator(self.nz, self.ngf, self.nch, self.n_classes)
        self.g.weight_init(mean=0, std=0.02)

        # if trained weight exists
        if self.config.g != '':
            self.g.load_state_dict(torch.load(self.config.g))

        self.g = self.g.to(device)

        self.d = Discriminator(self.ndf, self.nch, self.n_classes)
        self.d.weight_init(mean=0, std=0.02)

        # if trained weight exists
        if self.config.d != '':
            self.d.load_state_dict(torch.load(self.config.d))

        self.d = self.d.to(device)

    # prepare label
    def setup_label(self):
        onehot = torch.zeros(self.n_classes, self.n_classes)
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2, 1, 1)
        fill = torch.zeros([2, 2, self.image_size, self.image_size])

        for i in range(2):
            fill[i, i, :, :] = 1

        self.onehot = onehot
        self.fill = fill

    # setup fixed noise vectors
    def setup_fix_vectors(self):
        temp_z0_ = torch.randn(4, 100)
        temp_z0_ = torch.cat([temp_z0_, temp_z0_], 0)
        temp_z1_ = torch.randn(4, 100)
        temp_z1_ = torch.cat([temp_z1_, temp_z1_], 0)

        fixed_z_ = torch.cat([temp_z0_, temp_z1_], 0)
        fixed_y_ = torch.cat([torch.zeros(4), torch.ones(4), torch.zeros(4), torch.ones(4)], 0).type(
            torch.LongTensor).squeeze()

        fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
        fixed_y_label_ = self.onehot[fixed_y_]

        self.fixed_z = fixed_z_.to(device)
        self.fixed_y_label = fixed_y_label_.to(device)

    # save sample fake results
    def save_sample_results(self, epoch, path, show=False, save=True):
        self.g.eval()
        fake_image = self.g(self.fixed_z, self.fixed_y_label)
        self.g.train()

        size_figure_grid = 4
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

        for k in range(size_figure_grid * size_figure_grid):
            i = k // size_figure_grid
            j = k % size_figure_grid
            ax[i, j].cla()
            ax[i, j].imshow((fake_image[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

        label = 'Epoch {0}'.format(epoch)
        fig.text(0.5, 0.04, label, ha='center')

        if save:
            plt.savefig(path)

        if show:
            plt.show()
        else:
            plt.close()

    # training method
    def train(self):
        # Binary Cross Entropy loss
        criterion = nn.BCELoss()

        # Adam optimizer
        g_opt = optim.Adam(self.g.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        d_opt = optim.Adam(self.d.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        print("Learning started!!")

        for epoch in range(self.n_epochs):
            # learning rate decay
            if (epoch + 1) == 11:
                g_opt.param_groups[0]['lr'] /= 5
                d_opt.param_groups[0]['lr'] /= 5
                print("Learning rate change!")

            if (epoch + 1) == 16:
                g_opt.param_groups[0]['lr'] /= 5
                d_opt.param_groups[0]['lr'] /= 5
                print("Learning rate change!")

            for step, (x_real, _) in enumerate(self.dataloader):
                # ================================================================== #
                #                      Train the discriminator                       #
                # ================================================================== #
                for p in self.d.parameters():
                    p.requires_grad = True

                step_batch = x_real.size(0)

                target_real = torch.ones(step_batch).to(device)
                target_fake = torch.zeros(step_batch).to(device)

                if step_batch != self.batch_size:
                    y_ = self.gender_label[self.batch_size*step:]
                else:
                    y_ = self.gender_label[self.batch_size*step: self.batch_size*(step+1)]

                x_real = x_real.to(device)
                y_ = y_.to(device)
                y_fill = self.fill[y_].to(device)

                # compute output and loss with real data
                D_out_from_real = self.d(x_real, y_fill)
                D_x = D_out_from_real.data.mean()
                D_loss_from_real = criterion(D_out_from_real, target_real)

                z_ = torch.randn((step_batch, self.nz)).view(-1, self.nz, 1, 1).to(device)
                y_ = (torch.rand(step_batch, 1) * self.n_classes).long().to(device).squeeze()  # batch

                y_label = self.onehot[y_].to(device)  # batch, 10, 1, 1
                y_fill = self.fill[y_].to(device)  # batch, 10, 32, 32

                x_fake = self.g(z_, y_label)
                x_fake = x_fake.to(device)

                D_out_from_fake = self.d(x_fake, y_fill)
                D_G_z1 = D_out_from_fake.data.mean()
                D_loss_from_fake = criterion(D_out_from_fake, target_fake)

                D_loss = D_loss_from_real + D_loss_from_fake

                # reset + forward + backward
                self.d.zero_grad()
                D_loss.backward()
                d_opt.step()

                # ================================================================== #
                #                      Train the Generator                           #
                # ================================================================== #
                for p in self.d.parameters():
                    p.requires_grad = False

                z_ = torch.randn((step_batch, self.nz)).view(-1, self.nz, 1, 1).to(device)
                y_ = (torch.rand(step_batch, 1) * self.n_classes).long().to(device).squeeze()  # batch

                y_label = self.onehot[y_].to(device)  # batch, 10, 1, 1
                y_fill = self.fill[y_].to(device)  # batch, 10, 32, 32

                x_fake = self.g(z_, y_label)
                x_fake = x_fake.to(device)
                D_out_from_fake = self.d(x_fake, y_fill)
                D_G_z2 = D_out_from_fake.data.mean()

                G_loss = criterion(D_out_from_fake, target_real)

                # reset + forward + backward
                self.g.zero_grad()
                G_loss.backward()
                g_opt.step()

                if step % 10 == 0:
                    # do logging
                    print("[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
                          % (epoch + 1, self.n_epochs, step + 1, len(self.dataloader),
                             D_loss.item(), G_loss.item(), D_x, D_G_z1, D_G_z2))

                    # plot to visdom
                    self.vis.plot("D loss per 100 steps", D_loss.item())
                    self.vis.plot("G loss per 100 steps", G_loss.item())

                    # save results
                    x_real = x_real.view(-1, self.nch, self.image_size, self.image_size)
                    vutils.save_image(self.denorm(x_real.data),
                                      '%s/real_samples.png' % self.out_folder)

                    self.save_sample_results(epoch, '%s/epoch%02d.png' % (self.out_folder, epoch))

                    print("Save result!!")

            if epoch != 0 and epoch % 5 == 0:
                os.makedirs('%s/weights' % self.out_folder, exist_ok=True)

                # save checkpoints
                torch.save(self.g.state_dict(), '%s/weights/G_epoch_%03d.pth' % (self.out_folder, epoch))
                torch.save(self.d.state_dict(), '%s/weights/D_epoch_%03d.pth' % (self.out_folder, epoch))

        print("learning finished!")
        torch.save(self.g.state_dict(), '%s/weights/G_final_%03d.pth' % (self.out_folder, epoch))
        torch.save(self.d.state_dict(), '%s/weights/D_final_%03d.pth' % (self.out_folder, epoch))
        print("save checkpoint finished!")