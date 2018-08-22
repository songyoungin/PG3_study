import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

import time

from model.cgan import Generator, Discriminator
from vis_tool import Visualizer

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer(object):
    # initializer
    def __init__(self, config, dataloader):
        self.config = config
        self.dataloader = dataloader

        self.nz = config.nz
        self.ngf = config.ngf
        self.ndf = config.ndf
        self.n_classes = config.n_classes

        self.batch_size = config.batch_size
        self.image_size = config.image_size

        self.lr = config.lr
        self.beta1 = config.beta1

        self.n_epochs = config.n_epochs
        self.out_folder = config.out_folder

        self.vis = Visualizer()

        self.build_net()

    # network init
    def build_net(self):
        self.g = Generator(self.nz, self.ngf, self.n_classes, self.image_size)
        self.g.weight_init(mean=0, std=0.02)

        # if trained weight exists
        if self.config.g != '':
            self.g.load_state_dict(torch.load(self.config.g))

        self.g = self.g.to(device)

        self.d = Discriminator(self.ndf, self.n_classes, self.image_size)
        self.d.weight_init(mean=0, std=0.02)

        # if trained weight exists
        if self.config.d != '':
            self.d.load_state_dict(torch.load(self.config.d))

        self.d = self.d.to(device)

    # transform label to onehot format
    def get_onehot(self, label):
        step_batch = label.size(0)
        label = label.long().to(device)
        oneHot = torch.zeros(step_batch, self.n_classes).to(device)
        oneHot.scatter_(1, label.view(step_batch, 1), 1)
        oneHot = oneHot.to(device)
        return oneHot

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    # training method
    def train(self):
        # setup loss function, optimizers
        criterion = nn.BCELoss()
        g_opt = optim.Adam(self.g.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        d_opt = optim.Adam(self.d.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        # setup fixed noise, label
        fixed_noise = torch.FloatTensor(50, self.nz).normal_(0, 1).to(device)
        fixed_label = (torch.rand(50, 1) * self.n_classes).long().to(device)
        fixed_label = self.get_onehot(fixed_label)

        print("Learning started!!")

        for epoch in range(self.n_epochs):
            # learning rate decay
            if (epoch+1) == 30:
                g_opt.param_groups[0]['lr'] /= 10
                d_opt.param_groups[0]['lr'] /= 10
                print("Learning rate change!")

            if (epoch+1) == 40:
                g_opt.param_groups[0]['lr'] /= 10
                d_opt.param_groups[0]['lr'] /= 10
                print("Learning rate change!")

            for step, (x_real, y_) in enumerate(self.dataloader):
                # ================================================================== #
                #                      Train the discriminator                       #
                # ================================================================== #
                for p in self.d.parameters():
                    p.requires_grad = True

                step_batch = x_real.size(0)

                x_real = x_real.view(-1, self.image_size**2).to(device)   # real data X: batch, 28*28
                y_real = self.get_onehot(y_)                              # real data Y: batch, 10

                target_real = torch.ones(step_batch).to(device)           # target for real: batch
                target_fake = torch.zeros(step_batch).to(device)          # target tor fake: batch

                # compute output and loss with real data
                D_out_from_real = self.d(x_real, y_real).squeeze()        # batch
                D_x = D_out_from_real.data.mean()

                D_loss_from_real = criterion(D_out_from_real, target_real)

                # compute output and loss with fake data
                random_z = torch.rand((step_batch, self.nz)).to(device)
                random_y = (torch.rand(step_batch, 1) * self.n_classes).long().to(device)
                random_y = self.get_onehot(random_y)

                x_fake = self.g(random_z, random_y).to(device)       # batch, 28*28
                D_out_from_fake = self.d(x_fake, random_y).squeeze()
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

                random_z = torch.rand((step_batch, self.nz)).to(device)
                random_y = (torch.rand(step_batch, 1) * self.n_classes).long().to(device)
                random_y = self.get_onehot(random_y)

                x_fake = self.g(random_z, random_y).to(device)  # batch, 28*28
                D_out_from_fake = self.d(x_fake, random_y).squeeze()
                D_G_z2 = D_out_from_fake.data.mean()

                G_loss = criterion(D_out_from_fake, target_real)

                # reset + forward + backward
                self.g.zero_grad()
                G_loss.backward()
                g_opt.step()

                if step % 100 == 0:
                    print("[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
                          % (epoch+1, self.n_epochs, step+1, len(self.dataloader),
                             D_loss.item(), G_loss.item(), D_x, D_G_z1, D_G_z2))

                    # plot to visdom
                    self.vis.plot("D loss per 100 steps", D_loss.item())
                    self.vis.plot("G loss per 100 steps", G_loss.item())

                    # save results
                    x_real = x_real.view(-1, 1, self.image_size, self.image_size)
                    x_real = self.denorm(x_real)
                    vutils.save_image(x_real,
                                      '%s/real_samples.png' % self.out_folder)

                    fake = self.g(fixed_noise, fixed_label)
                    fake = fake.view(-1, 1, self.image_size, self.image_size)
                    fake = self.denorm(fake)
                    vutils.save_image(fake,
                                      '%s/fake_samples_epoch_%03d.png' % (self.out_folder, epoch))

            if epoch % 10 == 0:
                # save checkpoints
                torch.save(self.g.state_dict(), '%s/G_epoch_%03d.pth' % (self.out_folder, epoch))
                torch.save(self.d.state_dict(), '%s/D_epoch_%03d.pth' % (self.out_folder, epoch))


        print("learning finished!")
        torch.save(self.g.state_dict(), '%s/G_final_%03d.pth' % (self.out_folder, epoch))
        torch.save(self.d.state_dict(), '%s/D_final_%03d.pth' % (self.out_folder, epoch))
        print("save checkpoint finished!")
