import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from model.cdcgan import Generator, Discriminator
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
        # print(self.g)
        # print(self.d)

    # network init
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


    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    # training method
    def train(self):
        # setup loss function, optimizers
        criterion = nn.BCELoss()
        g_opt = optim.Adam(self.g.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        d_opt = optim.Adam(self.d.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        # prepare label
        onehot = torch.zeros(10, 10)
        onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1).view(10, 10, 1, 1)
        fill = torch.zeros([10, 10, self.image_size, self.image_size])
        for i in range(10):
            fill[i, i, :, :] = 1  # shape: 10, 10, image_size, image_size

        # setup fixed noise, label
        fixed_noise = torch.FloatTensor(50, self.nz).normal_(0, 1).view(-1, self.nz, 1, 1)
        fixed_label = (torch.rand(50, 1) * self.n_classes).long().to(device).squeeze()

        fixed_onehot = onehot[fixed_label]
        fixed_onehot = fixed_onehot.to(device)

        print("Learning started!!")

        for epoch in range(self.n_epochs):
            # learning rate decay
            if (epoch+1) == 30:
                g_opt.param_groups[0]['lr'] /= 5
                d_opt.param_groups[0]['lr'] /= 5
                print("Learning rate change!")

            if (epoch+1) == 40:
                g_opt.param_groups[0]['lr'] /= 5
                d_opt.param_groups[0]['lr'] /= 5
                print("Learning rate change!")

            for step, (x_real, y_) in enumerate(self.dataloader):
                # ================================================================== #
                #                      Train the discriminator                       #
                # ================================================================== #
                for p in self.d.parameters():
                    p.requires_grad = True

                step_batch = x_real.size(0)

                target_real = torch.ones(step_batch).to(device)
                target_fake = torch.zeros(step_batch).to(device)

                x_real = x_real.to(device)
                y_ = y_.to(device)
                y_fill = fill[y_].to(device)

                # compute output and loss with real data
                D_out_from_real = self.d(x_real, y_fill)
                D_x = D_out_from_real.data.mean()

                D_loss_from_real = criterion(D_out_from_real, target_real)

                z_ = torch.randn((step_batch, self.nz)).view(-1, self.nz, 1, 1).to(device)
                y_ = (torch.rand(step_batch, 1) * self.n_classes).long().to(device).squeeze()  # batch

                y_label = onehot[y_] # batch, 10, 1, 1
                y_fill = fill[y_]  # batch, 10, 32, 32

                x_fake = self.g(z_, y_label)
                x_fake = x_fake.to(device)

                # compute output and loss with fake data
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

                y_label = onehot[y_]  # batch, 10, 1, 1
                y_fill = fill[y_]     # batch, 10, 32, 32

                x_fake = self.g(z_, y_label).to(device)
                D_out_from_fake = self.d(x_fake, y_fill)
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

                    fake = self.g(fixed_noise, fixed_onehot)
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

