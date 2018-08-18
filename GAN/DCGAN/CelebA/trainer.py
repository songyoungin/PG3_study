import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import model.dcgan as dcgan
from vis_tool import Visualizer

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Trainer(object):
    # initializer
    def __init__(self, config, dataloader):
        self.config = config
        self.dataloader = dataloader

        self.nch = int(config.nch)
        self.nz = int(config.nz)
        self.ngf = int(config.ngf)
        self.ndf = int(config.ndf)

        self.batch_size = config.batch_size
        self.image_size = config.image_size

        self.lr = config.lr
        self.beta1 = config.beta1

        self.n_epochs = config.n_epochs
        self.out_folder = config.out_folder

        self.vis = Visualizer()

        self.build_model()

    # building network
    def build_model(self):
        self.g = dcgan.Generator(self.nz, self.ngf, self.nch)
        self.g.apply(weights_init)

        # if trained weights exists
        if self.config.g != '':
            self.g.load_state_dict(torch.load(self.config.g))

        self.g.to(device)

        self.d = dcgan.Discriminator(self.ndf, self.nch)
        self.d.apply(weights_init)

        # if trained weights exists
        if self.config.d != '':
            self.d.load_state_dict(torch.load(self.config.d))

        self.d.to(device)

    # trainer method
    def train(self):
        criterion = nn.BCELoss()

        fixed_noise = torch.FloatTensor(self.batch_size, self.nz, 1, 1).normal_(0, 1).to(device)

        # setup optimizers
        g_opt = optim.Adam(self.g.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        d_opt = optim.Adam(self.d.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        print("Learning started!")

        for epoch in range(self.n_epochs):
            for step, (real_data, _) in enumerate(self.dataloader):
                # ================================================================== #
                #                      Train the discriminator                       #
                # ================================================================== #
                for p in self.d.parameters():
                    p.requires_grad = True

                real_data = real_data.to(device)
                step_batch = real_data.size(0)

                # create the labels
                target_real = torch.ones(step_batch).to(device)
                target_fake = torch.zeros(step_batch).to(device)

                # train with real data
                self.d.zero_grad()
                D_out_from_real = self.d(real_data)
                D_loss_from_real = criterion(D_out_from_real, target_real)
                D_x = D_out_from_real.data.mean()

                # train with fake data
                z = torch.randn(step_batch, self.nz).to(device)
                z = z.view(-1, self.nz, 1, 1)

                fake_data = self.g(z)

                D_out_from_fake = self.d(fake_data)
                D_loss_from_fake = criterion(D_out_from_fake, target_fake)
                D_G_z1 = D_out_from_fake.data.mean()

                # loss + forward + backward
                D_loss = D_loss_from_real + D_loss_from_fake
                D_loss.backward()
                d_opt.step()

                # ================================================================== #
                #                      Train the generator                           #
                # ================================================================== #
                for p in self.d.parameters():
                    p.requires_grad = False

                self.g.zero_grad()

                z = torch.randn(step_batch, self.nz).to(device)
                z = z.view(-1, self.nz, 1, 1)
                fake_data = self.g(z)
                D_out_from_fake = self.d(fake_data)
                D_G_z2 = D_out_from_fake.data.mean()

                # loss + forward + backward
                G_loss = criterion(D_out_from_fake, target_real)
                G_loss.backward()
                g_opt.step()

                if step % 100 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                          % (epoch, self.n_epochs, step, len(self.dataloader),
                             D_loss.item(), G_loss.item(), D_x, D_G_z1, D_G_z2))

                    # plot to visdom
                    self.vis.plot("D loss per 100 step", D_loss.item())
                    self.vis.plot("G loss per 100 step", G_loss.item())

                    # save images
                    vutils.save_image(real_data,
                                      '%s/real_samples.png' % self.out_folder,
                                      normalize=True)
                    fake = self.g(fixed_noise)
                    vutils.save_image(fake.data,
                                      '%s/fake_samples_epoch_%03d.png' % (self.out_folder, epoch),
                                      normalize=True)

            if epoch % 10 == 0:
                # save checkpoints
                torch.save(self.g.state_dict(), '%s/G_epoch_%03d.pth' % (self.out_folder, epoch))
                torch.save(self.d.state_dict(), '%s/D_epoch_%03d.pth' % (self.out_folder, epoch))

        print("learning finished!")
        torch.save(self.g.state_dict(), '%s/G_final_%03d.pth' % (self.out_folder, epoch))
        torch.save(self.d.state_dict(), '%s/D_final_%03d.pth' % (self.out_folder, epoch))
        print("save checkpoint finished!")

