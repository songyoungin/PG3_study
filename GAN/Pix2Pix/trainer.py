import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

from models.pix2pix import Generator, Discriminator
from misc import *

import time, sys

from vis_tool import Visualizer

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer(object):
    def __init__(self, config, dataloader, val_dataloader):
        self.config = config
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader

        self.batch_size = config.batch_size

        self.inch = config.inch
        self.outch = config.outch
        self.ngf = config.ngf
        self.ndf = config.ndf

        self.mode = config.mode
        self.origin_size = config.origin_size
        self.image_size = config.image_size

        self.nepochs = config.nepochs
        self.lrG = config.lrG
        self.lrD = config.lrD

        self.lambdaGAN = config.lambdaGAN
        self.lambdaIMG = config.lambdaIMG

        self.weight_decay = config.weight_decay
        self.beta1 = config.beta1

        self.out_folder = config.out_folder
        self.log_interval = config.log_interval
        self.val_interval = config.val_interval

        self.train_logger = open("%s/train.log" % self.out_folder, 'w')

        self.build_net()
        self.sample_val_images()

    def build_net(self):
        netG = Generator(self.inch, self.outch, self.ngf)
        netG.apply(weights_init)

        # to continue training
        if self.config.netG != "":
            netG.load_state_dict(torch.load(self.config.netG))
            print("load generator!")

        netD = Discriminator(self.inch+self.outch, self.ndf)
        netD.apply(weights_init)

        # to continue training
        if self.config.netD != "":
            netD.load_state_dict(torch.load(self.config.netD))
            print("load discriminator!")

        self.netG = netG.to(device)
        self.netD = netD.to(device)

    def sample_val_images(self):
        val_iter = iter(self.val_dataloader)
        data_val = val_iter.next()

        if self.mode == "B2A":
            val_target, val_input = data_val
        elif self.mode == "A2B":
            val_input, val_target = data_val

        self.val_input, self.val_target = val_input.to(device), val_target.to(device)

        vutils.save_image(val_target,"%s/real_target.png" % self.out_folder, normalize=True)
        vutils.save_image(val_input, "%s/real_input.png" % self.out_folder, normalize=True)


    def train(self):
        vis = Visualizer()

        bce = nn.BCELoss()
        cae = nn.L1Loss()
        size_PatchGAN = 30

        # setup optimizers
        if self.config.rmsprop:
            optG = optim.RMSprop(self.netG.parameters(), lr=self.lrG)
            optD = optim.RMSprop(self.netD.parameters(), lr=self.lrD)
        else:
            optG = optim.Adam(self.netG.parameters(), lr=self.lrG,
                              betas=(self.beta1, 0.999), weight_decay=0.0)
            optD = optim.Adam(self.netD.parameters(), lr=self.lrD,
                              betas=(self.beta1, 0.999), weight_decay=self.weight_decay)


        # training loop
        gan_iter = 0
        start_time = time.time()
        for epoch in range(self.nepochs):
            for step, data in enumerate(self.dataloader, 0):
                # set models train mode
                self.netD.train()
                self.netG.train()

                # facades-> imageA: target, imageB: input (B2A)
                # data = [A | B]
                if self.mode == "B2A":
                    target, input = data
                elif self.mode == "A2B":
                    input, target = data

                step_batch = target.size(0)

                input, target = input.to(device), target.to(device)

                targetD_real = torch.ones(step_batch, 1, size_PatchGAN, size_PatchGAN)
                targetD_fake = torch.ones(step_batch, 1, size_PatchGAN, size_PatchGAN)

                targetD_real, targetD_fake = targetD_real.to(device), targetD_fake.to(device)

                #=============================================#
                #             Train discriminator             #
                #=============================================#
                for param in self.netD.parameters():
                    param.requires_grad = True
                self.netD.zero_grad()

                outD_real = self.netD(torch.cat((target, input), 1)) # conditional GAN
                errD_real = bce(outD_real, targetD_real)

                D_x = outD_real.data.mean()

                x_hat = self.netG(input)
                fake = x_hat.detach()
                outD_fake = self.netD(torch.cat([fake, input], 1)) # conditional GAN
                errD_fake = bce(outD_fake, targetD_fake)

                errD = (errD_real + errD_fake) * 0.5 # combined loss
                errD.backward()

                D_G_z1 = outD_fake.data.mean()

                optD.step()

                # =============================================#
                #             Train generator                  #
                # =============================================#
                for param in self.netD.parameters():
                    param.requires_grad = False
                self.netG.zero_grad()

                # compute L_L1
                if self.lambdaIMG != 0:
                    errG_l1 = cae(x_hat, target) * self.lambdaIMG

                # compute L_cGAN
                outD_fake = self.netD(torch.cat((x_hat, input), 1)) # conditional
                targetD_real = torch.ones(step_batch, 1, size_PatchGAN, size_PatchGAN).to(device)

                if self.lambdaGAN != 0:
                    errG_gan =  bce(outD_fake, targetD_real)

                D_G_z2 = outD_fake.data.mean()

                # combined loss
                errG = errG_l1 + errG_gan
                errG.backward()
                optG.step()

                gan_iter += 1

                if gan_iter % self.log_interval == 0:
                    end_time = time.time()
                    print("[%d/%d] [%d/%d] time:%f D loss:%.3f G_L1 loss:%.3f G_gan loss:%.3f D(x)=%.3f D(G(z))=%.3f/ %.3f"
                          % (epoch+1, self.nepochs, step+1, len(self.dataloader),
                             end_time-start_time, errD.item(), errG_l1.item(), errG_gan.item(), D_x, D_G_z1, D_G_z2))

                    sys.stdout.flush()
                    self.train_logger.write('%d\t%f\t%f\t%f\t%f\t%f\t%f\n' % \
                                      (gan_iter, errD.item(), errG_l1.item(), errG_gan.item(), D_x, D_G_z1, D_G_z2))
                    self.train_logger.flush()

                    vis.plot("D loss per %d steps" % self.log_interval, errD.item())
                    vis.plot("G_L1 loss per %d steps" % self.log_interval, errG_l1.item())
                    vis.plot("G_gan loss per %d steps" % self.log_interval, errG_gan.item())

            # do checkpointing
            torch.save(self.netG.state_dict(), "%s/netG_epoch_%d.pth" % (self.out_folder, epoch + 200))
            torch.save(self.netD.state_dict(), "%s/netD_epoch_%d.pth" % (self.out_folder, epoch + 200))

            # do validating
            self.netD.eval()
            self.netG.eval()
            val_batch_output = torch.zeros(self.val_input.size())
            for idx in range(self.val_input.size(0)):
                img = self.val_input[idx, :, :, :].unsqueeze(0)
                input_img = Variable(img, volatile=True).to(device)
                x_hat_val = self.netG(input_img)
                val_batch_output[idx, :, :, :].copy_(x_hat_val.data[0])

            vutils.save_image(val_batch_output, "%s/generated_epoch%03d.png"
                              % (self.out_folder, epoch+200))

        print("Learning finished!")