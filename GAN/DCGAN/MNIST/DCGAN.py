import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

import os
import numpy as np

from torchvision.utils import save_image

from vis_tool import Visualizer

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Weight & bias normal initializer
# only works for Deconv layer & Conv layer
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# Generator
class Generator(nn.Module):
    # initializer
    def __init__(self, d=128):
        super(Generator, self).__init__()

        self.dc1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.bn1 = nn.BatchNorm2d(d*8)
        self.dc2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(d*4)
        self.dc3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(d*2)
        self.dc4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(d)
        self.dc5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

        # activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    # weight init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):

        # generator input: batch, 100, 1, 1

        # conv: large & slim image -> small & thick image
        # deconv: small & thick image -> large & slim image

        out = self.relu(self.bn1(self.dc1(x)))  # batch, 1024, 4, 4
        out = self.relu(self.bn2(self.dc2(out)))  # batch, 512, 8, 8
        out = self.relu(self.bn3(self.dc3(out)))  # batch, 256, 16, 16
        out = self.relu(self.bn4(self.dc4(out)))  # batch, 128, 32, 32
        out = self.tanh(self.dc5(out))  # batch, 1, 64, 64

        return out

# Discriminator
class Discriminator(nn.Module):
    # initializer
    def __init__(self, d=128):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

        self.leakyRelu = nn.LeakyReLU(0.2)
        self.sig = nn.Sigmoid()

    # weight init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        # input: batch, 1, 64, 64

        out = self.leakyRelu(self.conv1(x))               # batch, 128, 32, 32
        out = self.leakyRelu(self.bn2(self.conv2(out)))   # batch, 256, 16, 16
        out = self.leakyRelu(self.bn3(self.conv3(out)))   # batch, 512, 8, 8
        out = self.leakyRelu(self.bn4(self.conv4(out)))   # batch, 1024, 4, 4
        out = self.sig(self.conv5(out))                   # batch, 1, 1, 1

        out = out.squeeze() # batch (1d)

        return out

# Hyper parameters
batch_size = 100
lr = 0.0002
num_epochs = 20

# Data loader
img_size = 64

transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data',
                   train=True,
                   download=True,
                   transform=transform),
                batch_size=batch_size, shuffle=True)

sample_dir = 'samples'

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


if __name__ == "__main__":
    # set visdom tool
    vis = Visualizer()

    # build network
    G = Generator(128).to(device)
    D = Discriminator(128).to(device)

    G.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)

    # define loss function and optimizers
    criterion = nn.BCELoss()
    G_opt = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_opt = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # reset gradient of optimizers
    def reset_grad():
        G_opt.zero_grad()
        D_opt.zero_grad()

    total_step = len(dataloader)
    for epoch in range(num_epochs):
        avg_D_loss = []
        avg_G_loss = []

        for step, (real_data, _) in enumerate(dataloader):
            real_data = real_data.to(device) # shape: batch, 1, 28, 28
            # print("read_data shape:", real_data.shape)

            # create the labels
            target_real = torch.ones(batch_size).to(device)
            # print("target_real shape:", target_real.shape)

            target_fake = torch.zeros(batch_size).to(device)
            # print("target_fake shape:", target_fake.shape)

            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            # D(real image) => must be ~ 1
            # D(fake image) => must be ~ 0
            # loss = abs(D(real image)-1) + abs(D(fake image)-0)
            # minimize this loss

            # compute loss using real images
            D_result_from_real = D(real_data)
            D_loss_real = criterion(D_result_from_real, target_real)
            D_score_real = D_result_from_real

            # compute loss using fake images generated by G
            z = torch.randn(batch_size, 100).to(device)
            z = z.view(-1, 100, 1, 1)
            # print("random vector Z shape:", z.shape)

            fake_data = G(z)
            # print("fake_data shape:", fake_data.shape)

            D_result_from_fake = D(fake_data)
            D_loss_fake = criterion(D_result_from_fake, target_fake)
            D_score_fake = D_result_from_fake

            # loss + forward + backward
            D_loss = D_loss_real + D_loss_fake
            reset_grad()
            D_loss.backward()
            D_opt.step()

            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            # compute loss using fake images generated by G
            z = torch.randn(batch_size, 100).to(device)
            z = z.view(-1, 100, 1, 1)

            fake_data = G(z)
            D_result_from_fake = D(fake_data)

            # loss + forward + backward
            G_loss = criterion(D_result_from_fake, target_real)

            reset_grad()
            G_loss.backward()
            G_opt.step()

            if (step+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                      .format(epoch+1, num_epochs,
                              step + 1, total_step,
                              D_loss.item(), G_loss.item(),
                              D_score_real.mean().item(),
                              D_score_fake.mean().item()))

            avg_D_loss.append(D_loss.item())
            avg_G_loss.append(G_loss.item())

        avg_D_loss = np.mean(avg_D_loss)
        avg_G_loss = np.mean(avg_G_loss)

        true_positive_rate = (D_result_from_real > 0.5).float().mean().item()
        true_negative_rate = (D_result_from_fake < 0.5).float().mean().item()

        base_message = ("Epoch: {epoch:<3d} D Loss: {d_loss:<8.6} G Loss: {g_loss:<8.6} "
                        "True Positive Rate: {tpr:<5.1%} True Negative Rate: {tnr:<5.1%}"
                        )

        message = base_message.format(
            epoch=epoch+1,
            d_loss=avg_D_loss,
            g_loss=avg_G_loss,
            tpr=true_positive_rate,
            tnr=true_negative_rate
        )
        print(message) # pring logging

        vis.plot("Discriminator Loss per epoch", avg_D_loss)
        vis.plot("Generator Loss per epoch", avg_G_loss)

        # Save real images
        if (epoch + 1) == 1:
            save_image(denorm(real_data), os.path.join(sample_dir, 'real_images.png'))

        # Save sampled images
        save_image(denorm(fake_data), os.path.join(sample_dir, 'fake_images-{}.png'.format(100+ epoch + 1)))

    torch.save(G, "weights/DCGAN_generator.pth")
    torch.save(D, "weights/DCGAN_discriminator.pth")


