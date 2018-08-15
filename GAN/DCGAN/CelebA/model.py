import torch
import torch.nn as nn

from dataloader import get_loader

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

        # CelebA -> RGB channels
        # [cf] MNIST => self.dc5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        self.dc5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

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
        out = self.tanh(self.dc5(out))  # batch, 3, 64, 64

        return out

# Discriminator
class Discriminator(nn.Module):
    # initializer
    def __init__(self, d=128):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
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
        # input: batch, 3, 64, 64

        out = self.leakyRelu(self.conv1(x))               # batch, 128, 32, 32
        out = self.leakyRelu(self.bn2(self.conv2(out)))   # batch, 256, 16, 16
        out = self.leakyRelu(self.bn3(self.conv3(out)))   # batch, 512, 8, 8
        out = self.leakyRelu(self.bn4(self.conv4(out)))   # batch, 1024, 4, 4
        out = self.sig(self.conv5(out))                   # batch, 1, 1, 1

        out = out.squeeze() # batch (1d)

        return out

if __name__ == "__main__":
    batch_size = 200

    # Test code
    G = Generator().to(device)
    D = Discriminator().to(device)

    dataloader = get_loader('../../data/resized_celebA/',
                            batch_size=batch_size)

    for image, _ in dataloader:
        print(image.shape) # batch, 3, 64, 64

        image = image.to(device)
        fake = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
        fake = fake.to(device)

        g_out = G(fake)
        print("Generator output:", g_out.shape) # batch, 3, 64, 64

        d_out = D(image)
        print("Discriminator output:", d_out.shape) # batch

        break

