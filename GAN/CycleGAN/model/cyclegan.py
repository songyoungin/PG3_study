import torch
import torch.nn as nn
import torch.nn.functional as F

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.rp1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_features, in_features, 3)
        self.in1 = nn.InstanceNorm2d(in_features)
        self.rp2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(in_features, in_features, 3)
        self.in2 = nn.InstanceNorm2d(in_features)

    def forward(self, x):
        # input: 1, 3, 300, 300
        residual = x
        h = self.rp1(x)                     # 1, 3, 302, 302
        h = F.relu(self.in1(self.conv1(h))) # 1, 3, 300, 300
        h = self.rp2(h)                     # 1, 3, 302, 302
        h = self.in2(self.conv2(h))         # 1, 3, 300, 300

        return residual + h                 # 1, 3, 300, 300

class Generator(nn.Module):
    def __init__(self, inch, outch, nblocks=9):
        super(Generator, self).__init__()

        # initial convoluation block
        model = [nn.ReflectionPad2d(3),
                     nn.Conv2d(inch, 64, 7),
                     nn.InstanceNorm2d(64),
                     nn.ReLU(inplace=True)]

        # downsampling (encoder)
        # smaller, deeper
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        # setup residual blocks
        for _ in range(nblocks):
            model += [ResidualBlock(in_features)]

        # upsampling (decoder)
        # larger, flatten
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # setup output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, outch, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x) # return a image that has same size with input image

class Discriminator(nn.Module):
    def __init__(self, inch):
        super(Discriminator, self).__init__()

        model = [nn.Conv2d(inch, 64, 4, 2, 1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, 2, 1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, 2, 1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # input: 15, 3, 300, 300
        out = self.model(x)                       # 15, 1, 35, 35
        out = F.avg_pool2d(out, out.size()[2:])   # 15, 1, 1, 1
        out = out.view(out.size()[0], -1)         # 15, 1
        out = out.squeeze()
        return out


if __name__ == "__main__":
    x = torch.randn(15, 3, 300, 300).to(device)
    # res = ResidualBlock(in_features=3).to(device)
    #
    # out = res(x)
    # print("output:", out.shape)

    # netG = Generator(3, 3).to(device)
    # print(netG)
    # out = netG(x)
    # print("Generator output:", out.shape)

    # netD = Discriminator(3).to(device)
    # print(netD)
    # out = netD(x)
    # print("Discriminator output:", out.shape)