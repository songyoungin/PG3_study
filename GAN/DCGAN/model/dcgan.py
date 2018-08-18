import torch.nn as nn

class Generator(nn.Module):
    # initializer
    def __init__(self, nz, ngf, nch):
        super(Generator, self).__init__()

        self.nz = nz          # input(z) dimension
        self.ngf = ngf        # hidden dimension
        self.nch = nch        # num of channels

        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*8, self.ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*4, self.ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf*2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf, self.nch, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    # forward method
    def forward(self, x):
        out = self.net(x)
        return out


class Discriminator(nn.Module):
    # initializer
    def __init__(self, ndf, nch):
        super(Discriminator, self).__init__()

        self.ndf = ndf     # hidden dimension
        self.nch = nch     # num of channels

        self.net = nn.Sequential(
            # Input: nch, 64, 64
            nn.Conv2d(self.nch, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf*4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    # forward method
    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, 1).squeeze()
        return out