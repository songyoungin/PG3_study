import torch
import torch.nn as nn
import torch.nn.functional as F

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def get_onehot(label, ncls):
    size = label.size(0)
    label = label.long().to(device)
    oneHot = torch.zeros(size, ncls).to(device)
    oneHot.scatter_(1, label.view(size, 1), 1)
    oneHot = oneHot.to(device)
    return oneHot

class Generator(nn.Module):
    # initializer
    # nz=100, ngf=128, nch=1, ncls=10
    def __init__(self, nz, ngf, nch, ncls):
        super(Generator, self).__init__()

        # setup network
        self.dc1_x = nn.ConvTranspose2d(nz, ngf*2, 4, 1, 0)
        self.bn1_x = nn.BatchNorm2d(ngf*2)

        self.dc1_y = nn.ConvTranspose2d(ncls, ngf*2, 4, 1, 0)
        self.bn1_y = nn.BatchNorm2d(ngf*2)

        self.dc2 = nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ngf*2)

        self.dc3 = nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ngf)

        self.dc4 = nn.ConvTranspose2d(ngf, nch, 4, 2, 1)

    # weight init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x, y):
        x_out = F.relu(self.bn1_x(self.dc1_x(x)))
        y_out = F.relu(self.bn1_y(self.dc1_y(y)))
        h = torch.cat((x_out, y_out), 1)

        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        out = F.tanh(self.dc4(h))
        return out

class Discriminator(nn.Module):
    # initializer
    # ndf=128, nch=1, ncls=10
    def __init__(self, ndf, nch, ncls):
        super(Discriminator, self).__init__()


        # setup network
        self.conv1_x = nn.Conv2d(nch, int(ndf/2), 4, 2, 1)
        self.conv1_y = nn.Conv2d(ncls, int(ndf/2), 4, 2, 1)

        self.conv2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.conv3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.conv4 = nn.Conv2d(ndf*4, 1, 4, 1, 0)

    # weight init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x, y):
        x_out = F.leaky_relu(self.conv1_x(x), 0.2)          # batch, ndf/2, 16, 16
        y_out = F.leaky_relu(self.conv1_y(y), 0.2)          # batch, ndf/2, 16, 16

        h = torch.cat((x_out, y_out), 1)                    # batch, ndf, 16, 16

        h = F.leaky_relu(self.bn2(self.conv2(h)), 0.2)      # batch, ndf*2, 8, 8
        h = F.leaky_relu(self.bn3(self.conv3(h)), 0.2)      # batch, ndf*4, 4, 4
        out = F.sigmoid(self.conv4(h))                      # batch, 1, 1, 1
        out = out.squeeze()                                 # batch

        return out