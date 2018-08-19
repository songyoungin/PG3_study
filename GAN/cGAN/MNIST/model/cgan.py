import torch
import torch.nn as nn

class Generator(nn.Module):
    # initializer
    # nz=100, ngf=256, num_classes=10, image_size=28
    def __init__(self, nz, ngf, num_classes, image_size):
        super(Generator, self).__init__()

        self.fc1_x = nn.Linear(nz, ngf)
        self.fc1_x_bn = nn.BatchNorm1d(ngf)

        self.fc1_y = nn.Linear(num_classes, ngf)
        self.fc1_y_bn = nn.BatchNorm1d(ngf)

        self.fc2 = nn.Linear(ngf*2, ngf*2)
        self.fc2_bn = nn.BatchNorm1d(ngf*2)

        self.fc3 = nn.Linear(ngf*2, ngf*4)
        self.fc3_bn = nn.BatchNorm1d(ngf*4)

        self.fc4 = nn.Linear(ngf*4, image_size**2)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    # normal init
    def normal_init(self, m, mean, std):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

    # initialize weight
    def weight_init(self, mean, std):
        for m in self._modules:
           self.normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x, y):
        x_out = self.relu(self.fc1_x_bn(self.fc1_x(x)))
        y_out = self.relu(self.fc1_y_bn(self.fc1_y(y)))

        h = torch.cat((x_out, y_out), 1)

        h = self.relu(self.fc2_bn(self.fc2(h)))
        h = self.relu(self.fc3_bn(self.fc3(h)))

        out = self.tanh(self.fc4(h))
        return out

class Discriminator(nn.Module):
    # initializer
    # ndf=256, num_classes=10, image_size=28
    def __init__(self, ndf, num_classes, image_size):
        super(Discriminator, self).__init__()

        self.fc1_x = nn.Linear(image_size**2, ndf*4)
        self.fc1_y = nn.Linear(num_classes, ndf*4)

        self.fc2 = nn.Linear(ndf*8, ndf*2)
        self.fc2_bn = nn.BatchNorm1d(ndf*2)

        self.fc3 = nn.Linear(ndf*2, ndf)
        self.fc3_bn = nn.BatchNorm1d(ndf)

        self.fc4 = nn.Linear(ndf, 1)

        self.leaky = nn.LeakyReLU(0.2)
        self.sig = nn.Sigmoid()

    # normal init
    def normal_init(self, m, mean, std):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

    # initialize weight
    def weight_init(self, mean, std):
        for m in self._modules:
            self.normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x, y):
        x_out = self.leaky(self.fc1_x(x))
        y_out = self.leaky(self.fc1_y(y))

        h = torch.cat((x_out, y_out), 1)

        h = self.leaky(self.fc2_bn(self.fc2(h)))
        h = self.leaky(self.fc3_bn(self.fc3(h)))
        out = self.sig(self.fc4(h))
        return out
