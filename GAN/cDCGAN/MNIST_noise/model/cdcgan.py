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


if __name__ == "__main__":
    # test codes
    import torchvision.datasets as dsets
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms

    image_size = 32

    dataloader = DataLoader(dataset=dsets.MNIST(root='../../../data',
                                                download=True,
                                                transform=transforms.Compose([
                                                    transforms.Resize(image_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                ])),
                            shuffle=True,
                            num_workers=0,
                            batch_size=128)

    gen = Generator(100, 128, 1, 10)
    dis = Discriminator(128, 1, 10)

    gen = gen.to(device)
    dis = dis.to(device)


    # prepare label
    onehot = torch.zeros(10, 10)
    onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1).view(10, 10, 1, 1)
    fill = torch.zeros([10, 10, image_size, image_size])
    for i in range(10):
        fill[i, i, :, :] = 1       # shape: 10, 10, image_size, image_size


    for x_, y_ in dataloader:
        step_batch = x_.size(0)

        x_ = x_.to(device)    # shape: batch, 1, image_size, image_size
        y_ = y_.to(device)    # shape: batch
        y_fill = fill[y_]  # shape: batch, 10, image_size, image_size

        dis_out = dis(x_, y_fill)   # dis input: x_, y_fill
                                    # output: batch

        z_ = torch.randn((step_batch, 100)).view(-1, 100, 1, 1).to(device)
        y_ = (torch.rand(step_batch, 1) * 10).long().to(device).squeeze()  # batch

        y_label = onehot[y_]     # batch, 10, 1, 1
        y_fill = fill[y_]        # batch, 10, 32, 32

        gen_out = gen(z_, y_label)  # gen input: z_, y_label
                                    # output: batch, 1, image_size, image_size

        break

    single = torch.rand(step_batch, 1)
    print(single.shape)

    many = torch.rand(step_batch, 1) * 10
    print(many.shape)


