import torch
import torch.nn as nn
from torch.autograd import Variable

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define block UNET in generator
def blockUNet(inch, outch, name, transposed=False, bn=True, relu=True, dropout=False):
    block = nn.Sequential()

    # setup activation function
    if relu:
        block.add_module("%s_relu" % name, nn.ReLU(inplace=True))
    else:
        block.add_module("%s_leakyrelu" % name, nn.LeakyReLU(0.2, inplace=True))
    # setup convolutional layer
    if not transposed:
        block.add_module("%s_conv" % name, nn.Conv2d(inch, outch, 4, 2, 1, bias=False))
    else:
        block.add_module("%s_tconv" % name, nn.ConvTranspose2d(inch, outch, 4, 2, 1, bias=False))
    # setup batchnorm layer
    if bn:
        block.add_module("%s_bn" % name, nn.BatchNorm2d(outch))
    # setup dropout layer
    if dropout:
        block.add_module("%s_dropout" % name, nn.Dropout(0.5, inplace=True))

    return block

# define Generator
class Generator(nn.Module):
    def __init__(self, inch, outch, ngf):
        super(Generator, self).__init__()
        # ====================================#
        #        Define encoder layers        #
        # ====================================#
        # input: 256, 256
        layer_idx = 1
        name = "layer%d" % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(inch, ngf, 4, 2, 1, bias=False))
        # input: 128, 128
        layer_idx += 1
        name = "layer%d" % layer_idx
        layer2 = blockUNet(ngf, ngf*2, name,
                           transposed=False, bn=True, relu=False, dropout=False)
        # input: 64, 64
        layer_idx += 1
        name = "layer%d" % layer_idx
        layer3 = blockUNet(ngf*2, ngf*4, name,
                           transposed=False, bn=True, relu=False, dropout=False)
        # input: 32, 32
        layer_idx += 1
        name = "layer%d" % layer_idx
        layer4 = blockUNet(ngf*4, ngf*8, name,
                           transposed=False, bn=True, relu=False, dropout=False)
        # input: 16, 16
        layer_idx += 1
        name = "layer%d" % layer_idx
        layer5 = blockUNet(ngf*8, ngf*8, name,
                           transposed=False, bn=True, relu=False, dropout=False)
        # input: 8, 8
        layer_idx += 1
        name = "layer%d" % layer_idx
        layer6 = blockUNet(ngf*8, ngf*8, name,
                           transposed=False, bn=True, relu=False, dropout=False)
        # input: 4, 4
        layer_idx += 1
        name = "layer%d" % layer_idx
        layer7 = blockUNet(ngf*8, ngf*8, name,
                           transposed=False, bn=True, relu=False, dropout=False)
        # input: 2, 2
        layer_idx += 1
        name = "layer%d" % layer_idx
        layer8 = blockUNet(ngf*8, ngf*8, name,
                           transposed=False, bn=False, relu=False, dropout=False)

        # ====================================#
        #        Define decoder layers        #
        # ====================================#
        name = "dlayer%d" % layer_idx
        d_inch = ngf*8
        # input: 1, 1
        dlayer8 = blockUNet(d_inch, ngf*8, name,
                            transposed=True, bn=True, relu=True, dropout=True)
        # input: 2, 2
        layer_idx -= 1
        name = "dlayer%d" % layer_idx
        d_inch = ngf*8*2
        dlayer7 = blockUNet(d_inch, ngf*8, name,
                            transposed=True, bn=True, relu=True, dropout=True)
        # input: 4, 4
        layer_idx -= 1
        name = "dlayer%d" % layer_idx
        d_inch = ngf * 8 * 2
        dlayer6 = blockUNet(d_inch, ngf * 8, name,
                            transposed=True, bn=True, relu=True, dropout=True)
        # input: 8, 8
        layer_idx -= 1
        name = "dlayer%d" % layer_idx
        d_inch = ngf * 8 * 2
        dlayer5 = blockUNet(d_inch, ngf * 8, name,
                            transposed=True, bn=True, relu=True, dropout=False)
        # input: 16, 16
        layer_idx -= 1
        name = "dlayer%d" % layer_idx
        d_inch = ngf * 8 * 2
        dlayer4 = blockUNet(d_inch, ngf * 4, name,
                            transposed=True, bn=True, relu=True, dropout=False)
        # input: 32, 32
        layer_idx -= 1
        name = "dlayer%d" % layer_idx
        d_inch = ngf*4*2
        dlayer3 = blockUNet(d_inch, ngf*2, name,
                            transposed=True, bn=True, relu=True, dropout=False)
        # input: 64, 64
        layer_idx -= 1
        name = "dlayer%d" % layer_idx
        d_inch = ngf * 2 * 2
        dlayer2 = blockUNet(d_inch, ngf, name,
                            transposed=True, bn=True, relu=True, dropout=False)
        # input: 128, 128
        layer_idx -= 1
        name = "dlayer%d" % layer_idx
        dlayer1 = nn.Sequential()
        d_inch = ngf*2
        dlayer1.add_module('%s_relu' % name, nn.ReLU(inplace=True))
        dlayer1.add_module('%s_tconv' % name, nn.ConvTranspose2d(d_inch, outch, 4, 2, 1, bias=False))
        dlayer1.add_module('%s_tanh' % name, nn.Tanh())

        # assign layers
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.layer7 = layer7
        self.layer8 = layer8
        self.dlayer8 = dlayer8
        self.dlayer7 = dlayer7
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

    def forward(self, x):
        # ====================================#
        #      forward to encoder layers      #
        # ====================================#
        out1 = self.layer1(x)              # 1, ngf, 128, 128
        out2 = self.layer2(out1)           # 1, ngf*2, 128, 128
        out3 = self.layer3(out2)           # 1, ngf*4, 32, 32
        out4 = self.layer4(out3)           # 1, ngf*8, 16, 16
        out5 = self.layer5(out4)           # 1, ngf*8, 8, 8
        out6 = self.layer6(out5)           # 1, ngf*8, 4, 4
        out7 = self.layer7(out6)           # 1, ngf*8, 2, 2
        out8 = self.layer8(out7)           # 1, ngf*8, 1, 1

        # ====================================#
        #      forward to decoder layers      #
        # ====================================#
        dout8 = self.dlayer8(out8)                  # 1, ngf*8, 2, 2

        # connect with encoder layer's output
        dout8_out7 = torch.cat([dout8, out7], 1)    # 1, ngf*16, 2, 2

        dout7 = self.dlayer7(dout8_out7)            # 1, ngf*8, 4, 4
        dout7_out6 = torch.cat([dout7, out6], 1)    # 1, ngf*16, 4, 4
        dout6 = self.dlayer6(dout7_out6)            # 1, ngf*8, 8, 8
        dout6_out5 = torch.cat([dout6, out5], 1)    # 1, ngf*16, 8, 8
        dout5 = self.dlayer5(dout6_out5)            # 1, ngf*8, 16, 16
        dout5_out4 = torch.cat([dout5, out4], 1)    # 1, ngf*16, 16, 16
        dout4 = self.dlayer4(dout5_out4)            # 1, ngf*4, 32, 32
        dout4_out3 = torch.cat([dout4, out3], 1)    # 1, ngf*8, 32, 32
        dout3 = self.dlayer3(dout4_out3)            # 1, ngf*2, 64, 64
        dout3_out2 = torch.cat([dout3, out2], 1)    # 1, ngf*4, 64, 64
        dout2 = self.dlayer2(dout3_out2)            # 1, ngf, 128, 128
        dout2_out1 = torch.cat([dout2, out1], 1)    # 1, ngf*2, 128, 128

        out = self.dlayer1(dout2_out1)            # 1, outch, 256, 256

        return out

class Discriminator(nn.Module):
    def __init__(self, nch, ndf):
        super(Discriminator, self).__init__()

        net = nn.Sequential()

        # in: 256, 256
        layer_idx = 1
        name = "layer%d" % layer_idx
        net.add_module("%s_conv" % name, nn.Conv2d(nch, ndf, 4, 2, 1, bias=False))

        # in: 128, 128
        layer_idx += 1
        name = "layer%d" % layer_idx
        net.add_module(name,
                       blockUNet(ndf, ndf*2, name,
                                 transposed=False, bn=True, relu=False, dropout=False))

        # in: 64, 64
        layer_idx += 1
        name = "layer%d" % layer_idx
        ndf *= 2
        net.add_module(name,
                       blockUNet(ndf, ndf*2, name,
                                 transposed=False, bn=True, relu=False, dropout=False))
        
        # in: 32, 32
        layer_idx += 1
        name = "layer%d" % layer_idx
        ndf *= 2
        net.add_module("%s_leakyrelu" % name, nn.LeakyReLU(0.2, inplace=True))
        net.add_module("%s_conv" % name, nn.Conv2d(ndf, ndf*2, 4, 1, 1, bias=False))
        net.add_module("%s_bn" % name, nn.BatchNorm2d(ndf*2))

        # in: 31, 31
        layer_idx += 1
        name = "layer%d" % layer_idx
        ndf *= 2
        net.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
        net.add_module('%s_conv' % name, nn.Conv2d(ndf, 1, 4, 1, 1, bias=False))
        net.add_module('%s_sigmoid' % name, nn.Sigmoid())
        # out: 30, 30 (size of PatchGAN=30)

        # assign network
        self.net = net

    def forward(self, x):
        out = self.net(x) # 1, 1, 30
        return out

if __name__ == "__main__":

    imgA = torch.rand(1, 3, 256, 256).to(device)
    imgB = torch.rand(1, 3, 256, 256).to(device)

    netG = Generator(inch=3, outch=3, ngf=64).to(device)
    outG = netG(imgA)
    print("outG:", outG.shape)

    concat = torch.cat((imgA, imgB), 1).to(device)
    netD = Discriminator(6, ndf=64).to(device)
    outD = netD(concat)
    print("outD:", outD.shape)