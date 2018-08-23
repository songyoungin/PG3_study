import torch
import torch.nn as nn

from torchvision import models

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']

        # select feature extraction layers of vgg
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            # print("name:", name, " layer:", layer)
            x = layer(x)
            if name in self.select:
                features.append(x)

        return features

if __name__ == "__main__":
    vgg = VGGNet().to(device).eval()
    randIn = torch.rand(1, 3, 500, 400).to(device)
    randOut = vgg(randIn)

    for f in randOut:
        print(f.shape)

    """
    torch.Size([1, 64, 400, 400])
    torch.Size([1, 128, 200, 200])
    torch.Size([1, 256, 100, 100])
    torch.Size([1, 512, 50, 50])
    torch.Size([1, 512, 25, 25])
    """