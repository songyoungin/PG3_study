import torch
import torch.nn as nn
from torchvision.models import resnet18

from dataloader import get_ds_loaders

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.build_ResNet()

        self.attached_net = nn.Sequential(
            nn.Linear(512, 256),     # output: 100, 256
            nn.BatchNorm1d(256),     # output: 100, 256
            nn.Linear(256, 10)       # output: 100, 10
        ) # default: requires_grad=True

    def build_ResNet(self):
        self.resnet = resnet18(pretrained=True).to(device)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1]).to(device) # remove last FC layer

        # fix layer's parameter
        for param in self.resnet.parameters():
            param.requires_grad = False

        # print("ResNet architecture:", self.resnet)
        # for param in self.resnet.parameters():
        #     print(param.requires_grad)
        # => all parameters: requires_grad=False

    def forward(self, x):
        resnetOut = self.resnet(x)
        resnetOut = resnetOut.view(resnetOut.shape[0], -1)

        out = self.attached_net(resnetOut)
        return out

if __name__ == "__main__":
    net = CustomNet()
    net = net.to(device)
    print(net)

    for name, param in net.named_parameters():
        print(name, param.requires_grad)

    # ResNet params : requires_grad == False
    # attached net params: requires_grad == True

    datasets, dataloaders = get_ds_loaders()

    for images, labels in dataloaders['train']:
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        print(outputs.shape)
        break