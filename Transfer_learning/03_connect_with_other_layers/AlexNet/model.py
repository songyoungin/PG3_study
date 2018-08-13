import torch
import torch.nn as nn
from torchvision.models.alexnet import alexnet

from dataloader import get_ds_loaders

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.build_ResNet()                # all params are requires_grad == False

        self.attached_net = nn.Sequential( # input 100, 256*6*6
            nn.Linear(256*6*6, 6*6),       # output: 100, 36
            nn.BatchNorm1d(6*6),           # output: 100, 36
            nn.Linear(6*6, 10)             # output: 36, 10
        )                                  # default: requires_grad=True

    def build_ResNet(self):
        self.alexnet = alexnet(pretrained=True).to(device)
        self.alexnet = nn.Sequential(*list(self.alexnet.children())[:-1]).to(device) # remove classifier

        # fix layer's parameter
        for param in self.alexnet.parameters():
            param.requires_grad = False

        # for name, param in self.alexnet.named_parameters():
        #     print(name, param.requires_grad)
                                            # => all parameters: requires_grad=False

    def forward(self, x):
        alexOut = self.alexnet(x)
        alexOut = alexOut.view(alexOut.shape[0], -1) # 100, 256*6*6
        out = self.attached_net(alexOut)
        return out

if __name__ == "__main__":
    model = CustomNet().to(device)
    # print(model)
    #
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    datasets, dataloaders = get_ds_loaders()
    for images, labels in dataloaders['train']:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        print(outputs.shape)
        break