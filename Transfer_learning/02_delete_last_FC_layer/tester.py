from torchvision.models import resnet18
import torch.nn as nn
import torch

from dataloader import get_ds_loaders

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resnet_origin = resnet18(pretrained=True) # original ResNet
without_fc = nn.Sequential(*list(resnet_origin.children())[:-1]).to(device) # removing last FC layer from ResNet
resnet_ft = torch.load("../01_modify_last_FC_layer/ResNet18_CIFAR10_finetuned.pth",
                       map_location='cpu') # finetuning last FC layer with 10 classes

# print(resnet_origin)
# print(without_fc)
# print(resnet_ft)

datasets, dataloaders = get_ds_loaders()

for images, labels in dataloaders['train']:
    images = images.to(device)
    labels = labels.to(device)

    out1 = resnet_origin(images)
    print("ResNet original", out1.shape) # 100, 1000

    out2 = without_fc(images)
    print("Without FC layer", out2.shape) # 100, 512, 1, 1

    out3 = resnet_ft(images)
    print("Finetuned FC layer", out3.shape) # 100, 10
    break

