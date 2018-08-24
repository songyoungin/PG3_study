from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

def get_loader(config):
    dataroot = config.dataroot
    transform = transforms.Compose([
        transforms.Resize(config.content_size),
        transforms.CenterCrop(config.content_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    dataset = ImageFolder(dataroot, transform)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)

    return dataloader