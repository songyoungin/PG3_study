import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

def get_loader(_dataset, dataroot, batch_size, num_workers, image_size):
    # folder dataset
    if _dataset in ['imagenet', 'folder', 'lfw']:
        dataroot += '/resized_celebA'
        dataset = dset.ImageFolder(root=dataroot,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif _dataset == 'lsun':
        dataroot += '/lsun'
        dataset = dset.LSUN(db_path=dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    elif _dataset == 'cifar10':
        dataroot += '/cifar10'
        dataset = dset.CIFAR10(root=dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    elif _dataset == 'mnist':
        dataroot += '/mnist'
        dataset = dset.MNIST(root=dataroot, download=True,
                             transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    elif _dataset == 'fashion':
        dataroot += '/fashion'
        dataset = dset.FashionMNIST(root=dataroot, download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))

    elif _dataset == 'fake':
        dataroot += '/fake'
        dataset = dset.FakeData(image_size=(3, image_size, image_size),
                                transform=transforms.ToTensor())

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)

    return dataloader