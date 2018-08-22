from torch.utils.data import DataLoader

def get_loader(dataset, dataroot,
               origin_size, image_size, batch_size, workers,
               split='train', shuffle=True, seed=None):

    if dataset =="trans":
        from datasets.trans import trans as commonDataset
        import transforms.pix2pix as transforms
    elif dataset == 'folder':
        from torchvision.datasets.folder import ImageFolder as commonDataset
        import torchvision.transforms as transforms
    elif dataset == 'pix2pix':
        from datasets.pix2pix import pix2pix as commonDataset
        import transforms.pix2pix as transforms

    if dataset != "folder":
        # for training set
        if split == "train":
            transform = transforms.Compose([
                                    transforms.Resize(origin_size),
                                    transforms.RandomCrop(image_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                         std=(0.5, 0.5, 0.5))])
            dataset = commonDataset(root=dataroot, transform=transform, seed=seed)
        # for validating set
        else:
            transform = transforms.Compose([
                transforms.Resize(origin_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))])
            dataset = commonDataset(root=dataroot, transform=transform, seed=seed)
    else:
        # for training set
        if split == "train":
            transform = transforms.Compose([
                transforms.Resize(origin_size),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))])
            dataset = commonDataset(root=dataroot, transform=transform, seed=seed)
        # for validating set
        else:
            transform = transforms.Compose([
                transforms.Resize(origin_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))])
            dataset = commonDataset(root=dataroot, transform=transform, seed=seed)

    assert dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers)
    return dataloader
