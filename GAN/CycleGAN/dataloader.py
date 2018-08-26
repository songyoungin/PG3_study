import glob, random, os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, dataroot, transform=None, unaligned=False, mode="train"):
        self.transform = transform
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(dataroot, "%s\\A" % mode) +"\\*.*"))
        print(self.files_A)


        self.files_B = sorted(glob.glob(os.path.join(dataroot, "%s\\B" % mode) +"\\*.*"))
        print(self.files_A)

    def __getitem__(self, idx):
        item_A = self.transform(Image.open(self.files_A[idx % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B)-1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[idx % len(self.files_B)]))

        return {'A': item_A, 'B':item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

def get_loader(config):
    transform = transforms.Compose([
                    transforms.Resize(int(config.image_size*1.12), Image.BICUBIC),
                    transforms.RandomCrop(config.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ])

    ds = ImageDataset(config.dataroot, transform=transform, unaligned=config.unaligned)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True, num_workers=config.n_workers)
    return dl