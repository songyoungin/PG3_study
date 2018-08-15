from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import cv2, os
from scipy.misc import imresize

# Define image processing
img_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))
])

# Resize images
def preprocessing(dataroot):
    save_root = '../../data/resized_celebA/'
    resize = 64

    # make folders
    if not os.path.isdir(save_root):
        os.mkdir(save_root)
    if not os.path.isdir(save_root+'celebA'):
        os.mkdir(save_root+'celebA')

    img_list = os.listdir(dataroot)

    # resizing images
    for idx in range(len(img_list)):
        img = cv2.imread(dataroot + img_list[idx]) # original image
        img = imresize(img, (resize, resize)) # resize 64*64
        cv2.imwrite(save_root+'celebA/'+img_list[idx],
                    img)

        if (idx % 1000 == 0):
            print("%d images complete!" % idx)

    # if user wants to delete original dataroot
    print("Image preprocessing complete! Do you want to delete original dataroot?")
    ch = input()
    if ch == 'y':
        os.removedirs(dataroot) # delete folder
    print("Delete complete!")

# Return image data loader
def get_loader(save_root, batch_size):
    ds = datasets.ImageFolder(save_root, transform)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # get a tester image
    temp = cv2.imread(dataloader.dataset.imgs[0][0])

    if temp.shape[0] != img_size:
        print("Error! Resize images!")
        return None
    else:
        return dataloader

if __name__ == "__main__":
    dataroot = "../../data/img_align_celeba/"
    preprocessing(dataroot)
    

    
