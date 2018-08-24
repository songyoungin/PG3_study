import torch
from PIL import Image
from torchvision import transforms
import torchvision.utils as vutils

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load image
def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale:
        img = img.resize((int(img.size[0]/scale), int(img.size[1]/scale)), Image.ANTIALIAS)
    return img

# save image
def save_image(filename, img):
    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    img = img.clone().squeeze()
    img = denorm(img).clamp(0, 1)

    print("in save_image:", img)
    vutils.save_image(img, filename)

# compute Gram matrix
def gram_matrix(vector):
    b, ch, h, w = vector.size()
    features = vector.view(b, ch, w*h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch*h*w)
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std