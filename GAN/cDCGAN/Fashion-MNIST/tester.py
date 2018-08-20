import torch
import numpy as np
import cv2
from model.cgan import Generator
from config import get_config

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform label to onehot format
def get_onehot(label, n_classes):
    step_batch = label.size(0)
    label = label.long().to(device)
    oneHot = torch.zeros(step_batch, n_classes).to(device)
    oneHot.scatter_(1, label.view(step_batch, 1), 1)
    oneHot = oneHot.to(device)
    return oneHot

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

config = get_config()
modelGen = Generator(config.nz, config.ngf, config.n_classes, config.image_size).to(device)
modelGen.load_state_dict(torch.load("samples/weights/G_final_049.pth"))

print(modelGen)

random_z = torch.FloatTensor(2, config.nz).normal_(0, 1).to(device)
random_y = (torch.from_numpy(np.asarray([[5], [8]])) * config.n_classes).long().to(device)
random_y = get_onehot(random_y, config.n_classes)

out = modelGen(random_z, random_y)
out = denorm(out)
out = out.view(-1, 1, 28, 28).cpu().detach().numpy()

img1 = np.transpose(out[0], (2, 1, 0))
img2 = np.transpose(out[1], (2, 1, 0))

img1 = cv2.resize(img1, (120, 120))
img2 = cv2.resize(img2, (120, 120))

cv2.imshow("CGAN test",img1)
cv2.waitKey(0)

cv2.imshow("CGAN test",img2)
cv2.waitKey(0)