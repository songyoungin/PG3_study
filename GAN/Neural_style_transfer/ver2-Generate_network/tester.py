import torch
from torchvision import transforms
import torchvision.utils as vutils

from model.nst import TransferNet
from utils import *

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester(object):
    def __init__(self, model_path, image_path):
        self.model_path = model_path
        self.image_path = image_path

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])

        self.load_net()

    def load_net(self):
        model = TransferNet()
        model.load_state_dict(torch.load(self.model_path))
        self.model = model.to(device)
        self.model.eval()

    def test(self):
        image = load_image(self.image_path)
        image = self.transform(image)
        image = image.unsqueeze(0).to(device)

        output = self.model(image)
        output = output.div(255.0).clamp(0, 1)
        vutils.save_image(output, "test\\output2.png")


if __name__ == '__main__':
    model = "samples\\model_final.pth"
    image = "test\\input2.jpg"

    tester = Tester(model, image)
    tester.test()

