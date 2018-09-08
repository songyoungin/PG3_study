import os
import torch
from torchvision import transforms
import torchvision.utils as vutils

from model.nst import TransferNet
from utils import *

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester(object):
    def __init__(self, model_path, image_path, image_num, out_folder):
        self.model_path = model_path
        self.image_path = image_path

        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])

        self.image_num = image_num
        self.style_name = style_name
        self.out_folder = out_folder

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
        file = "%s/output%d.jpg" % (self.out_folder, self.image_num)
        vutils.save_image(output, file)
        print("saving output image completed!")


if __name__ == '__main__':
    style_name = "dot-cartoon"
    in_folder = "test"
    out_folder = "test/%s" % style_name

    os.makedirs(out_folder, exist_ok=True)

    model = "samples/%s/%s.pth" % (style_name, style_name)

    for i in range(3):
        image_num = i+1
        image = "%s/input%d.jpg" % (in_folder, image_num)

        tester = Tester(model, image, image_num, out_folder)
        tester.test()

