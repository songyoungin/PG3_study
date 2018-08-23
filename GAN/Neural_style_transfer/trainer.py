import torch
import torch.optim as optim
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image

import numpy as np

from model.nst import VGGNet
from vis_tool import Visualizer

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(img_path, transform=None, max_size=None, shape=None):
    # load an image and convert it to a tensor
    image = Image.open(img_path)

    if max_size:
        r = max_size / max(image.size)
        size = np.array(image.size) * r
        image = image.resize(size.astype(int), Image.ANTIALIAS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform(image).unsqueeze(0)

    return image.to(device)

class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.lr = config.lr
        self.nepochs = config.nepochs


        self.log_interval = config.log_interval
        self.sample_interval = config.sample_interval

        self.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))])

        # prepare data
        self.content = load_image(config.content, self.transform, max_size=config.max_size)
        self.style = load_image(config.style, self.transform,
                                shape=[self.content.size(2), self.content.size(3)])

        # target-> output
        # optimize output's parameter directly
        self.output = self.content.clone().requires_grad_(True)

        self.build_net()

    def build_net(self):
        vgg = VGGNet()
        self.vgg = vgg.to(device).eval()

        print("model training mode?", self.vgg.training)

    def train(self):
        opt = optim.Adam([self.output], lr=self.lr, betas=[0.5, 0.999])
        vis = Visualizer()

        print("Learning started!!!")
        for epoch in range(self.nepochs):
            output_features = self.vgg(self.output)
            content_features = self.vgg(self.content)
            style_features = self.vgg(self.style)

            style_loss = 0
            content_loss = 0

            for f_out, f_content, f_style in zip(output_features, content_features, style_features):
                content_loss += torch.mean((f_out-f_content)**2)

                # reshape
                _, c, h, w = f_out.size()
                f_out = f_out.view(c, h*w)
                f_style = f_style.view(c, h*w)

                f_out = torch.mm(f_out, f_out.t())
                f_style = torch.mm(f_style, f_style.t())

                style_loss += torch.mean((f_out-f_style)**2) / (c*h*w)

            loss = content_loss + self.config.style_weight * style_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            # do logging
            if (epoch+1) % self.log_interval == 0:
                print("[{}/{}]: Content loss: {:.4f}, Style loss: {:.4f}".format(
                    epoch+1, self.nepochs, content_loss.item(), style_loss.item()
                ))
                vis.plot("Content loss per %d epochs" % self.log_interval, content_loss.item())
                vis.plot("Style loss per %d epochs" % self.log_interval, style_loss.item())

            if (epoch+1) % self.sample_interval == 0:
                denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
                img = self.output.clone().squeeze()
                img = denorm(img).clamp_(0, 1)
                vutils.save_image(img, "%s\\output-%04d.png" % (self.config.out_folder, epoch))

        print("Learning finished!!!")