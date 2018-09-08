import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import cv2

from model.nst import TransferNet
from model.vgg import VGGNet
import utils
from vis_tool import Visualizer

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer(object):
    def __init__(self, config, dataloader):
        self.config = config
        self.dataloader = dataloader

        self.batch_size = config.batch_size

        # setup style image preprocessing
        self.style_path = config.style_path
        self.style_size = config.style_size


        self.style_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])

        self.content_weight = config.content_weight
        self.style_weight = config.style_weight

        # setup training hyper-parameters
        self.lr = config.lr
        self.nepochs = config.nepochs
        self.log_interval = config.log_interval
        self.sample_interval = config.sample_interval

        self.sample_folder = config.sample_folder

        self.build_net()

    def build_net(self):
        tfNet = TransferNet()

        # to continue training
        if self.config.training != "":
            tfNet.load_state_dict(torch.load(self.config.training))

        self.tfNet = tfNet.to(device)

        vggNet = VGGNet()

        # fix vgg network's parameter
        for p in vggNet.parameters():
            p.requires_grad = False

        self.vggNet = vggNet.to(device)

    def train(self):
        vis = Visualizer()

        optimizer = optim.Adam(self.tfNet.parameters(), self.lr, betas=[0.5, 0.999])
        criterion = nn.MSELoss()

        style = utils.load_image(self.style_path)
        style = self.style_transform(style)
        style = style.repeat(self.batch_size, 1, 1, 1).to(device)
        style = utils.normalize_batch(style)

        features_style = self.vggNet(style)
        gram_style = [utils.gram_matrix(f) for f in features_style]


        start_time = time.time()
        print("Learning started!!!")
        for epoch in range(self.nepochs):
            for step, (content, _) in enumerate(self.dataloader):
                self.tfNet.train()
                step_batch = content.size(0)

                optimizer.zero_grad()

                content = content.to(device)
                output = self.tfNet(content)

                content = utils.normalize_batch(content)
                output = utils.normalize_batch(output)

                output_img = output

                features_content = self.vggNet(content)
                features_output = self.vggNet(output)

                content_loss = self.content_weight * criterion(features_output.relu2_2,
                                                               features_content.relu2_2)

                style_loss = 0.
                for ft_output, gm_style in zip(features_output, gram_style):
                    gm_output = utils.gram_matrix(ft_output)
                    style_loss += criterion(gm_output,
                                            gm_style[:step_batch, :, :])
                style_loss *= self.style_weight

                total_loss = content_loss + style_loss
                total_loss.backward()
                optimizer.step()

                if (step+1) % self.log_interval == 0:
                    end_time = time.time()
                    print("[%d/%d] [%d/%d] time: %f content loss:%.4f style loss:%.4f total loss: %.4f"
                          % (epoch+1, self.nepochs, step+1, len(self.dataloader), end_time - start_time,
                             content_loss.item(), style_loss.item(), total_loss.item()))
                    vis.plot("Content loss per %d steps" % self.log_interval, content_loss.item())
                    vis.plot("Style loss per %d steps" % self.log_interval, style_loss.item())
                    vis.plot("Total loss per %d steps" % self.log_interval, total_loss.item())

                    img = output_img.cpu()
                    img = img[0]
                    utils.save_image("%s\\output_epoch%d.png" % (self.sample_folder, epoch + 1), img)

            # do checkpointing
            if (epoch+1) % self.sample_interval == 0:
                self.tfNet.eval()
                torch.save(self.tfNet.state_dict(), "%s\\model_epoch%d.pth" % (self.sample_folder, epoch+1))

                img = output_img.cpu()
                img = img[0]
                utils.save_image("%s\\output_epoch%d.png" % (self.sample_folder, epoch+1), img)

        print("Learning finished!!!")
        self.tfNet.eval().cpu()
        torch.save(self.tfNet.state_dict(), "%s\\model_final.pth" % self.sample_folder)
        print("Save model complete!!!")


