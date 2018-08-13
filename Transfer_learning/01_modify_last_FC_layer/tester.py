import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np

from dataloader import get_ds_loaders

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Tester():
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.datasets, self.dataloaders = get_ds_loaders()

        self.build_net()
        self.load_trained_model()

    def build_net(self):
        model = models.resnet18(pretrained=True)

        # fix layer's parameter
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)  # default: requires_grad=True

        self.model = model.to(device)

    def load_trained_model(self):
        self.model = torch.load(self.ckpt_path, map_location='cpu')
        #self.model.load_state_dict(torch.load(self.ckpt_path, map_location='cpu'))
        print(self.model)

    def test(self):
        self.model.eval()
        total_loss = []
        total_acc = []
        criterion = nn.CrossEntropyLoss()

        for step, (inputs, labels) in enumerate(self.dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                # print(loss)

            # statistics
            step_loss = loss.item()
            step_acc = float(torch.sum(preds == labels.data)) / len(preds)

            total_loss.append(step_loss)
            total_acc.append(step_acc)

        total_loss = np.mean(total_loss)
        total_acc = np.mean(total_acc)

        print("Test loss: %.3f Test accuracy: %.3f" % (total_loss, total_acc))


if __name__ == "__main__":
    tester = Tester("ResNet18_CIFAR10_finetuned.pth")
    tester.test()

