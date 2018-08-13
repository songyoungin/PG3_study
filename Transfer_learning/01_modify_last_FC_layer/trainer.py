import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler

import time
import copy
import numpy as np

from dataloader import get_ds_loaders
from vis_tool import Visualizer

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, lr, n_epochs):
        self.criterion = nn.CrossEntropyLoss()
        self.n_epochs = n_epochs
        self.lr = lr

        self.build_net()
        self.build_opt()

        self.datasets, self.dataloaders = get_ds_loaders()

        self.vis = Visualizer()

    def build_net(self):
        model = models.resnet18(pretrained=True)

        # fix layer's parameter
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)  # default: requires_grad=True

        self.model = model.to(device)
        print(self.model)

        # => all parameters except last FC layer's weight and bias are requires_grad=False

    def build_opt(self):
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, momentum=0.9)  # train only FC layer parameter
        # print(self.optimizer.param_groups) => only final FC layer's W and B

        # Decay LR by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def train(self):
        start = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.n_epochs):
            print("Epoch [%d/%d]" % (epoch+1, self.n_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.exp_lr_scheduler.step()
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                epoch_loss = []
                epoch_acc = []

                # Iterate over data.
                for step, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        # for param in self.model.parameters():
                        #     print(param, param.requires_grad)

                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    step_loss = loss.item()
                    step_acc = float(torch.sum(preds == labels.data)) / len(preds)

                    if phase == 'train':
                        print("[%d/%d] [%d/%d] loss: %.3f acc: %.3f" % (epoch+1, self.n_epochs, step+1, len(self.dataloaders[phase]), step_loss, step_acc))
                        self.vis.plot("Train loss plot per step", step_loss)
                        self.vis.plot("Train acc plot per step", step_acc)

                    epoch_loss.append(step_loss)
                    epoch_acc.append(step_acc)

                epoch_loss = np.mean(epoch_loss)
                epoch_acc = np.mean(epoch_acc)

                print("[%d/%d] phase=%s: Avg loss: %.3f Avg acc: %.3f" % (epoch+1, self.n_epochs, phase, epoch_loss, epoch_acc))
                self.vis.plot("%s avg loss plot per epoch" % phase, epoch_loss)
                self.vis.plot("%s avg acc plot per epoch" % phase, epoch_acc)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model

if __name__ == "__main__":
    trainer = Trainer(lr=0.001, n_epochs=20)
    trained_model = trainer.train()

    torch.save(trained_model, "ResNet18_CIFAR10_finetuned.pth")