from __future__ import print_function
import random
import torch
import torch.backends.cudnn as cudnn

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from dataloader import get_loader
from config import get_config
from trainer import Trainer
from model.cgan import Generator, Discriminator

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(config):
    if config.out_folder is None:
        config.out_folder = 'samples'
    os.system('mkdir {0}'.format(config.out_folder))

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)

    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed_all(config.manual_seed)

    cudnn.benchmark = True

    dataloader = get_loader(_dataset=config.dataset,
                             dataroot=config.dataroot,
                             batch_size=config.batch_size,
                             num_workers=int(config.workers),
                             image_size=config.image_size)

    print("Prepare dataloader complete!")

    gNet = Generator(config.nz, config.ngf, config.n_classes, config.image_size).to(device)
    dNet = Discriminator(config.ngf, config.n_classes, config.image_size).to(device)
    dNet.weight_init(mean=0, std=0.02)

    # for input, label in dataloader:
    #     print("Input tensor size:", input.size())
    #     step_batch = input.size(0)
    #
    #     input = input.view(-1, config.image_size**2).to(device)
    #     label = label.to(device).long()
    #
    #     onehot = torch.zeros(step_batch, config.n_classes)
    #     onehot.scatter_(1, label.view(step_batch, 1), 1)
    #
    #     random_z = torch.rand((step_batch, config.nz)).to(device)
    #     random_label = (torch.rand(step_batch, 1) * config.n_classes).long().to(device)
    #     random_onehot = torch.zeros(step_batch, config.n_classes)
    #     random_onehot.scatter_(1, random_label.view(step_batch, 1), 1).to(device)
    #
    #     gNet_out = gNet(random_z, random_onehot)
    #     print("G out size:", gNet_out.size())
    #
    #     dNet_out = dNet(input, onehot)
    #     print("D out size:", dNet_out.size())
    #
    #     break

    trainer = Trainer(config, dataloader)
    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)