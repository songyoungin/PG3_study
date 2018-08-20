from __future__ import print_function
import random
import torch
import torch.backends.cudnn as cudnn

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from dataloader import get_loader
from config import get_config
from model.cdcgan import Generator, Discriminator

from trainer import Trainer

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

    # gen = Generator(config.nz, config.ngf, config.nch, config.n_classes)
    # dis = Discriminator(config.ndf, config.nch, config.n_classes)
    #
    # gen = gen.to(device)
    # dis = dis.to(device)
    #
    # # prepare label
    # onehot = torch.zeros(10, 10)
    # onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1).view(10, 10, 1, 1)
    # fill = torch.zeros([10, 10, config.image_size, config.image_size])
    # for i in range(10):
    #     fill[i, i, :, :] = 1  # shape: 10, 10, image_size, image_size
    #
    # for x_, y_ in dataloader:
    #     step_batch = x_.size(0)
    #
    #     x_ = x_.to(device)  # shape: batch, 1, image_size, image_size
    #     y_ = y_.to(device)  # shape: batch
    #     y_fill = fill[y_]  # shape: batch, 10, image_size, image_size
    #
    #     dis_out = dis(x_, y_fill)  # dis input: x_, y_fill
    #                                # output: batch
    #
    #     z_ = torch.randn((step_batch, 100)).view(-1, 100, 1, 1).to(device)
    #     y_ = (torch.rand(step_batch, 1) * 10).long().to(device).squeeze()  # batch
    #
    #     y_label = onehot[y_]  # batch, 10, 1, 1
    #     y_fill = fill[y_]  # batch, 10, 32, 32
    #
    #     gen_out = gen(z_, y_label)  # gen input: z_, y_label
    #                                 # output: batch, 1, image_size, image_size
    #
    #     break


    trainer = Trainer(config, dataloader)
    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)