from __future__ import print_function

import random

import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

from misc import *
from trainer import Trainer
from dataloader import get_loader
from config import get_config

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

    dataloader = get_loader(config.dataset, config.dataroot,
                            config.origin_size, config.image_size,
                            config.batch_size, config.workers,
                            split='train', shuffle=True, seed=config.manual_seed)

    val_dataloader = get_loader(config.dataset, config.val_dataroot,
                            config.image_size, config.image_size,
                            config.val_batch_size, config.workers,
                            split='val', shuffle=False, seed=config.manual_seed)

    print("Prepare dataloader complete!")

    trainer = Trainer(config, dataloader, val_dataloader)
    trainer.train()

config = get_config()
main(config)
