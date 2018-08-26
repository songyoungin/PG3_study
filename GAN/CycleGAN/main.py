from __future__ import print_function
import random
import os

import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

from trainer import Trainer
from config import get_config
from dataloader import get_loader

def main(config):
    if config.sample_folder is None:
        config.sample_folder = 'samples'
    os.system('mkdir {0}'.format(config.sample_folder))

    config.manual_seed = random.randint(1, 10000)
    print("Random Seed: ", config.manual_seed)

    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed_all(config.manual_seed)

    cudnn.benchmark = True

    dataloader = get_loader(config)

    print("Prepare dataloader complete!!!")

    trainer = Trainer(config, dataloader)
    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)

