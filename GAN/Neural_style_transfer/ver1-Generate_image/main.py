from __future__ import print_function
import random

import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

from trainer import Trainer
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

    trainer = Trainer(config)
    trainer.train()

config = get_config()
main(config)