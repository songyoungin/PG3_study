from __future__ import print_function
import random
import torch
import torch.backends.cudnn as cudnn

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from dataloader import get_loader
from config import get_config
from trainer import Trainer

def main(config):
    if config.out_folder is None:
        config.out_folder = 'new_samples'
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
    # for data, _ in dataloader:
    #     print(data.size())
    #     break

    trainer = Trainer(config, dataloader)
    trainer.train()

if __name__ == "__main__":
    config = get_config()
    main(config)