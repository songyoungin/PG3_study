import torch
import torch.nn as nn


from model.cdcgan import Generator, Discriminator
from vis_tool import Visualizer

class Trainer(object):
    # initializer
    def __init__(self, config, dataloader):
        self.config = config
        self.dataloader = dataloader

        
