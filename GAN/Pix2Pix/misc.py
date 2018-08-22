import torch


import os
import numpy as np

def create_exp_dir(exp):
    try:
        os.makedirs(exp, exist_ok=True)
        print('Creating exp dir: %s' % exp)
    except OSError:
        pass
    return True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def check_cuda(config):
    if torch.cuda.is_available() and not config.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        config.cuda = True
    return config


################
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image
        if self.num_imgs < self.pool_size:
            self.images.append(image.clone())
            self.num_imgs += 1
            return image
        else:
            # import pdb; pdb.set_trace()
            if np.random.uniform(0, 1) > 0.5:
                random_id = np.random.randint(self.pool_size, size=1)[0]
                tmp = self.images[random_id].clone()
                self.images[random_id] = image.clone()
                return tmp
            else:
                return image


def adjust_learning_rate(optimizer, init_lr, epoch, factor, every):
    # import pdb; pdb.set_trace()
    lrd = init_lr / every
    old_lr = optimizer.param_groups[0]['lr']
    lr = old_lr - lrd
    if lr < 0: lr = 0
    # optimizer.param_groups[0].keys()
    # ['betas', 'weight_decay', 'params', 'eps', 'lr']
    # optimizer.param_groups[1].keys()
    # *** IndexError: list index out of range
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr