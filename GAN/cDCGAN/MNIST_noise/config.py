import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', help='mnist | fashion_mnist | cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', default='../../data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--image_size', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nch', type=int, default=1, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--g', default='', help="path to netG (to continue training)")
parser.add_argument('--d', default='', help="path to netD (to continue training)")
parser.add_argument('--out_folder', default=None, help='folder to output images and model checkpoints')

def get_config():
    return parser.parse_args()