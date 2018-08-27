import argparse

parser = argparse.ArgumentParser()

# data processing hyper-parameters
parser.add_argument('--dataroot', type=str, default='E:\\dataset\\horse2zebra\\datasets\\horse2zebra', help='root directory of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_workers', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--unaligned', type=bool, default=True, help='whether dataset A-B matches')
parser.add_argument('--image_size', type=int, default=256, help='size of the data crop (squared assumed)')

# training hyper-parameters
parser.add_argument('--starting_epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--n_in', type=int, default=3, help='number of channels of input data')
parser.add_argument('--n_out', type=int, default=3, help='number of channels of output data')
parser.add_argument('--netG_A2B', type=str, default="", help="to continue training")
parser.add_argument('--netG_B2A', type=str, default="", help="to continue training")
parser.add_argument('--netD_A', type=str, default="", help="to continue training")
parser.add_argument('--netD_B', type=str, default="", help="to continue training")
parser.add_argument('--log_interval', type=int, default=20, help="step interval to print log message")
parser.add_argument('--sample_interval', type=int, default=5, help="epoch interval to save sample images")
parser.add_argument('--sample_folder', type=str, default=None)

config = parser.parse_args()

def get_config():
    return config