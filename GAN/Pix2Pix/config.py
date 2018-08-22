import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='pix2pix',  help='')
parser.add_argument('--dataroot', required=False,
                    default='C:\\Users\\USER\Desktop\pix2pix-master\pix2pix-master\datasets\\facades\\train',
                    help='path to trn dataset')
parser.add_argument('--val_dataroot', required=False,
                    default='C:\\Users\\USER\Desktop\pix2pix-master\pix2pix-master\datasets\\facades\\val',
                    help='path to val dataset')

parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')

parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--val_batch_size', type=int, default=64, help='input batch size')

parser.add_argument('--origin_size', type=int, default=286, help='size of the original input image')
parser.add_argument('--image_size', type=int, default=256, help='size of the cropped input image to network')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')

parser.add_argument('--inch', type=int, default=3, help='size of the input channels')
parser.add_argument('--outch', type=int, default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)

parser.add_argument('--nepochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')

parser.add_argument('--lambdaGAN', type=float, default=1, help='lambdaGAN, default=1')
parser.add_argument('--lambdaIMG', type=float, default=100, help='lambdaIMG, default=100')

parser.add_argument('--weight_decay', type=float, default=0.0004, help='weight decay, default=0.0004')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--rmsprop', action='store_true', help='Whether to use adam (default is rmsprop)')

parser.add_argument('--netG', default='samples\\netG_epoch_199.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='samples\\netD_epoch_199.pth', help="path to netD (to continue training)")

parser.add_argument('--out_folder', default='samples', help='folder to output images and model checkpoints')
parser.add_argument('--log_interval', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--val_interval', type=int, default=10, help='interval for evauating(generating) images from valDataroot')

def get_config():
    config = parser.parse_args()
    return config