import argparse

def get_config():
    parser = argparse.ArgumentParser()

    # data preprocessing hyper-parameters
    parser.add_argument('--dataroot', type=str, default="dataset")
    parser.add_argument('--content_size', type=int, default=256)
    parser.add_argument('--style_path', type=str, default='styles/ghibli.png')
    parser.add_argument('--style_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=4)

    # training hyper-parameters
    parser.add_argument('--nepochs', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=250, help="step interval")
    parser.add_argument('--sample_interval', type=int, default=2, help="epoch interval to sample output image")
    parser.add_argument('--checkpoint_interval', type=int, default=10, help="epoch interval to sample model checkpoint")

    parser.add_argument('--style_weight', type=float, default=1e10, help="weight for style loss, default is 1e10")
    parser.add_argument('--content_weight', type=float, default=1e5, help="weight for content loss, default is 1e5")

    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--training', type=str, default="")
    parser.add_argument('--sample_folder', default=None)

    config = parser.parse_args()

    return config