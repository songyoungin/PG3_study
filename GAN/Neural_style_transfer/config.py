import argparse

def get_config():
    parser = argparse.ArgumentParser()

    # data preprocessing hyper-parameters
    parser.add_argument('--content', type=str, default='png\\content.png')
    parser.add_argument('--style', type=str, default='png\\style.png')
    parser.add_argument('--max_size', type=int, default=400)

    # training hyper-parameters
    parser.add_argument('--nepochs', type=int, default=2000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--sample_interval', type=int, default=250)
    parser.add_argument('--style_weight', type=float, default=100)
    parser.add_argument('--lr', type=float, default=0.003)

    parser.add_argument('--out_folder', default=None)

    config = parser.parse_args()

    return config