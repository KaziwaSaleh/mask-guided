import argparse
import os
import yaml
import warnings
warnings.filterwarnings("ignore")
import Trainer
import Validator


def main(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in config.items():
        setattr(args, k, v)

    if not hasattr(args, 'exp_path'):
        args.exp_path = os.path.dirname(args.config)

    if args.mode == 'training':
        Trainer.trainer(args)
    else:
        Validator.validator(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mask Guided Gated Convolution for Amodal Content Completion.')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--mode', default='training', type=str)
    parser.add_argument('--launcher', default='pytorch', type=str)
    parser.add_argument('--load-pretrain', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--checkpoint_interval', type=int, default=1)
    parser.add_argument('--folder_name', type=str, default='masked_model')
    args = parser.parse_args()

    main(args)
