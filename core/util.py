import numpy as np
import argparse
import torch
import os

def create_folders(args):
    for d in [args.data_dir, args.checkpoint_dir, args.csv_dir, args.figure_dir]:
        os.makedirs(d, exist_ok=True)

def set_random_seed(args):
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)
    random.seed(args.seed)

def get_args():
    # create args parser
    parser = argparse.ArgumentParser()
    # scenario
    parser.add_argument('--scenario', type=str, default='main')
    # model
    parser.add_argument('--pretrain_model', type=str, default='scratch')
    parser.add_argument('--model', type=str, default='fc')
    # initializer
    parser.add_argument('--initializer', type=str, default='scratch')
    # training
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    # data directory
    parser.add_argument('--checkpoint_dir', type=str, default=f'../data/checkpoint')
    parser.add_argument('--figure_dir', type=str, default=f'../data/figure')
    parser.add_argument('--data_dir', type=str, default=f'../data/data')
    parser.add_argument('--csv_dir', type=str, default=f'../data/csv')
    # cuda
    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda:0')
    else:
        parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    # parse args
    args = parser.parse_args()
    # other defaults
    create_folders(args)
    return args
