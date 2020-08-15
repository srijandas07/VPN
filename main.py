#----------------------------------------
# VPN main file to start the training
# Created By Srijan Das and Saurav Sharma
#----------------------------------------

import argparse
import os
import sys
import yaml
from utils import read_yaml, map_yml_to_args
from train import trainer

# base args file with default values for vpn training and testing
def parse_args():
    parser = argparse.ArgumentParser(description='Video Pose Network')

    # model parameters
    parser.add_argument('--dataset', default='ntu60', type=str, choices=['ntu60', 'ntu120','smarthomes','nucla'], help='training dataset')
    parser.add_argument('--epochs', default=250, type=int, help='max mumber of epochs for training')
    parser.add_argument('--num_gpus', default=0, nargs='+', type=int, help='gpu ids for training')
    parser.add_argument('--model_name', default='vpn', type=str, choices=['vpn', 'i3d'], help='Model to use for training/validation')
    parser.add_argument('--part', default='full_body', type=str, choices=['full_body','left_part','right_part'], help='part of the body to use for training')
    parser.add_argument('--use_gpu', default=True, help='GPU to use for training/validation')
    parser.add_argument('--multi_gpu', default=True, help='Use multiple GPUs for training/validation')
    parser.add_argument('--batch_size', default=4, help='set batch size')
    parser.add_argument('--nw', default=16, type=int, help='number of worker to load data')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum to use for training')
    parser.add_argument('--lr', default=0.01, type=float, help='lr to use for training')
    parser.add_argument('--weights_loc', default="./checkpoint", type=str, help='location to save the weights of the model')
    parser.add_argument('--optim', default='SGD', type=str, choices=['SGD', 'Adam'])
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('--n_dropout', default=0.3, type=float, help='dropout to use in GCNN')
    parser.add_argument('--multi_proc', default=True, help='Use multiprocessing for training')

    # dataset parameters
    parser.add_argument('--num_classes', default=60, type=int, help='number of action classes')
    parser.add_argument('--protocol', default='cv', type=str, help='training/validation protocol for different datasets')
    parser.add_argument('--num_nodes', default=25, type=int, help='number of graph nodes to consider for a given pose data')
    parser.add_argument('--stack_size', default=16, type=int, help='clip width for training/testing')
    parser.add_argument('--num_neurons', default=64, type=int, help='number of nodes in GCNN')
    parser.add_argument('--timesteps', default=16, type=int, help='video clip size')

    # GCNN parameters
    parser.add_argument('--sym_norm', action='store_false', help='Symmetric Normalization flag for Graph Conv')
    parser.add_argument('--alpha', default=5, type=int, help='Edge weights for direct node connections')
    parser.add_argument('--beta', default=2, type=int, help='Edge weights for indirect node connections')
    parser.add_argument('--num_features', default=3, type=int, help='Initial feature width')
    parser.add_argument('--num_filters', default=2, type=int, help='Number of Filters for GCNN conv operation')

    # loss parameters
    parser.add_argument('--action_wt', default=99.9, type=float, help='weight for action recognition loss')
    parser.add_argument('--embed_wt', default=0.1, type=float, help='weight for feature embedding loss')

    # logger/metric parameters
    parser.add_argument('--monitor', default='val_loss', type=str, help='Loss to monitor in the logger')
    parser.add_argument('--factor', default=0.1, type=float, help='logger factor')
    parser.add_argument('--patience', default=5, type=int, help='number of epochs to wait before reducing LR')

    args = parser.parse_args()
    return args

# args file loaded by default and overloaded by config yaml file
def generate_config():
    args = parse_args()
    
    # overlay default args with dataset specific args
    cfg_file = f"{args.model_name}_{args.dataset}.yml"
    print(f'--Loading config file--')
    data_cfg = read_yaml(os.path.join('configs',cfg_file))

    # overwrite default args with dataset and model specific values
    args = map_yml_to_args(args, data_cfg)
    print('>>>>>>>>>>>>>>>>>>>>>      Model Config       <<<<<<<<<<<<<<<<<<<<')
    print(args)
    print('--Config file loaded and updated--')
    return args

# main function to start the process
def main():
    args = generate_config()
    
    # train models ..
    trainer(args)

    # TO DO list
    # 1. Create a separate file for models to include multiple video models
    # 2. Reorganize Data Loader to accomdate other datasets


if __name__ == '__main__':
    main()