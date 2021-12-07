import _pickle as cPickle
from numpy.core.fromnumeric import argmax
from numpy.core.shape_base import block
from src.data_helper import DataHelper
from configs import configs, img_configs
import cv2
import numpy as np
from src.trainer import Trainer
from models.vgg import VGG
from models.ProNet import ProNet
import torch.nn as nn
from copy import deepcopy as copy

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default='./input/original_images/')
    parser.add_argument('--output-path', type=str, default='./input/shuffle_images')
    parser.add_argument('--file-name-in', type=str, default='mcfaddin_2.ppm')
    parser.add_argument('--file-name-out', type=str, default=None)
    parser.add_argument('--block-size', type=int, default=(256, 256))
    parser.add_argument('--block-dim', type=int, default=(2, 2))
    parser.add_argument('--mode', type=str, default='rgb')
        
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    DataProcess = DataHelper()
    block_size = args.block_size
    block_dim = args.block_dim
    image_size = block_size[0] * block_dim[0], block_size[1] * block_dim[1]
    image = cv2.imread(args.image_path + args.file_name_in)
        
    if args.mode == 'bw':
        image = DataProcess.convert_rgb_to_bw(image)
        
    image = cv2.resize(image, image_size)
    blocks = DataProcess.split_image_to_blocks(image, block_dim)
    blocks, labels = DataProcess.shuffle_blocks(blocks, rotate=False)
    image = DataProcess.merge_blocks(blocks, block_dim, mode=1)
    
    if args.file_name_out is None:
        args.file_name_out = args.file_name_in.split('.')[0] + '.png'
    cv2.imwrite(args.output_path + '/' + args.file_name_out, image)
    print('image saved in {}'.format(args.output_path + '/' + args.file_name_out))
    
if __name__ == "__main__":
    main()