import _pickle as cPickle
from numpy.core.fromnumeric import argmax
from numpy.core.shape_base import block
from models.dla import DLA
from src.data_helper import DataHelper
from configs import configs, img_configs
import cv2
import numpy as np
from src.trainer import Trainer
from models.vgg import VGG
from models.ProNet import ProNet, dotdict
import torch.nn as nn
from copy import deepcopy as copy
from src.data_helper import DataProcessor
from src.MCTS import MCTS
from utils import *
from src.environment import Environment, State

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default='./input/shuffle_images/')
    parser.add_argument('--model-path', type=str, default='./trainned_models/')
    parser.add_argument('--model-name', type=str, default='model_2_0.pt')
    parser.add_argument('--output-path', type=str, default='./output/recovered_images/')
    parser.add_argument('--file-name', type=str, default='mcfaddin_2.png')
    parser.add_argument('--block-size', type=int, default=(16, 16))
    parser.add_argument('--block-dim', type=int, default=(2, 2))
    parser.add_argument('--image-size-out', type=int, default=(512, 512))
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    block_size = args.block_size
    block_dim = args.block_dim
    image_size = block_size[0] * block_dim[0], block_size[1] * block_dim[1]
    original_image = cv2.imread(args.image_path + args.file_name)
    original_blocks = DataProcessor.split_image_to_blocks(original_image, block_dim)
    blocks = DataProcessor.split_image_to_blocks(original_image, block_dim)
    blocks = [cv2.resize(block, block_size) for block in blocks]
    dropped_blocks, lost_blocks, lost_list = DataProcessor.random_drop_blocks(blocks, 1)
    
    block_length = block_dim[0] * block_dim[1]
    dropped_index_img_blocks = np.array([np.ones(block_size) if i in lost_list else np.zeros(block_size)
                                for i in range(block_length)])
    
    index_images = []
    for index in range(0, block_length):
        index_blocks = [np.zeros(block_size) if i != index else np.ones(block_size) 
                        for i in range(block_length)]
        index_image = DataProcessor.merge_blocks(index_blocks, block_dim)
        index_images.append(index_image)
        
    model = ProNet(image_size)
    model.load(2, 100)
    model.eval()
    
    env = Environment()
    mcts = MCTS(env, model, n_sim=10, c_puct=3)
    
    state = dotdict({
        'blocks': blocks,
        'dropped_blocks': dropped_blocks,
        'lost_blocks': lost_blocks,
        'lost_list': lost_list,
        'dropped_index_img_blocks': dropped_index_img_blocks,
        'index_images': index_images,
        'block_size': block_size,
        'block_dim': block_dim,
        'num_blocks': len(blocks),
        'image_size': image_size,
        'depth': 0,
        'max_depth': len(lost_list),
        'probs': [1.0],
        'actions': [],
        'orders': [0] * len(blocks),
        'angles': [0] * len(blocks),
        'mode': 1,
    })
    
    state = State(state)
    while state.depth < state.max_depth:
        action, _ = mcts.get_probs(state, 0)
        state = env.step(state, action)
        state.save_image('recovered_' + args.file_name)
        print('Done step:', state.depth + 1)
        
    original_blocks = [np.rot90(original_blocks[i], k=state.angles[i]) for i in state.orders]
    recovered_image = DataProcessor.merge_blocks(
        original_blocks, block_dim, mode=1)
    
    recovered_image = cv2.resize(recovered_image, args.image_size_out)
    cv2.imwrite(args.output_path + args.file_name, recovered_image)
    print('Recovered image saved at: ' + args.output_path + args.file_name)
    
if __name__ == "__main__":
    main()