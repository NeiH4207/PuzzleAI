import cv2
import numpy as np
from models.ProNet import ProNet
from src.data_helper import DataProcessor
from src.MCTS import MCTS
from utils import *
from src.environment import Environment, State
from matplotlib import pyplot as plt

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default='./input/shuffle_images/')
    parser.add_argument('--model-path', type=str, default='./trainned_models/')
    parser.add_argument('--model-name', type=str, default='model_2_0.pt')
    parser.add_argument('--output-path', type=str, default='./output/recovered_images/')
    parser.add_argument('-f', '--file-name', type=str, default='cayuga_1.png')
    parser.add_argument('--block-size', type=int, default=(64, 64))
    parser.add_argument('--block-dim', type=int, default=(2, 2))
    parser.add_argument('--image-size-out', type=int, default=(512, 512))
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    original_image = cv2.imread(args.image_path + args.file_name)
    state = State(original_image, args.block_size, args.block_dim)
    model = ProNet(state.image_size)
    model.load_checkpoint(1, 1300)
    
    model.eval()
    
    env = Environment()
    mcts = MCTS(env, model, n_sim=20, c_puct=2)
    
    start = time.time()
    
    while state.depth < state.max_depth:
        action, info = mcts.get_probs(state, 0)
        state = env.step(state, action)
        state.save_image('recovered_' + args.file_name)
        print('Done step: {} / {}'.format(state.depth, state.max_depth))
        # print('Probability: {}'.format(prob))
        print(info[0][0])
        
    original_blocks = np.zeros(state.original_blocks.shape, dtype=np.uint8)
    for i in range(args.block_dim[0]):
        for j in range(args.block_dim[1]):
            x, y, angle = state.inverse[i][j]
            original_blocks[i][j] = np.rot90(state.original_blocks[x][y], k=angle)
    recovered_image = DataProcessor.merge_blocks(original_blocks)
    
    end = time.time()
    
    recovered_image = cv2.resize(recovered_image, args.image_size_out)
    cv2.imwrite(args.output_path + args.file_name, recovered_image)
    print('Recovered image saved at: ' + args.output_path + args.file_name)
    print('Time: {}'.format(end - start))
    
if __name__ == "__main__":
    main()