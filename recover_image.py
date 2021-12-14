import cv2
import numpy as np
from models.SimpleProNet import ProNet as SimpleProNet
from models.ProNet import ProNet
from src.data_helper import DataProcessor
from src.MCTS import MCTS
from src.greedy import Greedy
from utils import *
from src.environment import Environment, State
from configs import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default='./input/original_images/')
    parser.add_argument('--model-path', type=str, default='./trainned_models/')
    parser.add_argument('--model-name', type=str, default='model_2_0.pt')
    parser.add_argument('--output-path', type=str, default='./output/recovered_images/')
    parser.add_argument('-f', '--file-name', type=str, default='natural_7.jpg')
    parser.add_argument('--block-size', type=int, default=(64, 64))
    parser.add_argument('-bd', '--block-dim', type=tuple_type, default=(7, 7))
    parser.add_argument('--image-size-out', type=int, default=(512, 512))   
    parser.add_argument('-a', '--algorithm', type=str, default='mcts')
    parser.add_argument('-v', '--verbose', type=bool, default=True)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    original_image = cv2.imread(args.image_path + args.file_name)
    
    original_image = DataProcessor.split_image_to_blocks(original_image, args.block_dim, outlier_rate=0.3)
    original_image = DataProcessor.merge_blocks(original_image)
    print(original_image.shape)
    cv2.imwrite('output/sample.png', original_image)
    
    state = State(original_image, args.block_size, args.block_dim)
    model = ProNet((2 * args.block_size[0], 2 * args.block_size[1]))
    model.load_checkpoint(2, 1347)
    
    model.eval()
    
    env = Environment()
    mcts = MCTS(env, model, n_sim=8, c_puct=1.5, n_bests=1, verbose=args.verbose)
    greedy = Greedy(env, model, verbose=args.verbose, n_bests=1)
    
    start = time.time()
    
    while state.depth < state.max_depth:
        if args.algorithm == 'greedy':
            action, prob = greedy.next_action(state)
            state = env.step(state, action)
            print(prob)
        elif args.algorithm == 'mcts':
            action, info = mcts.get_probs(state, 0)
            state = env.step(state, action)
            # print('Probability: {}'.format(prob))
            print(info[0], info[2])
        if args.verbose:
            state.save_image()
        print('Done step: {} / {}'.format(state.depth, state.max_depth))
        print('Time: {}'.format(time.time() - start))
        
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