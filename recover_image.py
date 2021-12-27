import cv2
import numpy as np
from models.ProNet2 import ProNet2
from models.SimpleProNet import ProNet as SimpleProNet
# from models.VGG import ProNet
from models.VGG import VGG
from src.data_helper import DataProcessor
from src.MCTS import MCTS
from src.greedy import Greedy
from utils import *
from src.environment import Environment, State
from configs import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default='./input/shuffle_images/')
    parser.add_argument('--model-path', type=str, default='./trainned_models/')
    parser.add_argument('--model-name', type=str, default='model_2_0.pt')
    parser.add_argument('--output-path', type=str, default='./output/recovered_images/')
    parser.add_argument('-f', '--file-name', type=str, default='sage_1.png')
    parser.add_argument('-s', '--block-size', type=int, default=(32, 32))
    parser.add_argument('-d', '--block-dim', type=tuple_type, default=(3,4))
    parser.add_argument('--image-size-out', type=int, default=(512, 512))   
    parser.add_argument('-a', '--algorithm', type=str, default='greedy')
    parser.add_argument('-v', '--verbose', type=bool, default=True)
    parser.add_argument('-t', '--threshold', type=float, default=0.9999)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    original_image = cv2.imread(args.image_path + args.file_name)
    state = State(original_image, args.block_size, args.block_dim)
    model = VGG('VGG9')
    model.load_checkpoint(0, 3344)
    model.eval()
    
    env = Environment()
    mcts = MCTS(env, model, n_sim=8, 
                c_puct=1.5, threshold=args.threshold,
                n_bests=5, verbose=args.verbose)
    greedy = Greedy(env, model, verbose=args.verbose, 
                    n_bests=1, threshold=0.95)
    
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
        
    new_blocks = np.zeros(state.original_blocks.shape, dtype=np.uint8)
    for i in range(args.block_dim[0]):
        for j in range(args.block_dim[1]):
            x, y, angle = state.inverse[i][j]
            new_blocks[i][j] = np.rot90(state.original_blocks[x][y], k=angle)
    recovered_image = DataProcessor.merge_blocks(new_blocks)

    end = time.time()
    
    # recovered_image = cv2.resize(recovered_image, args.image_size_out)
    cv2.imwrite(args.output_path + args.file_name, recovered_image)
    print('Recovered image saved at: ' + args.output_path + args.file_name)
    print('Time: {}'.format(end - start))
    DataProcessor.save_item_to_binary_file(
        state,
        'output/states/' + 'state' + '.bin') # _' + args.file_name
    
if __name__ == "__main__":
    main()