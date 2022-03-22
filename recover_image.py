''' python3 recover_image.py -f natural_1 -v -m '''
import cv2
import numpy as np
from models.VGG import VGG
from src.data_helper import DataProcessor
from src.recover.MCTS import MCTS
from src.recover.greedy import Greedy
from utils import *
from src.recover.environment import Environment, State
from configs import *
import argparse
from src.screen import Screen
seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game-info-path', type=str, default='./input/game_info/')
    parser.add_argument('--model-path', type=str, default='./trainned_models/')
    parser.add_argument('--model-name', type=str, default='model')
    parser.add_argument('--output-path', type=str, default='./output/recovered_images/')
    parser.add_argument('-f', '--file-name', type=str, default='matrix_4x4_test_2')
    parser.add_argument('--image-size-out', type=int, default=(512, 512))   
    parser.add_argument('-s', '--block-size', type=int, default=(32, 32))
    parser.add_argument('-a', '--algorithm', type=str, default='greedy')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--simple', action='store_true')
    parser.add_argument('--ref', action='store_true')
    parser.add_argument('--ref-dir', type=str, default=None)
    parser.add_argument('-t', '--threshold', type=float, default=1.0)
    parser.add_argument('-j', '--n_jumps', type=float, default=0)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    state = State(block_size=args.block_size)
    state.load_from_json(file_name=args.file_name + '.json')
    state.make()
    state.file_name = args.file_name.split('.')[0]
    model = VGG('VGG7')
    model.load(0, 1000, args.model_name)
        
    model.eval()
    env = Environment()
    if args.algorithm == 'greedy':
        algo = Greedy(env, model, verbose=args.verbose, 
                    n_bests=4, threshold=args.threshold)
    elif args.algorithm == 'mcts':
        algo = MCTS(env, model, n_sim=3, 
                c_puct=1, threshold=args.threshold,
                n_bests=3, verbose=False)
    else:
        raise Exception('Unknown algorithm')
    
    print('Start recovering...')
    print('Threshold:', args.threshold)
    
    start = time.time()
    
    if args.simple:
        state.to_simple_mode()
        screen = Screen(state)
        screen.render(state)
        state = screen.start_2(env, state)
    elif args.ref:
        # png image
        ref_img = cv2.imread(args.ref_dir)
        state.to_ref_mode(ref_img)
        state.depth = state.max_depth
        screen = Screen(state)
        screen.render(state)
        state = screen.start_3(env, state, algo)
    else:
        screen = Screen(state)
        screen.render(state)
        state = screen.start(env, state, algo)
        

    end = time.time()
    
    print('Time: {}'.format(end - start))
    DataProcessor.save_item_to_binary_file(
        state.small_copy(),
        'output/states/' + state.file_name + '.bin') 
    
if __name__ == "__main__":
    main()