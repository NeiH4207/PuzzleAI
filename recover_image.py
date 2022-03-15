''' python3 recover_image.py -f natural_1 -v -m '''
from turtle import position
import cv2
import numpy as np
from models.ProNet2 import ProNet2
from models.dla import DLA
from models.SimpleProNet import ProNet as SimpleProNet
# from models.VGG import ProNet
from models.VGG import VGG
from src.data_helper import DataProcessor
from src.recover.MCTS import MCTS
from src.recover.greedy import Greedy
from utils import *
from src.recover.environment import Environment, State, GameInfo
from configs import *
import argparse
from src.screen import Screen

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game-info-path', type=str, default='./input/game_info/')
    parser.add_argument('--model-path', type=str, default='./trainned_models/')
    parser.add_argument('--model-name', type=str, default='model')
    parser.add_argument('--output-path', type=str, default='./output/recovered_images/')
    parser.add_argument('-f', '--file-name', type=str, default='8x8')
    parser.add_argument('--image-size-out', type=int, default=(512, 512))   
    parser.add_argument('-s', '--block-size', type=int, default=(32, 32))
    parser.add_argument('-a', '--algorithm', type=str, default='greedy')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-m', '--monitor', action='store_true')
    parser.add_argument('-t', '--threshold', type=float, default=1.0)
    parser.add_argument('-j', '--n_jumps', type=float, default=0)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    state = State(block_size=args.block_size)
    state.load_from_json(file_name=args.file_name + '.json')
    state.make()
    model = VGG('VGG7')
    model.load(0, 1580, args.model_name)
        
    model.eval()
    screen = Screen(state)
    screen.render(state)
    env = Environment()
    if args.algorithm == 'greedy':
        algo = Greedy(env, model, verbose=args.verbose, 
                    n_bests=4, threshold=args.threshold)
    elif args.algorithm == 'mcts':
        algo = MCTS(env, model, n_sim=8, 
                c_puct=1, threshold=args.threshold,
                n_bests=3, verbose=False)
    else:
        raise Exception('Unknown algorithm')
    
    print('Start recovering...')
    print('Threshold:', args.threshold)
    
    start = time.time()
    state = screen.start(env,state, algo)
        
    new_blocks = np.zeros(state.original_blocks.shape, dtype=np.uint8)
    for i in range(state.block_dim[0]):
        for j in range(state.block_dim[1]):
            x, y, angle = state.inverse[i][j]
            new_blocks[i][j] = np.rot90(state.original_blocks[x][y], k=angle)
    recovered_image = DataProcessor.merge_blocks(new_blocks)

    end = time.time()
    
    # recovered_image = cv2.resize(recovered_image, args.image_size_out)
    cv2.imwrite(args.output_path + args.file_name + '.png', recovered_image)
    print('Recovered image saved at: ' + args.output_path + args.file_name+ '.png')
    print('Time: {}'.format(end - start))
    DataProcessor.save_item_to_binary_file(
        state,
        'output/states/' + args.file_name.split('.')[0] + '.bin') # _' + args.file_name
    
if __name__ == "__main__":
    main()