from time import sleep
import cv2
import numpy as np
from models.ProNet2 import ProNet2
from models.SimpleProNet import ProNet as SimpleProNet
# from models.VGG import ProNet
from models.VGG import VGG
from src.data_helper import DataProcessor
from src.MCTS import MCTS
from src.greedy import Greedy
from src.game import State as GameState
from src.game import Environment as GameEnvironment
from src.MCTS_2 import MCTS as GameMCTS
from src.TreeSearch import TreeSearch
from src.Astar import Astar
from utils import *
from src.environment import Environment, State
from configs import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-path', type=str, default='output/states/')
    parser.add_argument('--item-path', type=str, default='natural_1.bin')
    parser.add_argument('--model-name', type=str, default='model_2_0.pt')
    parser.add_argument('--output-path', type=str, default='./output/recovered_images/')
    parser.add_argument('-a', '--algorithm', type=str, default='astar')
    parser.add_argument('-v', '--verbose', type=bool, default=False)
    parser.add_argument('-r', '--rate', type=str, default='8/2')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    state = DataProcessor.load_item_from_binary_file(
        args.state_path + args.item_path)
    game_state = GameState(state.original_blocks, state.inverse)
    game_state.save_image()
    env = GameEnvironment(r1 = int(args.rate.split('/')[0]),
                          r2 = int(args.rate.split('/')[1]))
    mcts = GameMCTS(env, n_sim=20, 
                c_puct=1, verbose=False)
    astar = Astar(env, verbose=False)
    stree = TreeSearch(env, depth=6, breadth=2, verbose=False)
    start = time.time()
    game_state.save_image()
    print('True pieces: ', game_state.shape[0] * game_state.shape[1] - env.get_haminton_distance(game_state))
    while not env.get_game_ended(game_state, False):
        if args.algorithm == 'mcts':
            action = mcts.get_action(game_state)
        elif args.algorithm == 'greedy':
            action = Greedy.get_action(game_state)
        elif args.algorithm == 'astar':
            action = astar.get_action(game_state)
        elif args.algorithm == 'stree':
            action = stree.get_action(game_state)
        
        game_state = env.step(game_state, action)
        if args.verbose or True:
            game_state.save_image()
        # print("{}, {}".format(game_state.depth, action))
        # print("Overall Distance: {}".format(env.get_mahatan_distance(game_state)))
        # print('Time: {}'.format(time.time() - start))
        # sleep(0.2)
    game_state.show()
    game_state.save_image()
    print('True pieces: ', game_state.shape[0] * game_state.shape[1] - env.get_haminton_distance(game_state))
    print('Time: {}'.format(time.time() - start))
if __name__ == "__main__":
    main()