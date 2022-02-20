from time import sleep
from src.data_helper import DataProcessor
from src.recover.greedy import Greedy
from src.sorting.environment import Solution, State as GameState
from src.sorting.environment import Environment as GameEnvironment
from src.sorting.MCTS import MCTS as GameMCTS
from src.sorting.TreeSearch import TreeSearch
from src.sorting.Standard import Standard
from src.sorting.Astar import Astar
from utils import *
from configs import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", type=str, default="output/states/")
    parser.add_argument("-f", "--item-path", type=str, default="natural_5.bin")
    parser.add_argument("--model-name", type=str, default="model_2_0.pt")
    parser.add_argument("--output-path", type=str, default="./output/recovered_images/")
    parser.add_argument(
        "-a", "--algorithm", type=str, default="standard", help="algorithm to use"
    )
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-t", "--sleep", type=float, default=0.0)
    parser.add_argument("-k", "--skip", type=int, default=10)

    parser.add_argument("-r", "--rate", type=str, default="8/2")
    parser.add_argument("-c", "--max_choose", type=int, default=128)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    state = DataProcessor.load_item_from_binary_file(args.state_path + args.item_path)
    game_state = GameState(state.original_blocks, state.inverse)
    game_state.max_choose = args.max_choose
    game_state.save_image()
    if state.choose_swap_ratio:
        env = GameEnvironment(
            r1=state.choose_swap_ratio[0], r2=state.choose_swap_ratio[1]
        )
    else:
        env = GameEnvironment(
            r1=int(args.rate.split("/")[0]), r2=int(args.rate.split("/")[1])
        )

    mcts = GameMCTS(env, n_sim=20, c_puct=100, verbose=False)
    astar = Astar(env, verbose=False)
    stree = TreeSearch(env, depth=4, breadth=2, verbose=False)
    standard = Standard(env, verbose=False)

    start = time.time()
    """ initialize the model """
    game_state.save_image()
    game_state.original_distance = env.get_mahattan_distance(game_state)
    standard.set_cursor(game_state)
    solution = Solution(shape=game_state.shape)
    solution.save_angles(game_state.inverse)

    # Starting...
    print(
        "Num matched: ",
        game_state.shape[0] * game_state.shape[1]
        - env.get_haminton_distance(game_state),
    )
    while not env.get_game_ended(game_state):
        if args.algorithm == "mcts":
            action = mcts.get_action(game_state)
        elif args.algorithm == "greedy":
            action = Greedy.get_action(game_state)
        elif args.algorithm == "astar":
            action = astar.get_action(game_state)
        elif args.algorithm == "stree":
            action = stree.get_action(game_state)
        elif args.algorithm == "standard":
            action = standard.get_action(game_state)

        game_state = env.step(game_state, action)
        solution.store_action(action)
        
        if args.verbose and game_state.depth % args.skip == 0:
            game_state.save_image()
            distance = env.get_mahattan_distance(game_state)
            print("{}, {}".format(game_state.depth, action))
            print("Overall Distance: {}".format(distance))
            print("Time: {}".format(time.time() - start))
            sleep(args.sleep)
        # if args.algorithm != "standard" and game_state.n_chooses + game_state.shape[1] > game_state.max_choose:
        #     args.algorithm = "standard"
        #     standard.set_cursor(game_state)
    game_state.show()
    game_state.save_image()
    solution.save_to_json('output/solutions', args.item_path.split('.')[0] + '.json')
    solution.save_text('output/solutions', args.item_path.split('.')[0] + '.txt')
    print(
        "Num matched: ",
        game_state.shape[0] * game_state.shape[1]
        - env.get_haminton_distance(game_state),
    )
    print("Time: {}".format(time.time() - start))


if __name__ == "__main__":
    main()
