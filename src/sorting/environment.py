from copy import deepcopy
import os
import random
import cv2
import json
import numpy as np
from src.data_helper import DataProcessor
EPS = 1e-8

class Solution(object):
    
    def __init__(self, name="solution", shape=None) -> None:
        super().__init__()
        self.name = name
        self.angles = []
        self.num_selects = 0
        self.select_list = []
        self.swap_arrays = []
        self.curr_postion = (0, 0)
        self.shape = shape
    
    def int2hex(self, num):
        hex_code = '0123456789ABCDEF'
        return hex_code[num]
    
    def store_action(self, action):
        if action[0] == 'select':
            self.num_selects += 1
            self.select_list.append(self.int2hex(action[1][1]) + self.int2hex(action[1][0]))
            self.swap_arrays.append("")
            self.curr_postion = action[1]
        else:
            vx = (action[1][2] - action[1][0] + self.shape[0]) % self.shape[0]
            vy = (action[1][3] - action[1][1] + self.shape[1]) % self.shape[1]
            
            if vx != 0:
                if action[1][2] - 1 == action[1][0] or \
                    action[1][2] + self.shape[0] - 1 == action[1][0]:
                    self.swap_arrays[-1] += 'D'
                else:
                    self.swap_arrays[-1] += 'U'
            else:
                if action[1][3] - 1 == action[1][1] or \
                    action[1][3] + self.shape[1] - 1 == action[1][1]:
                    self.swap_arrays[-1] += 'R'
                else:
                    self.swap_arrays[-1] += 'L'
        # print(self.swap_arrays[-1])
        
    def save_angles(self, inverse):
        self.angles = []
        for i in range(len(inverse)):
            for j in range(len(inverse[i])):
                self.angles.append((4 - inverse[i][j][2]) % 4)
        # convert to string
        self.angles = np.array(self.angles, dtype=np.int32).tolist()
    
    def to_json(self):
        data = {
            "angles": ''.join([str(x) for x in self.angles]),
            "num_selects": self.num_selects,
            "swap_arrays": self.swap_arrays
        }
        # print(data)
        return data
    
    def save_to_json(self, path, file_name):
        # check if the path exists
        if not os.path.exists(path):
            os.makedirs(path)
            
        file_path = os.path.join(path, file_name)
            
        with open(file_path, "w") as f:
            json.dump(self.to_json(), f, indent=4)
            
    def save_text(self, path, file_name):
        # check if the path exists
        if not os.path.exists(path):
            os.makedirs(path)
            
        file_path = os.path.join(path, file_name)
            
        with open(file_path, "w") as f:
            f.write(''.join([str(x) for x in self.angles])+'\n')
            f.write(str(self.num_selects))
            f.write("\n")
            for i in range(self.num_selects):
                f.write(str(self.select_list[i]))
                f.write("\n")
                f.write(str(len(self.swap_arrays[i])))
                f.write("\n")
                f.write(str(self.swap_arrays[i]))
                f.write("\n")
        
    def load_from_json(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        self.angles = data["angles"]
        self.num_selects = data["num_selects"]
        self.swap_arrays = data["swap_arrays"]

class State:
    """
    Class for the state of the environment.
    """

    def __init__(self, original_blocks=None, inv_state=None):
        if original_blocks is None:
            pass
        else:
            self.inverse = inv_state
            self.original_blocks = original_blocks
            self.shape = original_blocks.shape[:2]
            self.depth = 0
            self.probs = [] 
            self.actions = []
            self.last_action = (0, 0)
            self.curr_position = None
            self.n_selects = 0
            self.max_select = self.shape[0] * self.shape[1] // 2
            self.n_swaps = 0
            self.mode = 'rgb'
            self.original_distance = 0
            self.parent_states = set()
            self.n_trues = 0
            self.exploration_rate = 1.0
            self.distance = 0
            self.reward = - np.inf
            self.make()
        
    def make(self):
        self.targets = np.zeros(self.shape, dtype=np.int32)
        self.inv_targets = np.zeros(self.shape, dtype=np.int32)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                x, y, angle = self.inverse[i][j]
                self.original_blocks[x][y] = np.rot90(self.original_blocks[x][y], 
                                                      k=angle)
                self.targets[x][y] = i * self.shape[1] + j
        self.select_actions = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.select_actions.append(('select',(i, j)))
        self.blocks = np.zeros((self.shape[0], self.shape[1], 64, 64, 3), dtype='uint8')
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.blocks[i][j] = cv2.resize(self.original_blocks[i][j],
                                            (64,64), interpolation=cv2.INTER_AREA)
        self.set_string_presentation()
        
    def copy(self):
        """
        Returns a copy of the state.
        """
        state = State()
        state.inverse = self.inverse
        state.original_blocks = self.original_blocks
        state.blocks = deepcopy(self.blocks)
        state.targets = deepcopy(self.targets)
        state.select_actions = self.select_actions
        state.shape = self.shape
        state.depth = self.depth
        state.probs = self.probs
        state.probs = deepcopy(self.probs)
        # state.actions = deepcopy(self.actions)
        state.last_action = deepcopy(self.last_action)
        state.mode = self.mode
        state.curr_position = self.curr_position
        state.n_selects = self.n_selects
        state.max_select = self.max_select
        state.n_swaps = self.n_swaps
        state.name = self.name
        state.original_distance = self.original_distance
        state.parent_states = deepcopy(self.parent_states)
        state.n_trues = self.n_trues
        state.exploration_rate = self.exploration_rate
        state.distance = self.distance
        state.reward = self.reward
        return state
       
    def string_presentation(self, items):
        return hash(str(items))
    
    def get_string_presentation(self):
        return self.name
    
    def set_string_presentation(self):
        self.name = self.string_presentation((self.targets, self.curr_position))
    
    def save_image(self, filename='sample.png'):
        new_img = DataProcessor.merge_blocks(self.blocks)
        cv2.imwrite('output/' + filename, new_img)
        
    def show(self):
        # print targets pretty format
        # print('-----------------------------------------------------')
        # if self.curr_position:
        #     print('curr_position: {}, value: {}'.format(self.curr_position, self.targets[self.curr_position[0]][self.curr_position[1]]))
        # for i in range(self.shape[0]):
        #     for j in range(self.shape[1]):
        #         print('{:>3d}'.format(self.targets[i][j]), end='')
        #     print(' | ', end='')
        #     for j in range(self.shape[1]):
        #         print('{:>3d}'.format(j + i * self.shape[1]), end='')
        #     print()
        print('-----------------------------------------------------')
        print('Number of selects: {}'.format(self.n_selects))
        print('Number of swaps: {}'.format(self.n_swaps))
        
class Environment():
    """
    Class for the environment.
    """
    def __init__(self, r1, r2, eta, name='recover_image'):
        self.name = name
        self.state = None
        self.r1 = r1 / (r1 + r2)
        self.r2 = r2 / (r1 + r2)
        self.reset()
        self.next_step = {}
        self.counter = {}
        self.eta = eta
        self.gamma = 1 / self.eta * 1.1

    def reset(self):
        return
    
    def get_strict_reward(self, state, action):
        reward = 0
        if action[0] == 'swap':
            x1, y1, x2, y2 = action[1]
            true_pos = (state.targets[x1][y1] // state.shape[1],\
                          state.targets[x1][y1] % state.shape[1])
            cost_1 = min(abs(true_pos[0] - x1), state.shape[0] - abs(true_pos[0] - x1)) + \
                        min(abs(true_pos[1] - y1), state.shape[1] - abs(true_pos[1] - y1))
            cost_2 = min(abs(true_pos[0] - x2), state.shape[0] - abs(true_pos[0] - x2)) + \
                        min(abs(true_pos[1] - y2), state.shape[1] - abs(true_pos[1] - y2))
            true_pos = (state.targets[x2][y2] // state.shape[1],\
                        state.targets[x2][y2] % state.shape[1])
            cost_3 = min(abs(true_pos[0] - x2), state.shape[0] - abs(true_pos[0] - x2)) + \
                        min(abs(true_pos[1] - y2), state.shape[1] - abs(true_pos[1] - y2))
            cost_4 = min(abs(true_pos[0] - x1), state.shape[0] - abs(true_pos[0] - x1)) + \
                        min(abs(true_pos[1] - y1), state.shape[1] - abs(true_pos[1] - y1)) 
            # print(state.exploration_rate)
            # mahattan_distance = self.get_mahattan_distance(state)
            reward = (1 - state.exploration_rate) * (cost_1 - cost_2) + \
                state.exploration_rate  * (cost_3 - cost_4) / ((1000000 + state.targets[x2][y2]) / 1000000)\
                    - self.r2
        else:
            x, y = action[1]
            true_pos = (state.targets[x][y] // state.shape[1],\
                            state.targets[x][y] % state.shape[1])
            cost = min(abs(true_pos[0] - x), state.shape[0] - abs(true_pos[0] - x)) + \
                        min(abs(true_pos[1] - y), state.shape[1] - abs(true_pos[1] - y))
                
            reward = - self.r1 + cost * 0.00001
        return reward
    
    def get_reward(self, state, action):
        reward = 0
        if action[0] == 'swap':
            x1, y1, x2, y2 = action[1]
            true_pos = (state.targets[x1][y1] // state.shape[1],\
                          state.targets[x1][y1] % state.shape[1])
            cost_1 = min(abs(true_pos[0] - x1), state.shape[0] - abs(true_pos[0] - x1)) + \
                        min(abs(true_pos[1] - y1), state.shape[1] - abs(true_pos[1] - y1))
            cost_2 = min(abs(true_pos[0] - x2), state.shape[0] - abs(true_pos[0] - x2)) + \
                        min(abs(true_pos[1] - y2), state.shape[1] - abs(true_pos[1] - y2))
            true_pos = (state.targets[x2][y2] // state.shape[1],\
                        state.targets[x2][y2] % state.shape[1])
            cost_3 = min(abs(true_pos[0] - x2), state.shape[0] - abs(true_pos[0] - x2)) + \
                        min(abs(true_pos[1] - y2), state.shape[1] - abs(true_pos[1] - y2))
            cost_4 = min(abs(true_pos[0] - x1), state.shape[0] - abs(true_pos[0] - x1)) + \
                        min(abs(true_pos[1] - y1), state.shape[1] - abs(true_pos[1] - y1)) 
                         
            # mahattan_distance = self.get_mahattan_distance(state)
            reward = (cost_1 - cost_2) + (cost_3 - cost_4) - self.r2
        else:
            x, y = action[1]
            true_pos = (state.targets[x][y] // state.shape[1],\
                            state.targets[x][y] % state.shape[1])
            cost = min(abs(true_pos[0] - x), state.shape[0] - abs(true_pos[0] - x)) + \
                        min(abs(true_pos[1] - y), state.shape[1] - abs(true_pos[1] - y))
                
            reward = - self.r1
        return reward
    
    def get_G_reward(self, state):
        haminton_distance = self.get_haminton_distance(state)
        mahattan_distance = self.get_mahattan_distance(state)
        reward = (state.original_distance - mahattan_distance) / state.original_distance / \
             np.log(1 + - state.n_selects * self.r1 - state.n_swaps * self.r2)
        return reward
    
    def get_haminton_distance(self, state):
        total_diff = 0
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                position = state.targets[i][j]
                _x, _y = position // state.shape[1], position % state.shape[1]
                if i != _x or j != _y:
                    total_diff += 1
        return total_diff
    
    def get_mahattan_distance(self, state):
        total_diff = 0
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                x1, y1 = i, j
                true_pos = (state.targets[x1][y1] // state.shape[1],\
                          state.targets[x1][y1] % state.shape[1])
                cost = min(abs(true_pos[0] - x1), state.shape[0] - abs(true_pos[0] - x1)) + \
                            min(abs(true_pos[1] - y1), state.shape[1] - abs(true_pos[1] - y1))
                total_diff += cost
        return total_diff
    
    def get_game_ended(self, state):
        mahattan_distance = self.get_mahattan_distance(state)
        if state.n_selects > state.max_select:
            return state.original_distance - mahattan_distance + \
                state.n_selects * self.r1 + state.n_swaps * self.r2
        if mahattan_distance == 0:
            return state.original_distance + \
                state.n_selects * self.r1 + state.n_swaps * self.r2
        
        return False
                
    def get_available_actions(self, state, repeat=False):
        actions = []
        if state.curr_position != None:
            # up, right, down, left
            dx = [1, 0, -1, 0] 
            dy = [0, 1, 0, -1]
            # targets = deepcopy(state.targets)
            x1, y1 = state.curr_position
            value = x1 * state.shape[1] + y1
            if value != state.targets[x1][y1]:
                
                # for x1 in range(state.shape[0]):
                #     for y1 in range(state.shape[1]):
                for i in range(4):
                    x2 = (x1 + dx[i]) % state.shape[0]
                    y2 = (y1 + dy[i]) % state.shape[1]
                    # if not repeat and state.last_action[1] == (x2, y2, x1, y1):
                    #     continue
                    # targets[x1][y1], targets[x2][y2] = \
                    #     deepcopy([state.targets[x2][y2], state.targets[x1][y1]])
                    # s = state.string_presentation((targets, (x2, y2)))
                    # if s in self.counter:
                    #     continue
                    actions.append(('swap', (x1, y1, x2, y2)))
        if state.n_selects < state.max_select:
            for action in state.select_actions:
                x1, y1 = action[1]
                value = x1 * state.shape[1] + y1
                if action[1] != state.curr_position and \
                    value != state.targets[x1][y1]:
                    actions.append(action)
        return actions
    
    def step(self, state, action):
        """
        Performs an action in the environment.
        """
        # s_name = state.get_string_presentation()
        # self.counter[s_name] = 1
        next_s = state.copy()
        if action[0] == 'swap':
            x1, y1, x2, y2 = action[1]
            next_s.blocks[x1][y1], next_s.blocks[x2][y2] = \
                deepcopy([next_s.blocks[x2][y2], next_s.blocks[x1][y1]])
            next_s.targets[x1][y1], next_s.targets[x2][y2] = \
                deepcopy([next_s.targets[x2][y2], next_s.targets[x1][y1]])
            next_s.n_swaps += 1
            pos = (x2, y2)
            next_s.curr_position = pos
            next_s.n_trues = 0
            pos = (next_s.n_trues // next_s.shape[1], next_s.n_trues % next_s.shape[1])
            while next_s.targets[pos[0]][pos[1]] == next_s.n_trues:
                next_s.n_trues += 1
                pos = (next_s.n_trues // next_s.shape[1], next_s.n_trues % next_s.shape[1])
                if pos[0] >= next_s.shape[0]:
                    break
        else:
            position = action[1]
            next_s.curr_position = position
            next_s.n_selects += 1
        next_s.depth += 1
        # state.actions.append(action)
        next_s.set_string_presentation()
        next_s.parent_states.add(state.get_string_presentation())
        next_s.last_action = action
        reward = self.get_reward(state, action)
        if reward > state.reward:
            next_s.exploration_rate = min(0.95, next_s.exploration_rate * self.gamma)
        elif reward < state.reward:
            next_s.exploration_rate = max(0.05, next_s.exploration_rate * self.eta)
        next_s.reward = reward
        return next_s
    