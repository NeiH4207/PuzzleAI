from copy import deepcopy
import random
import cv2

import numpy as np
from src.data_helper import DataProcessor
EPS = 1e-8

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
            self.n_chooses = 0
            self.max_choose = self.shape[0] * self.shape[1] // 2
            self.n_swaps = 0
            self.mode = 'rgb'
            self.make()
        
    def make(self):
        self.matrix = np.zeros(self.shape, dtype=np.int8)
        self.inv_matrix = np.zeros(self.shape, dtype=np.int8)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                x, y, angle = self.inverse[i][j]
                self.original_blocks[x][y] = np.rot90(self.original_blocks[x][y], 
                                                      k=angle)
                self.matrix[x][y] = i * self.shape[1] + j
        self.choose_actions = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.choose_actions.append(('choose',(i, j)))
        self.blocks = deepcopy(self.original_blocks)
        self.save_image('sample.png')
        self.set_string_presentation()
        
    
     
    def copy(self):
        """
        Returns a copy of the state.
        """
        state = State()
        state.inverse = self.inverse
        state.original_blocks = self.original_blocks
        state.blocks = deepcopy(self.blocks)
        state.matrix = deepcopy(self.matrix)
        state.choose_actions = self.choose_actions
        state.shape = self.shape
        state.depth = self.depth
        state.probs = self.probs
        state.probs = deepcopy(self.probs)
        state.actions = deepcopy(self.actions)
        state.last_action = deepcopy(self.last_action)
        state.mode = self.mode
        state.curr_position = self.curr_position
        state.n_chooses = self.n_chooses
        state.max_choose = self.max_choose
        state.n_swaps = self.n_swaps
        state.name = self.name
        return state
       
    def string_presentation(self, items):
        return hash(str(items))
    
    def get_string_presentation(self):
        return self.name
    
    def set_string_presentation(self):
        self.name = self.string_presentation((self.matrix, self.curr_position))
    
    def save_image(self, filename='sample.png'):
        new_img = DataProcessor.merge_blocks(self.blocks, 'rgb')
        cv2.imwrite('output/' + filename, new_img)
        
    def show(self):
        # print matrix pretty format
        print('-----------------------------------------------------')
        if self.curr_position:
            print('curr_position: {}, value: {}'.format(self.curr_position, self.matrix[self.curr_position[0]][self.curr_position[1]]))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                print('{:>3d}'.format(self.matrix[i][j]), end='')
            print(' | ', end='')
            for j in range(self.shape[1]):
                print('{:>3d}'.format(j + i * self.shape[1]), end='')
            print()
        print('-----------------------------------------------------')
        print('Number of chooses: {}'.format(self.n_chooses))
        print('Number of swaps: {}'.format(self.n_swaps))
        
class Environment():
    """
    Class for the environment.
    """
    def __init__(self, r1, r2, name='recover_image'):
        self.name = name
        self.state = None
        self.r1 = - r1 / (r1 + r2)
        self.r2 = - r2 / (r1 + r2)
        self.reset()
        self.next_step = {}
        self.counter = {}

    def reset(self):
        return
    
    def get_reward(self, state, action):
        reward = 0
        if action[0] == 'swap':
            x1, y1, x2, y2 = action[1]
            true_pos = (state.matrix[x1][y1] // state.shape[1],\
                          state.matrix[x1][y1] % state.shape[1])
            cost_1 = min(abs(true_pos[0] - x1), state.shape[0] - abs(true_pos[0] - x1)) + \
                        min(abs(true_pos[1] - y1), state.shape[1] - abs(true_pos[1] - y1))
            cost_2 = min(abs(true_pos[0] - x2), state.shape[0] - abs(true_pos[0] - x2)) + \
                        min(abs(true_pos[1] - y2), state.shape[1] - abs(true_pos[1] - y2))
            true_pos = (state.matrix[x2][y2] // state.shape[1],\
                        state.matrix[x2][y2] % state.shape[1])
            cost_3 = min(abs(true_pos[0] - x2), state.shape[0] - abs(true_pos[0] - x2)) + \
                        min(abs(true_pos[1] - y2), state.shape[1] - abs(true_pos[1] - y2))
            cost_4 = min(abs(true_pos[0] - x1), state.shape[0] - abs(true_pos[0] - x1)) + \
                        min(abs(true_pos[1] - y1), state.shape[1] - abs(true_pos[1] - y1)) 
                         
            reward = 0.51 * (cost_1 - cost_2) + cost_3 - cost_4 + self.r2
        else:
            x, y = action[1]
            true_pos = (state.matrix[x][y] // state.shape[1],\
                            state.matrix[x][y] % state.shape[1])
            cost_1 = min(abs(true_pos[0] - x), state.shape[0] - abs(true_pos[0] - x)) + \
                        min(abs(true_pos[1] - y), state.shape[1] - abs(true_pos[1] - y))
                
            reward = self.r1 + 0.0001 * cost_1
        # mn = - 2 - min(self.r1, self.r2)
        # mx = 2 + max(self.r1, self.r2)
        # return (reward - mn) / (mx - mn)
        return reward
    
    def get_haminton_distance(self, state):
        total_diff = 0
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                position = state.matrix[i][j]
                _x, _y = position // state.shape[1], position % state.shape[1]
                if i != _x or j != _y:
                    total_diff += 1
        return total_diff
    
    def get_mahatan_distance(self, state):
        total_diff = 0
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                x1, y1 = i, j
                true_pos = (state.matrix[x1][y1] // state.shape[1],\
                          state.matrix[x1][y1] % state.shape[1])
                cost = min(abs(true_pos[0] - x1), state.shape[0] - abs(true_pos[0] - x1)) + \
                            min(abs(true_pos[1] - y1), state.shape[1] - abs(true_pos[1] - y1))
                total_diff += cost
        return total_diff
    
    def get_game_ended(self, state, check_repeat=True):
        if state.n_chooses == state.max_choose * 2:
            return True
        # if check_repeat:
        #     if state.name in self.counter:
        #         if self.counter[state.name] > 3:
        #             return True
        total_diff = 0
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                position = state.matrix[i][j]
                _x, _y = position // state.shape[1], position % state.shape[1]
                if i != _x or j != _y:
                    total_diff += 1
        if total_diff == 0:
            return True
        return False
                
    def get_available_actions(self, state, repeat=False):
        actions = []
        if state.curr_position != None:
            # up, right, down, left
            dx = [1, 0, -1, 0] 
            dy = [0, 1, 0, -1]
            matrix = deepcopy(state.matrix)
            x1, y1 = state.curr_position
            value = x1 * state.shape[1] + y1
            if value != state.matrix[x1][y1]:
                
                # for x1 in range(state.shape[0]):
                #     for y1 in range(state.shape[1]):
                for i in range(4):
                    x2 = (x1 + dx[i]) % state.shape[0]
                    y2 = (y1 + dy[i]) % state.shape[1]
                    if not repeat and state.last_action[1] == (x2, y2, x1, y1):
                        continue
                #     matrix[x1][y1], matrix[x2][y2] = \
                #         deepcopy([state.matrix[x2][y2], state.matrix[x1][y1]])
                #     s = state.string_presentation((matrix, (x2, y2)))
                #     if s in self.counter:
                #         continue
                    actions.append(('swap', (x1, y1, x2, y2)))
        for action in state.choose_actions:
            x1, y1 = action[1]
            value = x1 * state.shape[1] + y1
            if action[1] != state.curr_position and \
                value != state.matrix[x1][y1]:
                actions.append(action)
        return actions
    
    def step(self, state, action):
        """
        Performs an action in the environment.
        """
        s_name = state.get_string_presentation()
        self.counter[s_name] = 1
        next_s = state.copy()
        if action[0] == 'swap':
            x1, y1, x2, y2 = action[1]
            next_s.blocks[x1][y1], next_s.blocks[x2][y2] = \
                deepcopy([next_s.blocks[x2][y2], next_s.blocks[x1][y1]])
            next_s.matrix[x1][y1], next_s.matrix[x2][y2] = \
                deepcopy([next_s.matrix[x2][y2], next_s.matrix[x1][y1]])
            next_s.n_swaps += 1
            pos = (x2, y2)
            next_s.curr_position = pos
        else:
            position = action[1]
            next_s.curr_position = position
            next_s.n_chooses += 1
        next_s.depth += 1
        # state.actions.append(action)
        next_s.set_string_presentation()
        next_s.last_action = action
        return next_s
    
    def soft_update_state(self, state, action):
        """
        Performs an action in the environment.
        """
        next_s = state.copy()
        if action[0] == 'swap':
            x1, y1, x2, y2 = action[1]
            next_s.matrix[x1][y1], next_s.matrix[x2][y2] = \
                deepcopy([next_s.matrix[x2][y2], next_s.matrix[x1][y1]])
            next_s.n_swaps += 1
            pos = (x2, y2)
            next_s.curr_position = pos
        else:
            position = action[1]
            next_s.curr_position = position
            next_s.n_chooses += 1
        next_s.depth += 1
        # state.actions.append(action)
        next_s.set_string_presentation()
        next_s.last_action = action
        return next_s