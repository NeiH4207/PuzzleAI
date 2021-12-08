from copy import deepcopy
import cv2

import numpy as np

from src.data_helper import DataProcessor

class State:
    """
    Class for the state of the environment.
    """

    def __init__(self, state):
        self.blocks = deepcopy(state.blocks)
        self.dropped_blocks = deepcopy(state.dropped_blocks)
        self.lost_blocks = deepcopy(state.lost_blocks)
        self.lost_list = deepcopy(state.lost_list)
        self.block_size = deepcopy(state.block_size)
        self.dropped_index_img_blocks = deepcopy(state.dropped_index_img_blocks)
        self.index_images = deepcopy(state.index_images)
        self.block_dim = deepcopy(state.block_dim)
        self.num_blocks = len(self.blocks)
        self.image_size = deepcopy(state.image_size)
        self.depth = deepcopy(state.depth)
        self.max_depth = deepcopy(state.max_depth)
        self.probs = deepcopy(state.probs)
        self.actions = deepcopy(state.actions)
        self.mode = deepcopy(state.mode)
        self.orders = deepcopy(state.orders)
        self.angles = deepcopy(state.angles)
        
        
    def copy(self):
        """
        Returns a copy of the state.
        """
        return State(self)
    
    def save_image(self, filename='sample.png'):
        new_img = DataProcessor.merge_blocks(self.dropped_blocks,
                                                          self.block_dim, self.mode)
        cv2.imwrite('output/' + filename, new_img)

class Environment():
    """
    Class for the environment.
    """
    def __init__(self, name='recover_image'):
        self.name = name
        self.state = None
        self.reset()

    def reset(self):
        return

    def step(self, state, action):
        """
        Performs an action in the environment.
        """
        block_id, index, angle = action
        next_s = state.copy()
        next_s.dropped_blocks[index] = np.rot90(next_s.lost_blocks[block_id], k = angle)
        next_s.lost_list.remove(index)
        next_s.lost_blocks[block_id] = None
        next_s.actions.append(action)
        next_s.dropped_index_img_blocks[index] = np.zeros(next_s.block_size)
        next_s.orders[index] = block_id + 1
        next_s.angles[block_id + 1] = angle
        next_s.depth += 1
        return next_s
    
    def get_next_block_ids(self, state, current_block_id):
        """
        Returns a list of block ids.
        """
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        next_block_ids = []
        for i in range(4):
            new_block_id = current_block_id + dx[i] * state.block_dim[1] + dy[i]
            if new_block_id not in state.lost_list:
                continue
            next_block_ids.append(new_block_id)
    
    def get_valid_block_ids(self, state):
        """
        Returns a list of actions.
        """
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        chosen_block_ids = set()
        for block_id in state.lost_list:
            x, y = block_id // state.block_dim[1], block_id % state.block_dim[1]
            for i in range(4):
                new_x = x + dx[i]
                new_y = y + dy[i]
                new_block_id = new_x * state.block_dim[1] + new_y
                if new_x < 0 or new_x >= state.block_dim[0] or new_y < 0 or new_y >= state.block_dim[1]:
                    continue
                if new_block_id not in state.lost_list:
                    chosen_block_ids.add(block_id)
                    break
                
        return chosen_block_ids