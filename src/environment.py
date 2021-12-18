from copy import deepcopy
import cv2

import numpy as np
from numpy.core.shape_base import block

from src.data_helper import DataProcessor

class State:
    """
    Class for the state of the environment.
    """

    def __init__(self, image=None, block_size=None, block_dim=None):
        if image is not None:
            self.make(image, block_size, block_dim)
        else:
            pass
        
    def make(self, image, block_size, block_dim):
        self.block_size = block_size
        self.block_shape = (block_size[0], block_size[1], 3)
        self.block_dim = block_dim
        self.original_blocks = DataProcessor.split_image_to_blocks(image, block_dim, smoothing=True)
        old_blocks = DataProcessor.split_image_to_blocks(image, block_dim)
        self.image_size = block_size[0] * block_dim[0], block_size[1] * block_dim[1]
        self.blocks = np.empty((block_dim[0], block_dim[1], block_size[0], block_size[1], 3), dtype=np.int8)
        for i in range(block_dim[0]):
            for j in range(block_dim[1]):
                self.blocks[i][j] = cv2.resize(old_blocks[i][j], (block_size[0], block_size[1]), interpolation=cv2.INTER_AREA)
        self.dropped_blocks, self.lost_block_labels, self.masked = DataProcessor.drop_all_blocks(self.blocks)
        self.set_string_presentation()
        self.num_blocks = len(self.blocks)
        self.depth = 0
        self.max_depth = int(np.sum(self.lost_block_labels))
        self.probs = [1.0] 
        self.actions = []
        self.last_action = (0, 0)
        self.inverse = np.zeros((block_dim[0], block_dim[1], 3), dtype=np.int8)
        for i in range(block_dim[0]):
            for j in range(block_dim[1]):
                self.inverse[i][j] = (i, j, 0)
        self.mode = 'rgb'
     
    def copy(self):
        """
        Returns a copy of the state.
        """
        state = State()
        state.block_size = self.block_size
        state.block_shape = self.block_shape
        state.block_dim = self.block_dim
        state.original_blocks = self.original_blocks
        state.image_size = self.image_size
        state.blocks = self.blocks
        state.dropped_blocks = deepcopy(self.dropped_blocks)
        state.lost_block_labels = deepcopy(self.lost_block_labels)
        state.masked = deepcopy(self.masked)
        state.num_blocks = self.num_blocks
        state.depth = self.depth
        state.max_depth = self.max_depth
        state.probs = deepcopy(self.probs)
        state.actions = deepcopy(self.actions)
        state.last_action = self.last_action
        state.inverse = deepcopy(self.inverse)
        state.mode = self.mode
        state.name = self.name
        return state
       
    def string_presentation(self, items):
        return hash(str(items))
    
    def get_string_presentation(self):
        return self.name
    
    def set_string_presentation(self):
        self.name = self.string_presentation([self.dropped_blocks, self.masked])
    
    def save_image(self, filename='sample.png'):
        new_img = DataProcessor.merge_blocks(self.dropped_blocks, 'rgb')
        cv2.imwrite('output/' + filename, new_img)

class Environment():
    """
    Class for the environment.
    """
    def __init__(self, name='recover_image'):
        self.name = name
        self.state = None
        self.reset()
        self.next_step = {}

    def reset(self):
        return
    
    def step(self, state, action):
        """
        Performs an action in the environment.
        """
        s_name = state.get_string_presentation()
        if (s_name, action) in self.next_step:
            return self.next_step[(s_name, action)]
        (x, y), (_x, _y), angle = action
        next_s = state.copy()
        next_s.dropped_blocks[x][y] = np.rot90(state.blocks[_x][_y], k=angle)
        next_s.masked[x][y] = 1
        next_s.actions.append(action)
        next_s.inverse[x][y] = (_x, _y, angle)
        next_s.depth += 1
        next_s.last_action = (x, y)
        next_s.set_string_presentation()
        self.next_step[(s_name, action)] = next_s
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
    
    def get_valid_block_pos(self, state, kmax=4):
        """
        Returns a list of actions.
        """
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        chosen_block_ids = set()
        best_square = np.zeros((state.block_dim[0], state.block_dim[1], 2), dtype=np.int8)
        for i in range(4):
            new_x = state.last_action[0] + dx[i]
            new_y = state.last_action[1] + dy[i]
            if new_x < 0 or new_x >= state.block_dim[0] \
                or new_y < 0 or new_y >= state.block_dim[1]:
                continue
            if state.masked[new_x][new_y] == 0:
                chosen_block_ids.add((new_x, new_y))
        
        if len(chosen_block_ids) == 0:
            for x in range(state.block_dim[0]):
                for y in range(state.block_dim[1]):
                    if state.masked[x][y] == 0:
                        continue
                    for i in range(4):
                        new_x = x + dx[i]
                        new_y = y + dy[i]
                        if new_x < 0 or new_x >= state.block_dim[0] \
                            or new_y < 0 or new_y >= state.block_dim[1]:
                            continue
                        if state.masked[new_x][new_y] == 0:
                            chosen_block_ids.add((new_x, new_y))
        ranks = np.zeros((state.block_dim[0], state.block_dim[1]), dtype=np.int8)
        max_rank = 0
        for x, y in chosen_block_ids:
            counts = np.zeros((2, 2), dtype=np.int8)
            corner = (max(0, x - 1), max(0, y - 1))
            for i in range(corner[0], min(state.block_dim[0] - 1, x + 1)):
                for j in range(corner[1], min(state.block_dim[1] - 1, y + 1)):
                    counts[i - corner[0]][j - corner[1]] += \
                        np.sum(state.masked[i:i + 2, j:j + 2])
            mx = counts.max()
            best_pos = np.argwhere(counts==mx)
            ranks[x][y] = mx
            if mx > max_rank:
                max_rank = mx
            best_pos = best_pos[np.random.randint(0, len(best_pos))]
            best_square[x][y] = (corner[0] + best_pos[0], corner[1] + best_pos[1]) 
        # print(ranks)
        chosen_block_ids = list(chosen_block_ids)
        final_block_ids = []
        for (x, y) in chosen_block_ids:
            if ranks[x][y] == max_rank:
                final_block_ids.append((x, y))
        final_block_ids = final_block_ids[:min(kmax, len(final_block_ids))]
        return final_block_ids, best_square, ranks