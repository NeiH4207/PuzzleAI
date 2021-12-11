from copy import deepcopy
import cv2

import numpy as np

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
        self.block_dim = block_dim
        self.original_blocks = DataProcessor.split_image_to_blocks(image, block_dim)
        old_blocks = DataProcessor.split_image_to_blocks(image, block_dim)
        self.image_size = block_size[0] * block_dim[0], block_size[1] * block_dim[1]
        self.blocks = np.empty((block_dim[0], block_dim[1], block_size[0], block_size[1], 3), dtype=np.int8)
        for i in range(block_dim[0]):
            for j in range(block_dim[1]):
                self.blocks[i][j] = cv2.resize(old_blocks[i][j], (block_size[0], block_size[1]), interpolation=cv2.INTER_AREA)
        self.dropped_blocks, self.lost_block_labels, self.masked = DataProcessor.drop_all_blocks(self.blocks)
        self.lost_index_img_blocks = np.empty((block_dim[0], block_dim[1], block_size[0], block_size[1]), dtype=np.int8)

        for x in range(block_dim[0]):
            for y in range(block_dim[1]):
                if self.lost_block_labels[x][y] == 0:
                    self.lost_index_img_blocks[x][y] = np.zeros((block_size[0], block_size[1]), dtype=np.int8)
                else:
                    self.lost_index_img_blocks[x][y] = np.ones((block_size[0], block_size[1]), dtype=np.int8)
                
        self.index_imgs = np.zeros((block_dim[0], block_dim[1], self.image_size[0], self.image_size[1]), dtype=np.int8)
        
        for i in range(block_dim[0]):
            for j in range(block_dim[1]):
                self.index_imgs[i][j][i * block_size[0]:(i + 1) * block_size[0], 
                                    j * block_size[1]:(j + 1) * block_size[1]] = np.ones((block_size[0], block_size[1]), dtype=np.int8)
        
        self.num_blocks = len(self.blocks)
        self.depth = 0
        self.max_depth = int(np.sum(self.lost_block_labels))
        self.probs = [1.0] 
        self.actions = []
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
        state.block_dim = self.block_dim
        state.original_blocks = deepcopy(self.original_blocks)
        state.image_size = self.image_size
        state.blocks = deepcopy(self.blocks)
        state.dropped_blocks = deepcopy(self.dropped_blocks)
        state.lost_block_labels = deepcopy(self.lost_block_labels)
        state.masked = deepcopy(self.masked)
        state.lost_index_img_blocks = deepcopy(self.lost_index_img_blocks)
        state.index_imgs = deepcopy(self.index_imgs)
        state.num_blocks = self.num_blocks
        state.depth = self.depth
        state.max_depth = self.max_depth
        state.probs = deepcopy(self.probs)
        state.actions = deepcopy(self.actions)
        state.inverse = deepcopy(self.inverse)
        state.mode = self.mode
        return state
    
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

    def reset(self):
        return

    def step(self, state, action):
        """
        Performs an action in the environment.
        """
        (x, y), (_x, _y), angle = action
        next_s = state.copy()
        next_s.dropped_blocks[x][y] = np.rot90(state.blocks[_x][_y], k = angle)
        next_s.lost_block_labels[_x][_y] = 0
        next_s.masked[x][y] = 1
        next_s.actions.append(action)
        next_s.lost_index_img_blocks[x][y] = np.zeros(state.block_size)
        next_s.inverse[x][y] = (_x, _y, angle)
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
    
    def get_valid_block_pos(self, state):
        """
        Returns a list of actions.
        """
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        chosen_block_ids = set()
        for x in range(state.block_dim[0]):
            for y in range(state.block_dim[1]):
                if state.masked[x][y] == 0:
                    continue
                for i in range(4):
                    new_x = x + dx[i]
                    new_y = y + dy[i]
                    if new_x < 0 or new_x >= state.block_dim[0] or new_y < 0 or new_y >= state.block_dim[1]:
                        continue
                    if state.masked[new_x][new_y] == 0:
                        chosen_block_ids.add((new_x, new_y))
                
        return chosen_block_ids