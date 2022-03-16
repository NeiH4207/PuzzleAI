from copy import deepcopy
import os
import random
import cv2
import json

import numpy as np
from src.data_helper import DataProcessor

class GameInfo():
    
    def __init__(self) -> None:
        self.block_dim = None
        self.block_size = None
        self.max_n_selects = None
        self.select_swap_ratio = None
        self.image_size = None
        self.max_image_point_value = None
        self.blocks = None    
        self.original_blocks = None
        self.mode = None
        self.file_path = None
        self.challenge_id = None
        
    def save_to_json(self, file_path='./input/game_info'):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, self.name + '.json')
        data = {
            'block_dim': self.block_dim,
            'original_block_size': self.original_block_size,
            'original_blocks': self.original_blocks,
            'max_n_selects': self.max_n_selects,
            'select_swap_ratio': self.select_swap_ratio,
            'image_size': self.image_size,
            'max_image_point_value': self.max_image_point_value,
            'original_blocks': self.original_blocks,
            'mode': self.mode
        }
        # dump to pretty json file
        with open(file_path, 'w') as f:
            json.dump(data, f)
        self.file_path = file_path
        print('Game info saved to binary file %s' % file_path)
    
    def load_from_json(self, file_path='./input/game_info', file_name='game_info.json'):
        file_path = os.path.join(file_path, file_name)
        # check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError('File %s not found' % file_path)
        # load json file
        with open(file_path, 'r') as f:
            data = json.load(f)
        # set attributes
        self.block_dim = data['block_dim']
        self.original_block_size = data['original_block_size']
        self.original_blocks = data['original_blocks']
        self.max_n_selects = data['max_n_selects']
        self.select_swap_ratio = data['select_swap_ratio']
        self.image_size = data['image_size']
        self.max_image_point_value = data['max_image_point_value']
        self.original_blocks = np.array(data['original_blocks'], dtype='uint8')
        self.mode = data['mode']
        self.file_path = file_path
        print('Game info loaded from binary file %s' % file_path)
        
    def save_to_binary(self, file_path='./input/game_info'):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, self.name + '.bin')
        self.file_path = file_path
        DataProcessor.save_data_to_binary_file(self,  file_path)
    
    def load_binary(self, file_path='./input/game_info', file_name='game_info.bin'):
        self = DataProcessor.load_data_from_binary_file(file_path, file_name)
    
class State(GameInfo):
    """
    Class for the state of the environment.
    """

    def __init__(self, blocks=None, block_size=None, block_dim=None):
        super().__init__()
        self.original_blocks = blocks
        self.block_size = block_size
        self.block_dim = block_dim
        self.bottom_right_corner = [0, 0]
        self.parent = None
        self.child = None
        
    def make(self):
        self.block_shape = (self.block_size[0], self.block_size[1], 3)
        self.blocks = np.empty((self.block_dim[0], self.block_dim[1], 
                                self.block_size[0], self.block_size[1], 3), dtype='uint8')
        self.std_errs = np.empty(self.block_dim, dtype=np.float32)
        
        for i in range(self.block_dim[0]):
            for j in range(self.block_dim[1]):
                # convert to uint8
                block = self.original_blocks[i][j].astype('uint8')
                # resize to block_size
                block = cv2.resize(block, self.block_size, interpolation = cv2.INTER_AREA)
                # add to blocks
                self.blocks[i][j] = block
                self.std_errs[i][j] = self.get_std_err(block)
                               
        self.image_size = (self.block_dim[0] * self.block_size[0],
                            self.block_dim[1] * self.block_size[1])
        self.dropped_blocks = self.blocks
        self.save_image()
        self.dropped_blocks, self.lost_block_labels, self.masked = DataProcessor.drop_all_blocks(self.blocks)
        self.set_string_presentation()
        self.num_blocks = len(self.blocks)
        self.depth = 0
        self.max_depth = int(np.sum(self.lost_block_labels))
        self.probs = [1.0] 
        self.actions = []
        self.last_action = (0, 0)
        self.inverse = np.zeros((self.block_dim[0], self.block_dim[1], 3), dtype=np.int8)
        for i in range(self.block_dim[0]):
            for j in range(self.block_dim[1]):
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
        state.std_errs = self.std_errs
        state.bottom_right_corner = self.bottom_right_corner
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
        state.select_swap_ratio = self.select_swap_ratio
        state.max_n_selects = self.max_n_selects
        state.mode = self.mode
        state.name = self.name
        return state
    
    
    def small_copy(self):
        """
        Returns a copy of the state.
        """
        state = State()
        small_size = (16, 16)
        state.original_blocks = np.zeros((self.block_dim[0], self.block_dim[1],
                                            small_size[0], small_size[1], 3), dtype='uint8')
            
        for i in range(self.block_dim[0]):
            for j in range(self.block_dim[1]):
                # convert to uint8
                block = self.original_blocks[i][j].astype('uint8')
                # resize to block_size
                block = cv2.resize(block, (16,16), interpolation = cv2.INTER_AREA)
                # add to blocks
                state.original_blocks[i][j] = block
                      
        state.block_size = self.block_size
        state.block_shape = self.block_shape
        state.block_dim = self.block_dim
        state.image_size = self.image_size
        state.inverse = self.inverse
        state.select_swap_ratio = self.select_swap_ratio
        state.max_n_selects = self.max_n_selects
        return state
    
    def get_std_err(self, block): 
        '''
        Returns the standard error of the block 2D array.
        '''
        # three first colunms
        std_1 = np.std(block[:, :5], axis=1)
        # three last colunms
        std_2 = np.std(block[:, -5:], axis=1)
        # three first rows
        std_3 = np.std(block[:5, :], axis=0)
        # three last rows
        std_4 = np.std(block[-5:, :], axis=0)
        # return the mean of the four std
        return np.mean([std_1, std_2, std_3, std_4])
    
    def translation(self, mode):
        cp_dropped_blocks = np.zeros((self.block_dim[0], self.block_dim[1],
                                                self.block_size[0], self.block_size[1], 3), dtype=np.int32)
        cp_masked = np.zeros((self.block_dim[0], self.block_dim[1]), dtype=np.int32)
        # cp_lost_block_labels = np.zeros((self.block_dim[0], self.block_dim[1]), dtype=np.int32)
        cp_inverse = np.zeros((self.block_dim[0], self.block_dim[1], 3), dtype=np.int8)
        cp_std_errs = np.zeros(self.block_dim, dtype=np.float32)
        if mode == 'down':
            for i in range(self.block_dim[0]):
                for j in range(self.block_dim[1]):
                    cp_dropped_blocks[(i + 1)%self.block_dim[0]][j] = self.dropped_blocks[i][j]
                    cp_masked[(i + 1)%self.block_dim[0]][j] = self.masked[i][j]
                    cp_inverse[(i + 1)%self.block_dim[0]][j] = self.inverse[i][j]
                    cp_std_errs[(i + 1)%self.block_dim[0]][j] = self.std_errs[i][j]
                    
        if mode == 'right':
            for i in range(self.block_dim[0]):
                for j in range(self.block_dim[1]):
                    cp_dropped_blocks[i][(j + 1)%self.block_dim[1]] = self.dropped_blocks[i][j]
                    cp_masked[i][(j + 1)%self.block_dim[1]] = self.masked[i][j]
                    cp_inverse[i][(j + 1)%self.block_dim[1]] = self.inverse[i][j]
                    cp_std_errs[i][(j + 1)%self.block_dim[1]] = self.std_errs[i][j]
                    
        if mode == 'up':
            for i in range(self.block_dim[0]):
                for j in range(self.block_dim[1]):
                    cp_dropped_blocks[(i - 1)%self.block_dim[0]][j] = self.dropped_blocks[i][j]
                    cp_masked[(i - 1)%self.block_dim[0]][j] = self.masked[i][j]
                    cp_inverse[(i - 1)%self.block_dim[0]][j] = self.inverse[i][j]
                    cp_std_errs[(i - 1)%self.block_dim[0]][j] = self.std_errs[i][j]
            
        if mode == 'left':
            for i in range(self.block_dim[0]):
                for j in range(self.block_dim[1]):
                    cp_dropped_blocks[i][(j - 1)%self.block_dim[1]] = self.dropped_blocks[i][j]
                    cp_masked[i][(j - 1)%self.block_dim[1]] = self.masked[i][j]
                    cp_inverse[i][(j - 1)%self.block_dim[1]] = self.inverse[i][j]
                    cp_std_errs[i][(j - 1)%self.block_dim[1]] = self.std_errs[i][j]
        
        self.dropped_blocks = cp_dropped_blocks
        self.masked = cp_masked
        self.inverse = cp_inverse
        self.std_errs = cp_std_errs
            
    def string_presentation(self, items):
        return hash(str(items))
    
    def get_string_presentation(self):
        return self.name
    
    def set_string_presentation(self):
        self.name = self.string_presentation([self.dropped_blocks, self.masked])
    
    def save_image(self, filename='sample.png'):
        new_img = DataProcessor.merge_blocks(self.dropped_blocks)
        cv2.imwrite('output/' + filename, new_img)
        
    def get_image(self):
        return DataProcessor.merge_blocks(self.dropped_blocks)
        
    def save_binary(self, path):
        DataProcessor.save_binary(self.dropped_blocks, path)
        
class Environment():
    """
    Class for the environment.
    """
    def __init__(self, name='recover_image'):
        self.name = name
        self.state = None
        self.reset()
        self.next_step = {}
        self.canvas = None
        
    def reset(self):
        return
    
    def step(self, state, action, verbose=False):
        """
        Performs an action in the environment.
        """
        # s_name = state.get_string_presentation()
        # if (s_name, action) in self.next_step:
        #     state.child = self.next_step[(s_name, action)]
        #     return self.next_step[(s_name, action)]
        (x, y), (_x, _y), angle = action
        next_s = state.copy()
        if x == -1:
            next_s.translation('down')
            next_s.bottom_right_corner[0] += 1
            x = 0
        
        if y == -1:
            next_s.translation('right')
            next_s.bottom_right_corner[1] += 1
            y = 0
        
        next_s.dropped_blocks[x][y] = np.rot90(state.blocks[_x][_y], k=angle)
        next_s.masked[x][y] = 1
        next_s.actions.append(action)
        next_s.lost_block_labels[_x][_y] = 0
        next_s.inverse[x][y] = (_x, _y, angle)
        next_s.depth += 1
        next_s.last_action = (x, y)
        next_s.set_string_presentation()
        # self.next_step[(s_name, action)] = next_s
        next_s.parent = state
        return next_s
    
    def remove(self, state, action):
        x, y = action
        next_s = state.copy()
        next_s.masked[x][y] = 0
        next_s.dropped_blocks[x][y] = np.zeros((state.block_shape[0], state.block_shape[1], 3), dtype=np.uint8)
        _x, _y = next_s.inverse[x][y][:2]
        next_s.inverse[x][y] = (x, y, 0)
        next_s.lost_block_labels[_x][_y] = 1
        next_s.parent = state
        next_s.set_string_presentation()
        next_s.depth -= 1
        
        if x == 0:
            sum_x = 0
            for i in range(state.block_dim[1]):
                sum_x += state.masked[0][i]
            if sum_x == 0:
                next_s.translation('up')
                next_s.bottom_right_corner[0] -= 1
        
        if y == 0:
            sum_y = 0
            for i in range(state.block_dim[0]):
                sum_y += state.masked[i][0]
            if sum_y == 0:
                next_s.translation('left')
                next_s.bottom_right_corner[1] -= 1
        
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
    
    def get_valid_block_pos(self, state, kmax=4, last_state=False, position=None):
        """
        Returns a list of actions.
        """
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        chosen_block_ids = set()
        best_square = {}
        from_position = {}
        num_masked = 0
        for i in range(state.block_dim[0]):
            for j in range(state.block_dim[1]):
                if state.masked[i][j] == 1:
                    num_masked += 1
                    
        full_row = (num_masked >= state.block_dim[1] * 2)
        full_col = (num_masked >= (state.block_dim[1] * 2 + state.block_dim[0] * 2 - 4))
                    
        # print('full_row:', full_row)
        # print('full_col:', full_col)
        if position is not None:
            _x, _y = position
            if ((_x < 0 and state.bottom_right_corner[0] < state.block_dim[0] - 1) \
                    and (_y < 0 and state.bottom_right_corner[1] < state.block_dim[1])) \
                    or state.masked[_x][_y] == 0:
                chosen_block_ids.add((_x, _y))
                from_position[(_x, _y)] = None
            
        if len(chosen_block_ids) == 0:
            # take random action
            for x in range(state.block_dim[0]):
                for y in range(state.block_dim[1]):
                    if state.masked[x][y] == 0:
                        continue
                    for i in range(4):
                        new_x = x + dx[i]
                        new_y = y + dy[i]
                        if new_x >= state.block_dim[0] \
                            or new_y >= state.block_dim[1]:
                            continue
                        if ((new_x < 0 and state.bottom_right_corner[0] < state.block_dim[0] - 1) \
                                and (new_y < 0 and state.bottom_right_corner[1] < state.block_dim[1])) \
                                or state.masked[new_x][new_y] == 0:
                            chosen_block_ids.add((new_x, new_y))
                            from_position[(new_x, new_y)] = (x, y)
    
        ranks = {}
        max_rank = 0
        for x, y in chosen_block_ids:
            counts = np.zeros((2, 2), dtype=np.int8)
            corner = (x - 1, y - 1)
            for i in range(corner[0], min(state.block_dim[0] - 1, x + 1)):
                for j in range(corner[1], min(state.block_dim[1] - 1, y + 1)):
                    counts[i - corner[0]][j - corner[1]] += \
                        np.sum(state.masked[max(0, i):i + 2, max(0, j):j + 2])
            mx = counts.max()
            num_masked_row, num_masked_col = 0, 0
            x_left, y_left, x_right, y_right = x, y, x, y
            while x_left > 0:
                if state.masked[x_left - 1][y] == 1:
                    num_masked_col += 1
                else:
                    break
                x_left -= 1
            while y_left > 0:
                if state.masked[x][y_left - 1] == 1:
                    num_masked_row += 1
                else:
                    break
                y_left -= 1
            while x_right < state.block_dim[0] - 1:
                if state.masked[x_right + 1][y] == 1:
                    num_masked_col += 1
                else:
                    break
                x_right += 1
            while y_right < state.block_dim[1] - 1:
                if state.masked[x][y_right + 1] == 1:
                    num_masked_row += 1
                else:
                    break
                y_right += 1
            best_pos = np.argwhere(counts==mx)
            if full_row and not full_col:
                if mx > 1:
                    mx += num_masked_col * 0.001 - 0.6 * num_masked_row
            elif not full_row:
                mx += (num_masked_row * 1.001 + num_masked_col) * 0.001
            ranks[(x, y)] = mx
            if mx > max_rank:
                max_rank = mx
            best_pos = best_pos[np.random.randint(0, len(best_pos))]
            best_square[(x, y)] = (corner[0] + best_pos[0], corner[1] + best_pos[1]) 
        # print(ranks)
        chosen_block_ids = list(chosen_block_ids)
        block_ids = []
        for (x, y) in chosen_block_ids:
            if ranks[(x, y)] == max_rank:
                block_ids.append((x, y))
        # shuffle(block_ids)
        # sort by std errors
        if (num_masked == state.block_dim[1] * 2) and not full_col:
            kmax = state.block_dim[1] + 1
        
        if full_col and full_row:
            kmax = 2
        block_ids.sort(key=lambda x: ranks[x], reverse=True)  
        # block_ids.sort(key=lambda x: state.std_errs[from_position[x][0]][from_position[x][1]], reverse=True)
        final_block_ids = block_ids[:min(kmax, len(block_ids))]
        return final_block_ids, best_square, ranks