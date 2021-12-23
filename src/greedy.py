from copy import deepcopy
import numpy as np
from src.data_helper import DataProcessor
from multiprocessing import Pool

class Greedy():
    def __init__(self, env, model, n_bests=256, verbose=False):
        self.env = env
        self.model = model
        self.n_bests = n_bests
        self.verbose = verbose
        
    def next_action(self, state):
        lost_positions = []
        for i in range(state.block_dim[0]):
            for j in range(state.block_dim[1]):
                if state.lost_block_labels[i][j] == 1:
                    lost_positions.append((i, j))
        best_action = None
        best_prob = 0
        if self.verbose:
            state.save_image()
        stop = False
        valid_block_pos, best_pos, ranks = self.env.get_valid_block_pos(state, kmax=self.n_bests, last_state=False)
        for x, y in valid_block_pos:
            for _x, _y in lost_positions:
                # get dropped_subblocks 2x2 from dropped_blocks
                i, j = best_pos[x][y]
                subblocks = deepcopy(state.dropped_blocks[i:i+2, j:j+2])
                index = np.zeros(4, dtype=np.int32)
                index[(x - i) * 2 + (y - j)] = 1
                for angle in range(4):
                    block = np.rot90(state.blocks[_x][_y], k=angle)
                    subblocks[x - i][y - j] = block
                    recovered_image = DataProcessor.merge_blocks(subblocks)
                    recovered_image_ = DataProcessor.convert_image_to_three_dim(recovered_image)
                    prob = self.model.predict(recovered_image_, index) ** 2
                    action = ((x, y), (_x, _y), angle)
                    # subblocks[x - i][y - j] = np.zeros(state.block_shape, dtype=np.uint8)
                    # new_image = deepcopy(state.dropped_blocks)
                    # new_image[x][y] = np.rot90(state.blocks[_x][_y], k=angle)
                    # new_image_ = DataProcessor.merge_blocks(new_image)
                    # cv2.imwrite('output/sample.png', new_image_)
                    # print(action,  prob)
                    if prob > best_prob:
                        best_prob = prob 
                        best_action = action
                    if prob > 0.8:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
        if self.verbose:
            state.save_image()
        return best_action, best_prob