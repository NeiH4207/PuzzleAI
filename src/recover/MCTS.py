from copy import deepcopy
import logging
import math
import cv2
import numpy as np
from numpy.core.fromnumeric import argmax
from multiprocessing import Pool

from src.data_helper import DataProcessor
EPS = 1e-8
log = logging.getLogger(__name__)

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, env, model, n_sim=20, 
                 c_puct=1, pos_rate=0.8, threshold=0.9,
                 n_bests = 256, verbose=False):
        self.env = env
        self.model = model
        self.n_sim = n_sim
        self.c_puct = c_puct
        self.pos_rate = pos_rate
        self.n_bests = n_bests
        self.threshold = threshold
        self.verbose = verbose
        self.Qsa  = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa  = {}  # stores #times edge s,a was visited
        self.Ns   = {}  # stores #times state s was visited
        self.Ps   = {}  # stores initial policy (returned by neural net)
        
        self.mask = None
        self.blocks_rotated = None
    
    def get_next_action(self, state, temp=1, position=None):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        state.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = state.get_string_presentation()
        
        if self.mask is None:
            self.mask = np.zeros((state.block_size[0] * 2, state.block_size[1] * 2, 3),
                                 dtype=np.uint8)
            self.blocks_rotated = np.zeros((state.block_dim[0], state.block_dim[1], 4, state.block_size[0], state.block_size[1], 3),
                                           dtype=np.uint8)
            for i in range(state.block_dim[0]):
                for j in range(state.block_dim[1]):
                    for k in range(4):
                        self.blocks_rotated[i][j][k] = np.rot90(state.blocks[i][j], k=k)
                        
        for _ in range(self.n_sim):
            self.search(state, position=position)

        counts = []
        actions = []
        for action in self.Ps[s]: 
            if (s, action) in self.Qsa:
                counts.append(self.Qsa[(s, action)])
                actions.append(action)
                        
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts)), dtype=object).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
        else:
            # probs = softmax(1.0/temp * np.log(np.array(counts) + 1e-10))
            counts = [x ** (1. / temp) for x in counts]
            counts_sum = float(sum(counts))
            if counts_sum == 0:
                probs = [1 / len(actions) for _ in range(len(actions))]
            else:
                probs = [x / counts_sum for x in counts]
        index = np.argsort(probs)[::-1]
        probs = np.array(probs, dtype=object)[index]
        actions = np.array(actions, dtype=object)[index]
        return actions, probs
    
    def search(self, state, position=None):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current state
        """
        # print('Simulating... ' + 'depth: ' + str(state.depth))
        s = state.get_string_presentation()
        terminate = state.depth == state.max_depth
        if terminate: 
            return min(state.probs)
        if self.verbose:
            state.save_image()
        if s not in self.Ps:
            lost_positions = []
            for i in range(state.block_dim[0]):
                for j in range(state.block_dim[1]):
                    if state.lost_block_labels[i][j] == 1:
                        lost_positions.append((i, j))
            stop = False
            probs = []
            actions = []
            valid_block_pos, best_pos, ranks = self.env.get_valid_block_pos(state, 
                                                                        kmax=self.n_bests, 
                                                                        last_state=False,
                                                                        position=position)
            for x, y in valid_block_pos:
                for _x, _y in lost_positions:
                    # get dropped_subblocks 2x2 from dropped_blocks
                    i, j = best_pos[(x, y)]
                    subblocks = np.array((
                        [state.dropped_blocks[i][j], state.dropped_blocks[i][j + 1]],
                        [state.dropped_blocks[i + 1][j], state.dropped_blocks[i + 1][j + 1]]), dtype=np.uint8)
                    index = np.zeros(4, dtype=np.int32)
                    index[(x - i) * 2 + (y - j)] = 1
                    mask = np.array([
                        state.masked[i][j], state.masked[i][j + 1],
                        state.masked[i + 1][j], state.masked[i + 1][j + 1]], dtype=np.uint8)
                    mask[(x - i) * 2 + (y - j)] = 1
                    index = np.concatenate((index, 1 - mask.flatten()), axis=0)
                    indexes = [index] * 4
                    images = []
                    for angle in range(4):
                        subblocks[x - i][y - j] = self.blocks_rotated[_x][_y][angle]
                        recovered_image = DataProcessor.merge_blocks(subblocks, mask=self.mask)
                        recovered_image_ = DataProcessor.convert_image_to_three_dim(recovered_image)
                        # cv2.imwrite('output/sample.png', recovered_image)
                        images.append(recovered_image_)
                        action = ((x, y), (_x, _y), angle)
                        actions.append(action)
                    _probs = self.model.predict(images, indexes)
                    probs.extend(_probs)
                    if np.max(_probs) > self.threshold:
                        stop = True
                        break
                    if stop:
                        break
                if stop:
                    break
                    
            self.Ps[s] = {}
            
            for i in range(len(actions)):
                self.Ps[s][actions[i]] = probs[i]
            self.Ns[s] = 0
            return max(probs)
     
        cur_best = -float('inf')
        best_act = -1
        for action in self.Ps[s].keys():
            if (s, action) in self.Qsa:
                u = self.Qsa[(s, action)] + \
                    self.c_puct * self.Ps[s][action] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, action)])
            else:
                u = self.c_puct * self.Ps[s][action] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
            if u > cur_best:
                cur_best = u
                best_act = action
                        
        state.probs.append(self.Ps[s][best_act])
        next_state = self.env.step(state, best_act)
        if next_state.depth == state.depth:
            print('Error: next_state.depth == state.depth')
            print()
        v = self.search(next_state)
        
        if (s, best_act) in self.Qsa:
            self.Qsa[(s, best_act)] = (self.Nsa[(s, best_act)] * self.Qsa[(s, best_act)] + v**2) / (
                    1 + self.Nsa[(s, best_act)])
            self.Nsa[(s, best_act)] += 1
        else:
            self.Qsa[(s, best_act)] = v
            self.Nsa[(s, best_act)] = 1
        self.Ns[s] += 1
        return v
