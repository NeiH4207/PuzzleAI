from copy import deepcopy
import logging
import math
import cv2
import numpy as np
from numpy.core.fromnumeric import argmax

from src.data_helper import DataProcessor
EPS = 1e-8
log = logging.getLogger(__name__)

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, env, model, n_sim=20, c_puct=1, pos_rate=0.8):
        self.env = env
        self.model = model
        self.n_sim = n_sim
        self.c_puct = c_puct
        self.Qsa  = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa  = {}  # stores #times edge s,a was visited
        self.Ns   = {}  # stores #times state s was visited
        self.Ps   = {}  # stores initial policy (returned by neural net)

        self.Es   = {}  # stores env.get_env_ended ended for state s
        self.Vs   = {}  # stores env.getValidMoves for state s
        self.As   = {}  # stores agent position list for state s
        
    def predict(self, state, temp=1):
        return self.getActionProb(state, temp)
    
    def string_presentation(self, state):
        return hash(str(state))
        
    def get_probs(self, state, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        state.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = self.string_presentation(state.dropped_blocks)
        
        for _ in range(self.n_sim):
            self.search(state)

        counts = []
        actions = []
        lost_positions = set()
        for i in range(state.block_dim[0]):
            for j in range(state.block_dim[1]):
                if state.lost_block_labels[i][j] == 1:
                    lost_positions.add((i, j))
        valid_block_pos = self.env.get_valid_block_pos(state)
        for x, y in valid_block_pos:
            for _x, _y in lost_positions:
                for angle in range(4): 
                    action = ((x, y), (_x, _y), angle)
                    if (s, action) in self.Qsa:
                        counts.append(self.Qsa[(s, action)])
                        actions.append(action)
                        
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
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
        return actions[argmax(probs)], (self.Ps[s][actions[argmax(probs)]], 
                                        self.Qsa[(s, actions[argmax(probs)])], 
                                        self.Nsa[(s, actions[argmax(probs)])])

    def search(self, state):
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
        s = self.string_presentation(state.dropped_blocks)
        terminate = state.depth == state.max_depth
        if terminate: 
            return min(state.probs)
        state.save_image()
        state = state.copy()
        if s not in self.Ps:
            # leaf node
            probs = []
            lost_positions = set()
            for i in range(state.block_dim[0]):
                for j in range(state.block_dim[1]):
                    if state.lost_block_labels[i][j] == 1:
                        lost_positions.add((i, j))
            
            self.Ps[s] = {}
            dropped_index_image = DataProcessor.merge_blocks(state.lost_index_img_blocks, mode='gray')
            valid_block_pos = self.env.get_valid_block_pos(state)
            for x, y in valid_block_pos:
                for _x, _y in lost_positions:
                    cp_dropped_block = deepcopy(state.dropped_blocks)
                    cp_dropped_block[x][y] = state.blocks[_x][_y]
                    recovered_image = DataProcessor.merge_blocks(cp_dropped_block)
                    recovered_image = DataProcessor.convert_image_to_three_dim(recovered_image)
                    prob, angle_prob = self.model.predict(recovered_image, state.index_imgs[x][y], dropped_index_image)
                    for angle in range(4):
                        action = ((x, y), (_x, _y), angle)
                        self.Ps[s][action] = prob * angle_prob[angle]
                        block = np.rot90(state.blocks[_x][_y], k=angle)
                        cp_dropped_block[x][y] = block
                        new_image = DataProcessor.merge_blocks(cp_dropped_block)
                        cv2.imwrite('output/sample.png', new_image)
                        print(action, prob, angle_prob[angle])
                        print()
                    probs.append(prob * angle_prob[angle])
            self.Ns[s] = 0
            return min(max(probs), min(state.probs))
     
        cur_best = -float('inf')
        best_act = -1
        lost_positions = set()
        for i in range(state.block_dim[0]):
            for j in range(state.block_dim[1]):
                if state.lost_block_labels[i][j] == 1:
                    lost_positions.add((i, j))
        valid_block_pos = self.env.get_valid_block_pos(state)
        for x, y in valid_block_pos:
            for _x, _y in lost_positions:
                for angle in range(4): 
                    action = ((x, y), (_x, _y), angle)
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
        v = self.search(next_state)
        
        if (s, best_act) in self.Qsa:
            self.Qsa[(s, best_act)] = (self.Nsa[(s, best_act)] * self.Qsa[(s, best_act)] + v) / (
                    1 + self.Nsa[(s, best_act)])
            self.Nsa[(s, best_act)] += 1
        else:
            self.Qsa[(s, best_act)] = v
            self.Nsa[(s, best_act)] = 1
        self.Ns[s] += 1
        return v
