from copy import deepcopy
import logging
import math
import cv2
import numpy as np
from numpy.core.fromnumeric import argmax
from numpy.random.mtrand import dirichlet

from src.data_helper import DataProcessor
EPS = 1e-8
log = logging.getLogger(__name__)

class TreeSearch():
    """
    This class handles the A* algorithm.
    """

    def __init__(self, env, depth, breadth, verbose=False):
        self.env = env
        self.max_depth = depth
        self._depth = depth
        self._breadth = breadth
        self.verbose = verbose
        self.depth = {}
        self.parent = {}
    
    def search(self, state, k):
        
        state = state.copy()
        probs = []
        actions = self.env.get_available_actions(state)
        Vs = []
        for action in actions:            
            if action[0] == 'choose':
                Vs.append(state.curr_reward)
            else:
                _state = self.env.soft_update_state(state, action)
                Vs.append(self.env.get_reward(_state))
                # _state.save_image()
                # _state = _state
        if k == 0:
            index = np.argmax(Vs)
            return actions[index], Vs[index]
        else:
            v_min = min(Vs)
            v_max = max(Vs)
            Vs = [(x - v_min) / (v_max - v_min + EPS) for x in Vs]
            sum_v = np.sum(Vs)
            if sum_v == 0:
                probs.append(1 / len(actions))
            else:
                probs = [x / sum_v for x in Vs]
            dirichlet_noise = dirichlet(np.ones(len(actions)) * 0.3)
            probs = np.array(probs ) * 0.9 + dirichlet_noise * 0.1
                    
            # choose the action with the 5 highest probabilities
            top_index = np.argsort(probs)[-self._breadth:]
            # top_probs = probs[top_index]
            # top_probs = top_probs / np.sum(top_probs)
            top_actions = [actions[i] for i in top_index]
            next_actions = []
            next_v = []
            for action in top_actions:
                _state = self.env.soft_update_state(state, action)
                act, v = self.search(_state, k - 1)
                next_actions.append(act)
                next_v.append(v)
            index = np.argmax(next_v)
            return top_actions[index], next_v[index]
                        
    def get_action(self, state):
        s = state.get_string_presentation()
        # if self.parent[s] is None:
        #     self.depth[s] = 0
        # else:
        #     self.depth[s] = self.depth[self.parent[s]] + 1
        
        action, prob = self.search(state, self._depth)
        return action
        
         