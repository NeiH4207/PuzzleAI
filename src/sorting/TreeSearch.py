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
        
        # if self.verbose:
        #     state.save_image()
        # state.show()
        actions = self.env.get_available_actions(state)
    
        rewards = []
        for action in actions:
            rewards.append(self.env.get_strict_reward(state, action))
        
        if len(rewards) == 0:
            return None, 100
        
        if k == 0:
            best_actions = np.array(np.argwhere(rewards == np.max(rewards)), dtype=object).flatten()
            best_action = np.random.choice(best_actions)
            return actions[best_action], rewards[best_action]
        else:
            # select the action with the 5 highest probabilities
            # probs = np.array(rewards) / np.sum(rewards)
            # dirichlet_noise = dirichlet(np.ones(len(actions)) * 0.3)
            # probs = np.array(probs) * 0.9 + np.array(dirichlet_noise) * 0.1
            top_index = np.argsort(rewards)[-self._breadth:]
            top_actions = [actions[i] for i in top_index]
            next_actions = []
            next_v = []
            for i, action in enumerate(top_actions):
                _state = self.env.step(state, action)
                act, v = self.search(_state, k - 1)
                next_actions.append(act)
                next_v.append(v + rewards[top_index[i]])
            best_indices = np.argwhere(next_v == np.max(next_v)).flatten()
            best_index = np.random.choice(best_indices)
            return top_actions[best_index], next_v[best_index]
                        
    def get_action(self, state):
        action, v = self.search(state, self._depth)
        # print(action, v)
        return action
        
         