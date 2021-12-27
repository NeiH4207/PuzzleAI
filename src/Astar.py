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

class Astar():
    """
    This class handles the A* algorithm.
    """

    def __init__(self, env, verbose=False):
        self.env = env
        self.verbose = verbose
        
    def get_action(self, state):
        s = state.get_string_presentation()
        state = state.copy()
        
        if self.verbose:
            state.save_image()
            
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
        v_min = min(Vs)
        v_max = max(Vs)
        Vs = [(x - v_min) / (v_max - v_min) for x in Vs]
        sum_v = np.sum(Vs)
        # dirichlet_noise = dirichlet(np.ones(len(actions)) * 0.3)
        # for i in range(len(actions)):
        #     probs.append(Pis[i] / sum_pis + dirichlet_noise[i])
        if sum_v == 0:
            probs.append(1 / len(actions))
        else:
            probs = [x / sum_v for x in Vs]
        best_prob = np.max(probs)
                
        # choose the action with the 5 highest probabilities
        top_3_probs = np.argsort(probs)[-3:]
        top_3_actions = [actions[i] for i in top_3_probs]
        best_act = top_3_actions[np.random.choice(range(len(top_3_actions)))]
        return best_act