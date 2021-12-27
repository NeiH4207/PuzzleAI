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

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, env, n_sim=20, c_puct=1, verbose=False):
        self.env = env
        self.n_sim = n_sim
        self.c_puct = c_puct
        self.verbose = verbose
        self.Qsa  = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa  = {}  # stores #times edge s,a was visited
        self.Ns   = {}  # stores #times state s was visited
        self.Ps   = {}  # stores initial policy (returned by neural net)
        self.Es   = {}  # stores game.getGameEnded ended for state s
        
    def get_action(self, state, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        state.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = state.get_string_presentation()
        
        for _ in range(self.n_sim):
            self.search(state)

        counts = []
        actions = []
        for action in self.Ps[s]: 
            if (s, action) in self.Nsa:
                counts.append(self.Nsa[(s, action)])
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
        return actions[argmax(probs)]

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
        # print('Simulating... ' + 'depth: ' + str(state.depth))
        s = state.get_string_presentation()
        state = state.copy()
        
        terminate = self.env.get_game_ended(state)
        
        if terminate: 
            return self.env.get_reward(state)
        
        if self.verbose:
            state.save_image()
            
        if s not in self.Ps:
            # leaf node
            probs = []
            actions = self.env.get_available_actions(state)
            
            self.Ps[s] = {}
            Pis = []
            for action in actions:
                
                if action[0] == 'choose':
                    Pis.append(state.curr_reward)
                else:
                    _state = self.env.soft_update_state(state, action)
                    Pis.append(self.env.get_reward(_state))
                    # _state.save_image()
                    # _state = _state
            sum_pis = np.sum(Pis)
            # dirichlet_noise = dirichlet(np.ones(len(actions)) * 0.3)
            # for i in range(len(actions)):
            #     probs.append(Pis[i] / sum_pis + dirichlet_noise[i])
            if sum_pis == 0:
                probs.append(1 / len(actions))
            else:
                probs = [x / sum_pis for x in Pis]
            best_prob = np.max(probs)
            for i in range(len(actions)):
                self.Ps[s][actions[i]] = probs[i]
                
            self.Ns[s] = 0
            return best_prob
        if state.depth >= 250: 
            print('Warning: depth = ' + str(state.depth))
        cur_best = -float('inf')
        best_acts = []
        avaliable_actions = self.env.get_available_actions(state)
        for action in self.Ps[s].keys():
            if (s, action) in self.Qsa:
                u = self.Qsa[(s, action)] + \
                    self.c_puct * self.Ps[s][action] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, action)])
            else:
                u = self.c_puct * self.Ps[s][action] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
            if u > cur_best:
                cur_best = u
                best_acts = [action]
            elif u == cur_best:
                best_acts.append(action)
        best_act = best_acts[np.random.choice(range(len(best_acts)))]
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
