from copy import deepcopy
import logging
import numpy as np

EPS = 1e-8
log = logging.getLogger(__name__)

class lecoDFS():
    """
    This class handles the IDA* algorithm.
    """

    def __init__(self, env, depth, breadth, verbose=False):
        self.env = env
        self.max_depth = depth
        self._depth = depth
        self._breadth = breadth
        self.verbose = verbose
        self.depth = {}
        self.parent = {}
        self.leafs = {}
        self.old_v = -np.inf
      
    def get_available_actions(self, state, repexat=False):
        actions = []
        should_be_select = False
        if state.curr_position != None:
            # up, right, down, left
            dx = [1, 0, -1, 0] 
            dy = [0, 1, 0, -1]
            x1, y1 = state.curr_position
            value = x1 * state.shape[1] + y1
            if value != state.targets[x1][y1]:
                
                for i in range(4):
                    x2 = (x1 + dx[i]) % state.shape[0]
                    y2 = (y1 + dy[i]) % state.shape[1]
                    value_2 = x2 * state.shape[1] + y2
                    if value_2 >= state.n_trues:
                        actions.append(('swap', (x1, y1, x2, y2)))
            else:
                should_be_select = True
        else:
            should_be_select = True
            
        if state.n_selects < state.max_select and should_be_select:
            stop = False
            for i in range(state.shape[0]):
                for j in range(state.shape[1]):
                    if state.targets[i][j] == state.n_trues:
                        actions.append(('select', (i, j)))
                        stop = True
                        break
                if stop:
                    break
        return actions
    
    def search(self, state, k):
        state = state.copy()
        actions = self.get_available_actions(state)
        rewards = []
        should_be_swap = False
        for action in actions:
            if action[0] == 'select' and should_be_swap:
                break
            rewards.append(self.env.get_strict_reward(state, action))
            if action[0] == 'swap' and rewards[-1] > 0:
                should_be_swap = True
            
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
            top_index = np.argsort(rewards)[::-1]
            
            top_actions = [actions[i] for i in top_index]
            next_actions = []
            next_v = []
            selected_index = []
            for i, action in enumerate(top_actions):
                _state = self.env.step(state, action)
                if _state.get_string_presentation() in state.parent_states:
                    continue
                act, v = self.search(_state, k - 1)
                next_actions.append(action)
                next_v.append(v + rewards[top_index[i]])
                selected_index.append(i)
                if len(next_actions) == self._breadth:
                  break
            if len(next_actions) == 0:
                # return random action
                return top_actions[np.random.randint(len(top_actions))], -1000
            best_indices = np.argwhere(next_v == np.max(next_v)).flatten()
            best_index = np.random.choice(best_indices)
            return next_actions[best_index], next_v[best_index]
                        
    def get_action(self, state):
        action, v = self.search(state, self._depth)
        return action
        
         