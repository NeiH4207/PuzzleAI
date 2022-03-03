from copy import deepcopy
import logging
import math
import cv2
import numpy as np
from requests.api import get
from utils import LongestIncreasingSubsequence as LIS
EPS = 1e-8
log = logging.getLogger(__name__)

class Standard():
    """
    This class handles the A* algorithm.
    """

    def __init__(self, env, verbose=False):
        self.env = env
        self.verbose = verbose
        self.cursor = None 
        self.curr_pos = None # Source
        self.shape = None
        self.sequential_mode = False
        self.curr_value = 0
        self.action_list = []
        self.n_fast_moves = -2
        
    def update_curr_pos(self, state):
        dx = [1, 0, -1, 0]
        dy = [0, 1, 0, -1]
        for i in range(4):
            _x, _y = self.curr_pos[0] + dx[i], self.curr_pos[1] + dy[i]
            _x = (_x + self.shape[0]) % self.shape[0]
            _y = (_y + self.shape[1]) % self.shape[1]
            if state.targets[_x][_y] == self.curr_value:
                self.curr_pos = (_x, _y)
                return
        
    def set_next_position(self, state, curr_pos):
        value = state.targets[curr_pos[0]][curr_pos[1]]
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if state.targets[i][j] == value + 1:
                    self.curr_pos = (i, j)
                    self.curr_value = value + 1
                    return
        
    def set_cursor(self, state, source=0, index=None):
        if index == None:
            index = state.shape[0] * state.shape[1] - 1
        self.action_list = []
        self.shape = state.shape
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if state.targets[i][j] == index:
                    self.cursor = (i, j)
                if state.targets[i][j] == source:
                    self.curr_pos = (i, j)
                    self.curr_value = source
                    
        self.action_list.append(('select', (self.cursor[0], self.cursor[1])))
        self.sequential_mode = True
                
    def move_left(self):
        action = ('swap', (self.cursor[0], 
                         self.cursor[1], 
                         self.cursor[0], 
                         (self.cursor[1] - 1 + self.shape[1]) % self.shape[1]))
        self.cursor = (self.cursor[0],
                          (self.cursor[1] - 1 + self.shape[1]) % self.shape[1])
        return action
        
    def move_right(self):
        action = ('swap', (self.cursor[0], 
                         self.cursor[1], 
                         self.cursor[0], 
                        (self.cursor[1] + 1) % self.shape[1]))
        self.cursor = (self.cursor[0],
                        (self.cursor[1] + 1) % self.shape[1])
        return action
    
    def move_up(self):
        action = ('swap', (self.cursor[0],
                            self.cursor[1],
                            (self.cursor[0] - 1 + self.shape[0]) % self.shape[0],
                            self.cursor[1]))
        self.cursor = ((self.cursor[0] - 1 + self.shape[0]) % self.shape[0],
                          self.cursor[1])
        return action
    
    def move_down(self):
        action = ('swap', (self.cursor[0], 
                         self.cursor[1], 
                         (self.cursor[0] + 1) % self.shape[0], 
                         self.cursor[1]))
        self.cursor = ((self.cursor[0] + 1) % self.shape[0],
                          self.cursor[1])
        return action
    
    def execute_last_row(self, state):
        
        arr = [x for x in state.targets[self.shape[0] - 1]]
        last = len(arr) - 1
        for _ in range(1, len(arr)):
            idx = np.argmax(arr[:last+1])
            if idx == last:
                last -= 1
                continue
            self.action_list.append(('select', (self.shape[0] - 1, idx)))
            while idx < last:
                arr[idx], arr[idx + 1] = arr[idx + 1], arr[idx]
                self.action_list.append(('swap', (self.shape[0] - 1, idx, self.shape[0] - 1, idx + 1)))
                idx += 1
            last -= 1
            
        if len(self.action_list) > 0:
            self.sequential_mode =  True
        return self.get_action(state)
    
    def execute_two_last_row(self, state):
        x_source, y_source = self.curr_pos[0], self.curr_pos[1]
        x_cursor, y_cursor = self.cursor[0], self.cursor[1]
        value = state.targets[x_source][y_source]
        x_target, y_target = value // self.shape[1], value % self.shape[1]
        
        if y_target < self.shape[1] - 2:
            value_2 = value + self.shape[0]
            x_source_2, y_source_2 = None, None
                
            for i in range(self.shape[0] - 2, self.shape[0]):
                for j in range(self.shape[1]):
                    if state.targets[i][j] == value_2:
                        x_source_2, y_source_2 = i, j
                        break
                if x_source_2 is not None:
                    break
            if x_source_2 is None:
                print("Error: cannot find source 2")
            else:
                if y_source_2 == y_target and x_source_2 == x_target + 1 and \
                    y_source == y_target and x_source == x_target:
                    self.set_next_position(state, self.curr_pos)
                    return self.get_action(state)
            if x_source == self.shape[0] - 2:
                if x_cursor == self.shape[0] - 2:
                    return self.move_down()
                else:
                    if y_cursor < y_source:
                        return self.move_right()
                    elif y_cursor > y_source:
                        return self.move_left()
                    else:
                        return self.move_up()
            else:
                if y_target < y_source:
                    if x_cursor == self.shape[0] - 1:
                        return self.move_up();
                    else:
                        if y_cursor < y_source:
                            return self.move_right()
                        elif y_cursor > y_source:
                            return self.move_left()
                        else:
                            self.sequential_mode = True
                            for _ in range(y_source - y_target - 1):
                                self.action_list.extend(
                                    [self.move_left(), self.move_down(), 
                                        self.move_right(), self.move_up(),
                                        self.move_left()]
                                )
                            self.action_list.extend(
                                [self.move_left(), self.move_down(), 
                                    self.move_right()]
                            )
                            return self.get_action(state)
                else:
                    # x_target_2, y_target_2 = value_2 // self.shape[1], value_2 % self.shape[1]
                    if state.targets[self.shape[0] - 2][y_target] == value + self.shape[0]:
                        if x_cursor == self.shape[0] - 1:
                            return self.move_up()
                        self.sequential_mode = True
                        # left down right right up left left
                        self.action_list.extend(
                            [self.move_left(), self.move_down(),
                                self.move_right(), self.move_right(),
                                self.move_up(), self.move_left(), self.move_left()]
                        )
                        return self.get_action(state)
                    else:
                        self.sequential_mode = True
                        if x_source_2 == self.shape[0] - 2:
                            if x_cursor == self.shape[0] - 2:
                                self.action_list.append(self.move_down())
                            if y_source_2 - y_cursor > 0:
                                for _ in range(y_source_2 - y_cursor):
                                    self.action_list.append(self.move_right())
                            else:
                                for _ in range(y_cursor - y_source_2):
                                    self.action_list.append(self.move_left())
                            self.action_list.append(self.move_up())
                        else:
                            if x_cursor == self.shape[0] - 1:
                                self.action_list.append(self.move_up())
                            if y_source_2 - y_cursor > 0:
                                for _ in range(y_source_2 - y_cursor):
                                    self.action_list.append(self.move_right())
                            else:
                                for _ in range(y_cursor - y_source_2):
                                    self.action_list.append(self.move_left())
                        
                        for _ in range(y_source_2 - y_target - 1):
                            self.action_list.extend(
                                [self.move_left(), self.move_down(), 
                                    self.move_right(), self.move_up(),
                                    self.move_left()]
                            )
                        self.action_list.extend(
                            [self.move_left(), self.move_down(), 
                                self.move_right()]
                        )
                            
                        return self.get_action(state)
        else:
            if x_cursor == self.shape[0] - 2:
                if y_cursor == self.shape[0] - 2:
                    return self.move_down()
                else:
                    return self.move_left()
            elif y_cursor == self.shape[1] - 2:
                return self.move_right()
            
            values = [self.shape[1] * (self.shape[0] - 1) - 2,
                      self.shape[1] * (self.shape[0] - 1) - 1,
                      self.shape[1] * self.shape[0] - 2,
                        self.shape[1] * self.shape[0] - 1]
            if values[1] != state.targets[self.shape[0] - 2][self.shape[1] - 1]:
                    self.sequential_mode = True
                    self.action_list.extend(
                        [self.move_up(), self.move_left(),
                            self.move_down(), self.move_right()])
                    return self.get_action(state)
            else:
                if values[0] != state.targets[self.shape[0] - 2][self.shape[1] - 2]:
                    self.sequential_mode = True
                    self.action_list.extend(
                        [('select', (self.shape[0] - 2, self.shape[1] - 2)),
                            ('swap', (self.shape[0] - 2, self.shape[1] - 2, 
                                      self.shape[0] - 1, self.shape[1] - 2))])
                    return self.get_action(state)
            
            return self.get_action(state)     
        
    def run_sequential(self):
        action = self.action_list[0]
        self.action_list = self.action_list[1:]
        if len(self.action_list) == 0:
            self.sequential_mode = False
        return action
    
    def get_action(self, state):
        if self.n_fast_moves == 0:
            if self.curr_pos is None:
                self.set_cursor(state)
            else:
                self.curr_pos = [(state.n_selects - 1) // self.shape[1], 
                                (state.n_selects - 1) % self.shape[1]]
                self.set_next_position(state, self.curr_pos)
                self.set_cursor(state, source=state.n_selects)
            self.n_fast_moves -= 1
        elif self.n_fast_moves == -2:
            self.set_cursor(state)
            self.n_fast_moves = -1
        if self.n_fast_moves < 0:
            self.update_curr_pos(state)
            
        if self.sequential_mode:
            return self.run_sequential()
        
        if self.n_fast_moves > 0:
            self.sequential_mode = True
            self.set_cursor(state, state.n_selects, state.n_selects)
            x_source, y_source = self.curr_pos[0], self.curr_pos[1]
            value = state.targets[x_source][y_source]
            x_target, y_target = value // self.shape[1], value % self.shape[1]
            if x_source == x_target:
                for _ in range(y_source - y_target):
                    self.action_list.append(self.move_left())
            else:
                if y_source > y_target:
                    if y_source - y_target < y_target - (y_source - self.shape[1]):
                        for _ in range(y_source - y_target):
                            self.action_list.append(self.move_left())
                    else:
                        for _ in range(y_target - (y_source - self.shape[1])):
                            self.action_list.append(self.move_right())
                else:
                    if y_target - y_source < y_source - (y_target - self.shape[1]):
                        for _ in range(y_target - y_source):
                            self.action_list.append(self.move_right())
                    else:
                        for _ in range(y_source - (y_target - self.shape[1])):
                            self.action_list.append(self.move_left())
            for _ in range(x_source - x_target):
                self.action_list.append(self.move_up())
            self.n_fast_moves -= 1
            return self.get_action(state)
        
        x_source, y_source = self.curr_pos[0], self.curr_pos[1]
        x_cursor, y_cursor = self.cursor[0], self.cursor[1]
        value = state.targets[x_source][y_source]
        x_target, y_target = value // self.shape[1], value % self.shape[1]
        

        if x_target == self.shape[0] - 2:
            return self.execute_two_last_row(state)
        
        if x_source == x_target and y_source == y_target:
            self.set_next_position(state, self.curr_pos)
            return self.get_action(state)
        
        
        if x_source == x_target:
            if x_cursor == x_target:
                return self.move_down()
            if y_cursor != y_source:
                return self.move_right()
            else:
                self.action_list.append(self.move_up())
                if self.cursor[1] == self.shape[1] - 1:
                    self.action_list.append(self.move_left())
                else:
                    self.action_list.append(self.move_right())
                self.sequential_mode = True
                return self.get_action(state)
        
        if x_cursor != x_source and y_source == y_cursor:
            if x_cursor < x_source and y_target == self.shape[1] - 1:
                return self.move_down()
            return self.move_right()
            
        if x_source > x_cursor:
            return self.move_down()
        elif x_source < x_cursor:
            return self.move_up()
        elif y_source != y_target:
            if y_source > y_target:
                if y_source - y_target < y_target - (y_source - self.shape[1]):
                    if (x_cursor < self.shape[0] - 1) and (y_cursor == (y_source - 1 + self.shape[1]) % self.shape[1]):
                        self.sequential_mode =  True
                        self.action_list.extend(
                            [self.move_right(), self.move_down(), 
                                self.move_left(), self.move_left()])
                        return self.get_action(state)
                    else:
                        return self.move_right()
                else:
                    if (x_cursor < self.shape[0] - 1) and (y_cursor == (y_source + 1 + self.shape[1]) % self.shape[1]):
                        self.sequential_mode =  True
                        self.action_list.extend(
                            [self.move_left(), self.move_down(), 
                                self.move_right(), self.move_right()]
                        )
                        return self.get_action(state)
                    else:
                        return self.move_left()
            else:
                if y_target - y_source < y_source - (y_target - self.shape[1]):
                    
                    if (x_cursor < self.shape[0] - 1) and (y_cursor == (y_source + 1 + self.shape[1]) % self.shape[1]):
                        self.sequential_mode =  True
                        self.action_list.extend(
                            [self.move_left(), self.move_down(), 
                                self.move_right(), self.move_right()]
                        )
                        return self.get_action(state)
                    else:
                        return self.move_left()
                else:
                    if (x_cursor < self.shape[0] - 1) and (y_cursor == (y_source - 1 + self.shape[1]) % self.shape[1]):
                        self.sequential_mode =  True
                        self.action_list.extend(
                            [self.move_right(), self.move_down(), 
                                self.move_left(), self.move_left()]
                        )
                        return self.get_action(state)
                    else:
                        return self.move_right()
        else:
            if abs(y_cursor -  y_target + self.shape[1]) % self.shape[1] > 1:
                return self.move_left()
        if (y_target == self.shape[1] - 1):
            if (y_source + 1) % self.shape[1] == y_cursor:
                self.sequential_mode =  True
                for i in range(x_source - x_target - 1):
                    self.action_list.extend(
                        [self.move_up(), self.move_left(), 
                            self.move_down(), self.move_right(),
                            self.move_up()]
                    )
                for i in range(self.shape[1] - 3):
                    self.action_list.append(self.move_right())
                self.action_list.extend(
                    [
                        self.move_up(),
                        self.move_right(),
                        self.move_right(),
                        self.move_down(),
                        self.move_left(),
                        self.move_up(),
                        self.move_left(),
                        self.move_down()
                        ]   
                )
                if self.cursor[0] == self.shape[0] - 1:
                    self.action_list.extend(
                        [
                            self.move_right(),
                            self.move_right(),
                        ]
                    )
                return self.get_action(state)
        elif y_source != y_cursor - 1 :
            return self.move_left()
        else:
            self.sequential_mode =  True
            for i in range(x_source - x_target):
                self.action_list.extend(
                    [self.move_up(), self.move_left(), 
                           self.move_down(), self.move_right(),
                           self.move_up()]
                )
            return self.get_action(state)
        