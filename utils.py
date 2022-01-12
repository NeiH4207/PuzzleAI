'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
from threading import Thread
import time
import numpy as np
import torch
from random import seed

seed(2)

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
    
def GetCeilIndex(arr, T, l, r, key):
    while (r - l > 1):
        m = l + (r - l)//2
        if (arr[T[m]] >= key):
            r = m
        else:
            l = m
    return r
  
def LongestIncreasingSubsequence(arr, n):
    tailIndices =[0 for i in range(n + 1)] 
    prevIndices =[-1 for i in range(n + 1)] 
    len = 1
    for i in range(1, n):
        if (arr[i] < arr[tailIndices[0]]):
            # new smallest value
            tailIndices[0] = i
        elif (arr[i] > arr[tailIndices[len-1]]):
            # arr[i] wants to extend
            # largest subsequence
            prevIndices[i] = tailIndices[len-1]
            tailIndices[len] = i
            len += 1
        else:
            pos = GetCeilIndex(arr, tailIndices, -1,
                                   len-1, arr[i])
            prevIndices[i] = tailIndices[pos-1]
            tailIndices[pos] = i
         
    indexes = []
    i = tailIndices[len-1]
    while(i >= 0):
        indexes.append(i)
        i = prevIndices[i]
    
    # reverse indexes to get actual values in arr
    indexes.reverse()
    return indexes
 
# # driver code
# arr = [37, 34, 35, 33, 38, 32, 36, 39]
# n = len(arr)
# lis = LongestIncreasingSubsequence(arr, n)
# move_list = [arr[i] for i in range(len(arr)) if i not in lis]
# print(move_list)
# p = np.argsort(move_list)[::-1]
# for v in p:
#     num = move_list[v]
#     st_index = 0
#     max_num_index = len(arr)
#     min_num_index = -1
#     for i in range(v + 1, len(arr)):
#         if arr[i] > num and max_num_index == len(arr):
#             max_num_index = i
#         if arr[i] == num:
#             st_index = i
#     # down
#     for i in range(v, 0, -1):
#         if arr[i] < num and min_num_index == -1:
#             min_num_index = i
    
#     if (st_index < max_num_index - 1):   
#         while(st_index < max_num_index - 1):
#             arr[st_index], arr[st_index + 1] = arr[st_index + 1], arr[st_index]
#             st_index += 1
#             print(arr)
#     else:
#         while(st_index > min_num_index + 1):
#             arr[st_index], arr[st_index - 1] = arr[st_index - 1], arr[st_index]
#             st_index -= 1
#             print(arr)
        
# print(arr)