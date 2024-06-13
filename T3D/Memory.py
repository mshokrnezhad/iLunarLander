#       In this file, we develop the memory to implement the replay buffer.

import numpy as np

class Memory():
    def __init__(self, size, input_size, actions_num):
        self.size = size
        self.index = 0
        self.state_store = np.zeros((size, *input_size))
        self.resulted_state_store = np.zeros((size, *input_size))
        self.action_store = np.zeros((size, actions_num))
        self.reward_store = np.zeros(size)
        self.done_store = np.zeros(size, dtype=bool)
    
    def store(self, state, action, reward, state_, done):
        index_ = self.index % self.size
        self.state_store[index_] = state        
        self.action_store[index_] = action        
        self.reward_store[index_] = reward       
        self.resulted_state_store[index_] = state_        
        self.done_store[index_] = done
        
        self.index += 1
        
    def sample(self, batch_size):
        size_ = min(self.size, self.index)
        batch_ = np.random.choice(size_, batch_size)
        
        states = self.state_store[batch_]
        actions = self.action_store[batch_]
        rewards = self.reward_store[batch_]
        states_ = self.resulted_state_store[batch_]
        dones = self.done_store[batch_]
        
        return states, actions, rewards, states_, dones
    
    
    
        