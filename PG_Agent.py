#In this file, we implement an agent which is working based on PGN.
import numpy as np
import torch as T #to work with its basic functionalities
import torch.nn as nn #to implement neural layers
import torch.nn.functional as F #to implement activation functions
import torch.optim as optim #to optimize the weights
from PGN import PGN

class PGN_Agent():
    def __init__(self, learning_rate, input_dim, actions_num = 4, gamma = 0.99):
        self.gamma = gamma
        self.reward_memory = []
        self.action_memory = []
        self.PGN = PGN(learning_rate, input_dim, actions_num)
        
    def store_reward(self, reward):
        self.reward_memory.append(reward)
        
    def store_action(self, action):
        self.action_memory.append(action)
        
    def act(self, state):
        state = T.Tensor([state]).to(self.PGN.device) #to send the state in the form of Tensor to the device selected in PGN
        action_values = self.PGN.forward(state) #the values estimated for all actions
        action_probabilities = F.softmax(action_values) #softmax takes the action_value and converts them into probabilities. the output is a probability distribution where the sum of all probabilities for each action is 1.
        actions_chances = T.distributions.Categorical(action_probabilities) #This line creates a categorical distribution object, representing the probability of choosing each action based on the calculated action_probabilities.
        action = actions_chances.sample() 
        action_log = actions_chances.log_prob(action) #This line calculates the log probability of the chosen action for calculating loss during training.
        
        self.store_action(action_log)        