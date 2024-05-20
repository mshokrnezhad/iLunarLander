#       In this file, we implement an agent which is working based on PGN.
#1:     To send the state in the form of Tensor to the device selected in PGN.
#2:     The values estimated for all actions.
#3:     softmax takes the action_value and converts them into probabilities. 
#       The output is a probability distribution where the sum of all probabilities for each action is 1.
#4:     This line creates a categorical distribution object, representing the probability of choosing each action based on the calculated action_probabilities.
#5:     This line calculates the log probability of the chosen action for calculating loss during training.
#6:     G_t = sum_{form k=0 to k=T) gamma**k * R_{t+k+1}. 
#       Note that I assume R_{t+1} is stored in reward_memory[t], as it is filled after the action at time slot t has taken place. 
#7:     G will be a zero numpy array with the shape of self.reward_memory.
#8:     To convert the np array to tensor and send it to the assigned device.
#9:     zip creates an iterator that combines elements from both lists at the same index into tuples. 
#       The loop iterates over this zip object, assigning each element of the tuple to separate variables. 
#       Imagine you have two lists: A = [1, 2, 3] and B = ["left", "right", "up"].
#       Using for a, b in zip(A, B), in the first iteration, a will be assigned the value 1 and b will be assigned the value "left".
#       In the second iteration, a will be assigned the value 2, and b will be assigned the value "right". 
#       And so on, iterating through all corresponding elements in both lists.
#10:    To calculate gradients.
#11:    To update NN weights. 
  
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
        state = T.Tensor([state]).to(self.PGN.device) #1
        action_values = self.PGN.forward(state) #2
        action_probabilities = F.softmax(action_values) #3
        actions_chances = T.distributions.Categorical(action_probabilities) #4
        action = actions_chances.sample() 
        action_log = actions_chances.log_prob(action) #5
        
        self.store_action(action_log)     
        
    def learn(self):
        self.PGN.optimizer.zero_grad()
        
        G = np.zeros_like(self.reward_memory) #6 and 7
        for t in range(len(self.reward_memory)):
            G_ = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_ += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_
        G = T.Tensor(G, dtype=T.float).to(self.PGN.device) #8
        
        loss = 0
        for g, log_probability in zip(G, self.action_memory): #9
            loss += -g * log_probability
        
        loss.backward() #10
        self.PGN.optimizer.step() #11
        
        self.action_memory = []
        self.reward_memory = []
        