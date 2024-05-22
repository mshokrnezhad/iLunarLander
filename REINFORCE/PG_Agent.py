#       In this file, we implement an agent which is working based on PGN.
#1:     To work with its basic functionalities.
#2:     To implement neural layers.
#3:     To implement activation functions.
#4:     To optimize the weights.
#5:     To send the state in the form of Tensor to the device selected in PGN.
#6:     The values estimated for all actions.
#7:     softmax takes the action_value and converts them into probabilities. 
#       The output is a probability distribution where the sum of all probabilities for each action is 1.
#8:     This line creates a categorical distribution object, representing the probability of choosing each action based on the calculated action_probabilities.
#9:     This line calculates the log probability of the chosen action for calculating loss during training.
#10:    G_t = sum_{form k=0 to k=T) gamma**k * R_{t+k+1}. 
#       Note that I assume R_{t+1} is stored in reward_memory[t], as it is filled after the action at time slot t has taken place. 
#11:    G will be a zero numpy array with the shape of self.reward_memory.
#12:    REINFORCE generally iterates over t in T steps, with a G_t assigned to each step. 
#       In lunar_lander, for each t, there are many sub-steps t_ from the lander's initial position in the sky to the point where it touches the surface. 
#       So, in order to calculate G_t, some loops over t_s are required. 
#13:    To convert the np array to tensor and send it to the assigned device.
#14:    zip creates an iterator that combines elements from both lists at the same index into tuples. 
#       The loop iterates over this zip object, assigning each element of the tuple to separate variables. 
#       Imagine you have two lists: A = [1, 2, 3] and B = ["left", "right", "up"].
#       Using for a, b in zip(A, B), in the first iteration, a will be assigned the value 1 and b will be assigned the value "left".
#       In the second iteration, a will be assigned the value 2, and b will be assigned the value "right". 
#       And so on, iterating through all corresponding elements in both lists.
#15:    To calculate gradients.
#16:    To update NN weights. 
  
import numpy as np
import torch as T #1
import torch.nn as nn #2
import torch.nn.functional as F #3
import torch.optim as optim #4 
from PGN import PGN

class PG_Agent():
    def __init__(self, learning_rate, gamma, input_size, actions_num):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.reward_memory = []
        self.action_memory = []
        self.PGN = PGN(learning_rate, input_size, actions_num)
        
    def store_reward(self, reward):
        self.reward_memory.append(reward)
        
    def store_action(self, action):
        self.action_memory.append(action)
        
    def act(self, state):
        state = T.tensor([state], dtype=T.float32).to(self.PGN.device) #5
        action_values = self.PGN.forward(state) #6
        action_probabilities = F.softmax(action_values) #7
        actions_chances = T.distributions.Categorical(action_probabilities) #8
        action = actions_chances.sample() 
        action_log = actions_chances.log_prob(action) #9
        self.store_action(action_log)     
        
        return action.item()
        
    def learn(self):
        self.PGN.optimizer.zero_grad()
        
        G_t = np.zeros_like(self.reward_memory, dtype=np.float64) #10 and 11
        for t_ in range(len(self.reward_memory)): #12
            G_t_ = 0
            discount = 1
            for k in range(t_, len(self.reward_memory)):
                G_t_ += self.reward_memory[k] * discount
                discount *= self.gamma
            G_t[t_] = G_t_
        G_t = T.tensor(G_t, dtype=T.float32).to(self.PGN.device) #13
        
        loss = 0
        for g_t_, log_probability in zip(G_t, self.action_memory): #14
            loss += -g_t_ * log_probability
        
        loss.backward() #15
        self.PGN.optimizer.step() #16
        
        self.action_memory = []
        self.reward_memory = []
        