#       In this file, we implement an agent which is working based on ACN. 
#       Note that storing rewards and actions is unnecessary in this context, as the approach relies on the temporal difference method.
#1:     To work with its basic functionalities.
#2:     To implement neural layers.
#3:     To implement activation functions.
#4:     To optimize the weights.
#5:     To send the state in the form of Tensor to the device selected in PGN.
#6:     The values estimated for all actions (pi in the return statement of ACN.forward).
#7:     softmax takes the action_value and converts them into probabilities. 
#       The output is a probability distribution where the sum of all probabilities for each action is 1.
#8:     This line creates a categorical distribution object, representing the probability of choosing each action based on 
#       the calculated action_probabilities.
#9:     This line calculates the log probability of the chosen action for calculating loss during training.
#10     Since we do not have memory in this case, [state, reward, state_, info] is directly passed to the learn function.
#11:    To convert the array to tensor and send it to the assigned device.
#12:    Regarding the step of the algorithm that involves updating the actor, the goal is to enhance its action selection. 
#       This is achieved by maximizing the probability of selecting the best action (action_log) 
#       multiplied by the expected value of that action (reward + V_ in the delta calculation). 
#       So, A good loss function could be expressed as -delta * self.action_log 
#13:    In the algorithm step concerning the update of the critic, the goal of optimizing the critic is to enhance the accuracy of 
#       the critic's value estimates. Therefore, an effective loss function would quantify the discrepancy between the critic's 
#       value estimates (V_) and the actual returns (V), which is gamma^2.
#14:    To calculate gradients. If we had two separate neural networks for the actor and the critic, this step should be done 
#       for each of them separately.
#15:    To update neural network weights. 

import numpy as np
import torch as T #1
import torch.nn as nn #2
import torch.nn.functional as F #3
import torch.optim as optim #4 
from ACN import ACN

class AC_Agent():
    def __init__(self, learning_rate, gamma, input_size, fcl1_size, fcl2_size, actions_num, model_file): 
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.fcl1_size = fcl1_size
        self.fcl2_size = fcl2_size
        self.ACN = ACN(learning_rate, input_size, fcl1_size, fcl2_size, actions_num, model_file)
        self.action_log = None
        
        
    def act(self, state):
        state = T.tensor([state], dtype=T.float).to(self.ACN.device) #5
        action_values, _ = self.ACN.forward(state) #6
        action_probabilities = F.softmax(action_values) #7
        actions_chances = T.distributions.Categorical(action_probabilities) #8
        action = actions_chances.sample() 
        action_log = actions_chances.log_prob(action) #9
        self.action_log = action_log
        
        return action.item()
        
    def learn(self, state, reward, state_, done): #10
        self.ACN.optimizer.zero_grad()

        state = T.tensor([state], dtype=T.float).to(self.ACN.device) #11
        state_ = T.tensor([state_], dtype=T.float).to(self.ACN.device) #11
        reward = T.tensor([reward], dtype=T.float).to(self.ACN.device) #11
        
        _, V = self.ACN.forward(state)
        _, V_ = self.ACN.forward(state_)
        V_ = V_ * (1 - int(done))
        
        delta = reward + self.gamma * V_ - V
        
        actor_loss = -delta * self.action_log #12
        critic_loss = delta**2 #13
        
        (actor_loss + critic_loss).backward() #14
        self.ACN.optimizer.step() #15
        