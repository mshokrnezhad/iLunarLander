#       This is the code of Actor Deep Network (ADN). 
#1:     To work with its basic functionalities.
#2:     To implement neural layers.
#3:     To implement activation functions.
#4:     To optimize the weights.
#5:     To create the action probability distribution function.
#6:     Calling the constructor of the parent class "nn.Module".
#7:     * unpacks the elements from the input_size tuple. Imagine input_size is a tuple like (3, 4). 
#       The asterisk (*) expands it into separate arguments like nn.Linear(3, 4, 128).
#8:     This line moves the AGN model (including its layers and optimizer) to the device specified in self.device. 
#       This ensures the model's operations are performed on the chosen device (GPU or CPU) for efficient training and inference.
#9:     The Layers and their connection structure is defined based on the source paper (see README.md).
#10:    If isReparamEnabled is True, rsample() is used for reparameterization, which allows gradients to flow through the sampling process.
#       Example: For mu = [0.5, -0.2] and sigma = [0.1, 0.1], a sample might be [0.52, -0.18]. 
#       Note that we have more than one action, and we have a mu and sigma for each action.
#11:    Applies the tanh function to the sampled action to squash it to the range [-1, 1], then scales it by actions_max.
#       Example: If actions_ = [0.52, -0.18], T.tanh(actions_) might be [0.478, -0.178], and the scaled action would be [0.478, -0.178], where the scaler is 1.
#12:    sum(1) means summing along the second dimension (i.e., summing the log probabilities of each action dimension to get a single log probability per action).
#       keepdim=True ensures that the resulting tensor maintains the same number of dimensions as the original, 
#       but with a size of 1 in the summed dimension. This is useful for maintaining the correct shape for further calculations.

import numpy as np
import torch as T #1
import torch.nn as nn #2
import torch.nn.functional as F #3
import torch.optim as optim #4 
from torch.distributions import Normal #5

class ADN(nn.Module):
    def __init__(self, learning_rate, input_size, fcl1_size, fcl2_size, actions_num, actions_max, model_file):
        super(ADN, self).__init__() #6        
        self.fcl1 = nn.Linear(*input_size, fcl1_size) #7
        self.fcl2 = nn.Linear(fcl1_size, fcl2_size)
        self.mu = nn.Linear(fcl2_size, actions_num)
        self.sigma = nn.Linear(fcl2_size, actions_num)
        
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device) #8
        self.model_file = model_file
        self.actions_max = actions_max
        self.noise = 1e-6
        
    def forward(self, state): #9
        state = state.to(self.device)
        state_value = self.fcl1(state)
        state_value = F.relu(state_value)
        state_value = self.fcl2(state_value)
        state_value = F.relu(state_value)
        
        mu = self.mu(state_value)
        sigma = self.sigma(state_value)
        sigma = T.clamp(sigma, min = self.noise, max = 1)

        return mu, sigma
    
    def sample_action(self, state, isReparamEnabled = True):
        mu, sigma = self.forward(state)
        action_pdf = Normal(mu, sigma)
        
        if(isReparamEnabled):
            actions_ = action_pdf.rsample() #10
        else:
            actions_ = action_pdf.sample()
        
        actions_max = T.tensor(self.actions_max).to(self.device)
        actions = T.tanh(actions_) * actions_max #11
        
        log_probabilities = action_pdf.log_prob(actions_)
        log_probabilities -= T.log(1 - actions.pow(2) + self.noise)
        log_probabilities = log_probabilities.sum(1, keepdim = True) #12
        
        return actions, log_probabilities
        
    def save_model(self):
        print(f'Saving {self.model_file}...')
        T.save(self.state_dict(), self.model_file)

    def load_model(self):
        print(f'Loading {self.model_file}...')
        self.load_state_dict(T.load(self.model_file, map_location = T.device('cpu')))