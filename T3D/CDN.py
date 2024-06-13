#       This is the code of Critic Deep Network (CDN). 
#1:     To work with its basic functionalities.
#2:     To implement neural layers.
#3:     To implement activation functions.
#4:     To optimize the weights.
#5:     Calling the constructor of the parent class "nn.Module".
#6:     input_size[0] extracts the first element of the input_size tuple, which in this context represents the number of features of the state vector.
#7:     This line moves the AGN model (including its layers and optimizer) to the device specified in self.device. 
#       This ensures the model's operations are performed on the chosen device (GPU or CPU) for efficient training and inference.
#8:     The Layers and their connection structure is defined based on the source paper (see README.md).
#9      T.cat concatenates (joins) a sequence of tensors along a specified dimension. For eaxmple, where state = T.randn(64, 8) (a batch of 64 states, 
#       each with 8 features) and action = T.randn(64, 2) (a batch of 64 actions, each with 2 features), combined = T.cat([state, action], dim=1) 
#       will be torch.Size([64, 10]).

import numpy as np
import torch as T #1
import torch.nn as nn #2
import torch.nn.functional as F #3
import torch.optim as optim #4 

class CDN(nn.Module):
    def __init__(self, learning_rate, input_size, fcl1_size, fcl2_size, actions_num, model_file):
        super(CDN, self).__init__() #5
        self.fcl1 = nn.Linear(input_size[0] + actions_num, fcl1_size) #6 
        self.fcl2 = nn.Linear(fcl1_size, fcl2_size)
        self.Q1 = nn.Linear(fcl2_size, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device) #7
        self.model_file = model_file
        
    def forward(self, state, action): #8
        action_value = self.fcl1(T.cat([state, action], dim = 1)) #9
        action_value = F.relu(action_value)
        action_value = self.fcl2(action_value)
        action_value = F.relu(action_value)
        action_value = self.Q1(action_value)
        
        return action_value
        
    def save_model(self):
        print(f'Saving {self.model_file}...')
        T.save(self.state_dict(), self.model_file)

    def load_model(self):
        print(f'Loading {self.model_file}...')
        self.load_state_dict(T.load(self.model_file, map_location = T.device('cpu')))