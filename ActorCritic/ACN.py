#       This is the code of Actor-Critic Network (ACN).
#1:     To work with its basic functionalities.
#2:     To implement neural layers.
#3:     To implement activation functions.
#4:     To optimize the weights.
#5:     Calling the constructor of the parent class "nn.Module".
#6:     * unpacks the elements from the input_size tuple. Imagine input_size is a tuple like (3, 4). 
#       The asterisk (*) expands it into separate arguments like nn.Linear(3, 4, 128).
#7:     This line moves the AGN model (including its layers and optimizer) to the device specified in self.device. 
#       This ensures the model's operations are performed on the chosen device (GPU or CPU) for efficient training and inference.

import numpy as np
import torch as T #1
import torch.nn as nn #2
import torch.nn.functional as F #3
import torch.optim as optim #4 

class ACN(nn.Module):
    def __init__(self, learning_rate, input_size, actions_num, fc1_size = 256, fc2_size = 256):
        super(ACN, self).__init__() #5
        self.fcl1 = nn.Linear(*input_size, fc1_size) #6 
        self.fcl2 = nn.Linear(fc1_size, fc2_size)
        self.pi = nn.Linear(fc2_size, actions_num)
        self.v = nn.Linear(fc2_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device) #7 
            
    def forward(self, state):
        x = F.relu(self.fcl1(state))
        x = F.relu(self.fcl2(x))
        pi = self.pi(x)
        v = self.v(x)
        
        return (pi, v)
    
    
    
    
    
    
    
    
        
        
        
        
