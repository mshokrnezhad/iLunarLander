#this is the code of Policy Gradient Network (PGN)
import numpy as np
import torch as T #to work with its basic functionalities
import torch.nn as nn #to implement neural layers
import torch.nn.functional as F #to implement activation functions
import torch.optim as optim #to optimize the weights

class PGN(nn.Module):
    def __init__(self, learning_rate, input_dim, actions_num):
        super(PGN, self).__init__() #calling the constructor of the parent class "nn.Module"
        self.fcl1 = nn.Linear(*input_dim, 128) #* unpacks the elements from the input_dim tuple. Imagine input_dim is a tuple like (3, 4). The asterisk (*) expands it into separate arguments like nn.Linear(3, 4, 128).
        self.fcl2 = nn.Linear(128, 128)
        self.fcl3 = nn.Linear(128, actions_num)
        self.optim = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device) #This line moves the PGN model (including its layers and optimizer) to the device specified in self.device. This ensures the model's operations are performed on the chosen device (GPU or CPU) for efficient training and inference.
    
    def forward(self, state):
        x = F.relu(self.fcl1(state))
        x = F.relu(self.fcl2(x))
        x = self.fcl3(x)
        
        return x        
        
        
        
        
