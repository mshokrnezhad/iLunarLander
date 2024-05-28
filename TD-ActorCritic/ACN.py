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
    def __init__(self, learning_rate, input_size, fcl1_size, fcl2_size, actions_num, model_file):
        super(ACN, self).__init__() #5
        self.fcl1 = nn.Linear(*input_size, fcl1_size) #6 
        self.fcl2 = nn.Linear(fcl1_size, fcl2_size)
        self.pi = nn.Linear(fcl2_size, actions_num)
        self.v = nn.Linear(fcl2_size, 1)
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device) #7 
        self.model_file = model_file
            
    def forward(self, state):
        x = F.relu(self.fcl1(state))
        x = F.relu(self.fcl2(x))
        pi = self.pi(x)
        v = self.v(x)
        
        return (pi, v)
    
    def save_model(self):
        print(f'Saving {self.model_file}...')
        T.save(self.state_dict(), self.model_file)

    def load_model(self):
        print(f'Loading {self.model_file}...')
        self.load_state_dict(T.load(self.model_file, map_location = T.device('cpu')))
    
    
    
    
    
    
    
    
        
        
        
        
