#       This is the code of Actor Deep Network (ADN). 
#1:     To work with its basic functionalities.
#2:     To implement neural layers.
#3:     To implement activation functions.
#4:     To optimize the weights.
#5:     Calling the constructor of the parent class "nn.Module".
#6:     * unpacks the elements from the input_size tuple. Imagine input_size is a tuple like (3, 4). 
#       The asterisk (*) expands it into separate arguments like nn.Linear(3, 4, 128).
#7:     This line moves the AGN model (including its layers and optimizer) to the device specified in self.device. 
#       This ensures the model's operations are performed on the chosen device (GPU or CPU) for efficient training and inference.
#8:     The Layers and their connection structure is defined based on the source paper (see README.md).
#9:     To apply the hyperbolic tangent (tanh) activation function to the tensor x, 
#       that is sinh(x) / cosh(x) = (e^x - e^{-x}) / (e^x + e^{-x}). For example, the tanh of 
#       [-1.0000,  0.0000,  1.0000], [ 2.0000, -2.0000,  0.5000]] is [-0.7616,  0.0000,  0.7616], [ 0.9640, -0.9640,  0.4621]].

import numpy as np
import torch as T #1
import torch.nn as nn #2
import torch.nn.functional as F #3
import torch.optim as optim #4 

class ADN(nn.Module):
    def __init__(self, learning_rate, input_size, fcl1_size, fcl2_size, actions_num, model_file):
        super(ADN, self).__init__() #5
        self.fcl1 = nn.Linear(*input_size, fcl1_size) #6 
        self.fcl2 = nn.Linear(fcl1_size, fcl2_size)
        self.mu = nn.Linear(fcl2_size, actions_num)
        
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device) #7
        self.model_file = model_file
        
    def forward(self, state): #8
        state = state.to(self.device)
        s = self.fcl1(state)
        s = F.relu(s)
        s = self.fcl2(s)
        s = F.relu(s)
        
        mu = self.mu(s)
        
        mu = T.tanh(mu) #9
        
        return mu
        
    def save_model(self):
        print(f'Saving {self.model_file}...')
        T.save(self.state_dict(), self.model_file)

    def load_model(self):
        print(f'Loading {self.model_file}...')
        self.load_state_dict(T.load(self.model_file, map_location = T.device('cpu')))