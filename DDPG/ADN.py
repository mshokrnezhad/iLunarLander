#       This is the code of Actor Deep Network (ADN). 
#1:     To work with its basic functionalities.
#2:     To implement neural layers.
#3:     To implement activation functions.
#4:     To optimize the weights.
#5:     Calling the constructor of the parent class "nn.Module".
#6:     * unpacks the elements from the input_size tuple. Imagine input_size is a tuple like (3, 4). 
#       The asterisk (*) expands it into separate arguments like nn.Linear(3, 4, 128).
#7:     When we apply the layer_norm to the input_tensor, it normalizes each feature across the individual data samples using 
#       normalized_feature = (feature - mean) / std. FOr example, the normalized version of [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] 
#       is [[-1.2247,  0.0000,  1.2247], [-1.2247,  0.0000,  1.2247]]. 
#8:     The . after 1 ensures that the result of the division will be a floating-point number, regardless of the type of x.
#9:     To initialize the layer with normalized initial values. 
#10:    This line moves the AGN model (including its layers and optimizer) to the device specified in self.device. 
#       This ensures the model's operations are performed on the chosen device (GPU or CPU) for efficient training and inference.
#11:    The Layers and their connection structure is defined based on the source paper (see README.md).
#12:    To apply the hyperbolic tangent (tanh) activation function to the tensor x, 
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
        self.bnl1 = nn.LayerNorm(fcl1_size) #7
        self.bnl2 = nn.LayerNorm(fcl2_size)
        self.mu = nn.Linear(fcl2_size, actions_num)
        
        f1 = 1./np.sqrt(self.fcl1.weight.data.size()[0]) #8
        self.fcl1.weight.data.uniform_(-f1, f1) #9
        self.fcl1.bias.data.uniform_(-f1, f1) #9
        
        f2 = 1./np.sqrt(self.fcl2.weight.data.size()[0])
        self.fcl2.weight.data.uniform_(-f2, f2)
        self.fcl2.bias.data.uniform_(-f2, f2)
        
        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)
        
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cuda:1")
        self.to(self.device) #10
        self.model_file = model_file
        
    def forward(self, state): #11
        x = self.fcl1(state)
        x = self.bnl1(x)
        x = F.relu(x)
        x = self.fcl2(x)
        x = self.bnl2(x)
        x = F.relu(x)
        x = self.mu(x)
        x = T.tanh(x) #12
        
        return x
        
    def save_model(self):
        print(f'Saving {self.model_file}...')
        T.save(self.state_dict(), self.model_file)

    def load_model(self):
        print(f'Loading {self.model_file}...')
        self.load_state_dict(T.load(self.model_file, map_location = T.device('cpu')))