#       In this file, we implement an agent which is working based on ADN and CDN.
#1:     OADN: Online Actor Deep Network
#2:     OCDN: Online Critic Deep Network
#3:     TADN: Target Actor Deep Network
#4:     TCDN: Target Critic Deep Network
#5:     self.OADN.eval() (activating the evaluation mode of the system) might not have a significant impact on the results, 
#       as LayerNorm itself (in ADN) does not change behavior between training and evaluation modes. 
#       However, to ensure consistency, reproducibility, and to follow best practices, 
#       it’s important to set the network to evaluation mode during inference. 
#       This practice is especially crucial if the network might include other types of layers in the future.
#6:     To send the state in the form of Tensor to the device selected in ADN.
#7:     It dose not explicitly need to send mu to the device because mu is already on the correct device. 
#       This is because mu is obtained as the output of self.OADN.forward(state), 
#       and since state was moved to the same device as the model (self.OADN.device), the output mu will also be on that device.
#8:     By moving noise to the same device as mu, you ensure the addition operation works correctly.
#9:     To go back to the train mode.
#10:    mu_ is a PyTorch tensor that contains the action computed by the actor network plus the added noise. 
#       .cpu() moves the tensor mu_ to the CPU. This is necessary because PyTorch tensors might be on a GPU for faster computation, 
#       but to convert them to a NumPy array, they need to be on the CPU.
#       .detach() detaches the tensor from the computation graph. This means that the tensor mu_ will no longer track 
#       operations for gradient computation. This is important because we’re only interested in the value of the tensor, 
#       not in computing gradients at this point.
#       .numpy() converts the PyTorch tensor to a NumPy array. Many environments and libraries outside of PyTorch expect 
#       data in NumPy format, so this conversion makes the data compatible with other tools and libraries.
#       [0] extracts the first element from the NumPy array. Since mu_ was originally a tensor with an extra batch dimension (of size 1), 
#       mu_.cpu().detach().numpy() results in a NumPy array with shape (1, actions_num). 
#       The [0] indexing removes the batch dimension, resulting in an array of shape (actions_num,).

import numpy as np
import torch as T
import torch.nn.functional as F
from ADN import ADN
from CDN import CDN
from OU_Noise import OU_Noise
from Memory import Memory

class DDPG_Agent():
    def __init__(self, a_lr, c_lr, gamma, tau, input_size, fcl1_size, fcl2_size, actions_num, memory_size, batch_size, a_mf, c_mf):
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.memory = Memory(memory_size, input_size, actions_num)
        self.noise = OU_Noise(mu = np.zeros(actions_num))
        self.online_ADN = ADN(a_lr, input_size, fcl1_size, fcl2_size, actions_num, a_mf) #1
        self.online_CDN = CDN(c_lr, input_size, fcl1_size, fcl2_size, actions_num, c_mf) #2
        self.target_ADN = ADN(a_lr, input_size, fcl1_size, fcl2_size, actions_num, a_mf) #3
        self.target_CDN = CDN(c_lr, input_size, fcl1_size, fcl2_size, actions_num, c_mf) #4
        
        self.update_targets(1)
        
    def act(self, state):
        self.online_ADN.eval() #5
        state = T.tensor([state], dtype=T.float).to(self.online_ADN.device) #6
        mu = self.online_ADN.forward(state).to(self.online_ADN.device) #7
        noise = T.tensor([self.noise()], dtype=T.float).to(self.online_ADN.device) #8
        mu_ = mu + noise 
        self.online_ADN.train() #9
        
        return mu_.cpu().detach().numpy()[0] #10
    
    def save_models(self):
        self.online_ADN.save_model()
        self.online_CDN.save_model()
        self.target_ADN.save_model()
        self.target_CDN.save_model()
        
    def load_models(self):
        self.online_ADN.load_model()
        self.online_CDN.load_model()
        self.target_ADN.load_model()
        self.target_CDN.load_model()
        
    def learn(self):
        if self.memory.index < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
        states = T.tensor(states, dtype=T.float).to(self.online_ADN.device)
        actions = T.tensor(actions, dtype=T.float).to(self.online_ADN.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.online_ADN.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.online_ADN.device)
        dones = T.tensor(dones).to(self.online_ADN.device)
        
        target_V_ = self.target_ADN.forward(states_)
        target_q_ = self.target_CDN.forward(states_, tV_)
        online_q = self.online_CDN.forward(states, actions)
        
        target_q_[dones] = 0.0
        target_q_ = target_q_.view(-1)
        
        target = rewards + self.gamma * target_q_
        target = target.view(self.batch_size, 1)
        
        self.online_CDN.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, online_q)
        critic_loss.backward()
        self.online_CDN.optimizer.step()

        self.online_ADN.optimizer.zero_grad()
        actor_loss = -self.online_CDN.forward(states, self.online_ADN.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.online_ADN.optimizer.step()
        
        self.update_targets()       
        
        
    def update_targets(self, t):
        pass
    
    