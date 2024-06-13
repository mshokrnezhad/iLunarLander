#       In this file, we implement an T3D-based agent which is working based on ADN and CDN.
#1:     ADN.eval() (activating the evaluation mode of the system) might not have a significant impact on the results.
#       However, to ensure consistency, reproducibility, and to follow best practices, it’s important to set the network to evaluation mode during inference. 
#       This practice is especially crucial if the network might include other types of layers in the future.
#2:     To send the state in the form of Tensor to the device selected in ADN.
#3:     It dose not explicitly need to send mu_ to the device because mu_ is already on the correct device. 
#       This is because mu_ is obtained as the output of ADN.forward(state), 
#       and since state was moved to the same device as the model (ADN.device), the output mu_ will also be on that device.
#4:     By moving noise to the same device as mu_, you ensure the addition operation works correctly.
#5:     T.clamp: This function clamps all elements in the input tensor mu into the range [min, max]. Any values in mu less than min are set to min, 
#       and any values greater than max are set to max.
#6:     To go back to the train mode.
#7:     mu is a PyTorch tensor that contains the action computed by the actor network plus the added noise. 
#       .cpu() moves the tensor mu to the CPU. This is necessary because PyTorch tensors might be on a GPU for faster computation, 
#       but to convert them to a NumPy array, they need to be on the CPU.
#       .detach() detaches the tensor from the computation graph. This means that the tensor mu will no longer track 
#       operations for gradient computation. This is important because we’re only interested in the value of the tensor, 
#       not in computing gradients at this point.
#       .numpy() converts the PyTorch tensor to a NumPy array. Many environments and libraries outside of PyTorch expect 
#       data in NumPy format, so this conversion makes the data compatible with other tools and libraries.
#8:     Check the source paper and README.md to see the logic of self.learn().
#9:     When target_Q_ is computed, it might have a shape that includes an extra dimension, 
#       making it a two-dimensional tensor with shape (batch_size, 1). 
#       For some operations, it’s necessary to have this tensor as a one-dimensional tensor (a vector) with shape (batch_size,). 
#       This ensures that arithmetic operations like addition with rewards and element-wise multiplication with self.gamma work correctly 
#       and that the resulting tensor can be reshaped back to (batch_size, 1) without issues. For example, 
#       target_Q_ = tensor([[0.5], [0.8], [0.3], [1.2]]) afeter .view(-1) would be tensor([0.5, 0.8, 0.3, 1.2]).
#10     .clone() method is used to create a copy of a tensor. 
#       This ensures that any subsequent operations on the cloned tensor do not affect the original tensor.

import numpy as np
import torch as T
import torch.nn.functional as F
from ADN import ADN
from CDN import CDN
from Memory import Memory

class T3D_Agent():
    def __init__(self, learning_rates, gamma, tau, sizes, files, intervals, noise):
        self.a_lr = learning_rates["a_lr"]
        self.c_lr = learning_rates["c_lr"]
        self.gamma = gamma
        self.tau = tau
        self.action_max = sizes["action_max"]
        self.action_min = sizes["action_min"]
        self.memory_size = sizes["memory_size"]
        self.input_size = sizes["input_size"]
        self.fcl1_size = sizes["fcl1_size"]
        self.fcl2_size = sizes["fcl2_size"]
        self.actions_num = sizes["actions_num"]
        self.batch_size = sizes["batch_size"]
        self.memory = Memory(self.memory_size, self.input_size, self.actions_num)
        self.oa_mf = files["oa_mf"]
        self.ta_mf = files["ta_mf"]
        self.oc_mf1 = files["oc_mf1"]
        self.oc_mf2 = files["oc_mf2"]
        self.tc_mf1 = files["tc_mf1"]
        self.tc_mf2 = files["tc_mf2"]
        self.learning_counter = 0
        self.time = 0
        self.warmup_interval = intervals["warmup_interval"]
        self.update_interval = intervals["update_interval"]
        self.noise = noise
        
        self.online_ADN = ADN(self.a_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.actions_num, self.oa_mf) 
        self.online_CDN1 = CDN(self.c_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.actions_num, self.oc_mf1)
        self.online_CDN2 = CDN(self.c_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.actions_num, self.oc_mf2)
        self.target_ADN = ADN(self.a_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.actions_num, self.ta_mf) 
        self.target_CDN1 = CDN(self.c_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.actions_num, self.tc_mf1) 
        self.target_CDN2 = CDN(self.c_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.actions_num, self.tc_mf2) 
        
        self.update_targets(tau = 1)
        
    def act(self, state):
        self.target_ADN.eval() #1
        
        if(self.time < self.warmup_interval):
            mu_ = T.tensor(np.random.normal(scale = self.noise, size = (self.actions_num, )))
        else:
            state = T.tensor(state, dtype = T.float, device = self.target_ADN.device) #2
            mu_ = self.target_ADN.forward(state).to(self.target_ADN.device) #3
        mu = mu_ + T.tensor(np.random.normal(scale = self.noise), dtype=T.float).to(mu_.device) #4
        mu = T.clamp(mu, self.action_max[0], self.action_min[0]) #5
        
        self.time += 1
        self.target_ADN.train() #6
        
        return mu.cpu().detach().numpy() #7
    
    def save_models(self):
        self.online_ADN.save_model()
        self.online_CDN1.save_model()
        self.online_CDN2.save_model()
        self.target_ADN.save_model()
        self.target_CDN1.save_model()
        self.target_CDN2.save_model()
        
    def load_models(self):
        self.online_ADN.save_model()
        self.online_CDN1.save_model()
        self.online_CDN2.save_model()
        self.target_ADN.save_model()
        self.target_CDN1.save_model()
        self.target_CDN2.save_model()
        
    def learn(self): #8
        if self.memory.index < self.batch_size:
            return
        
        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
        
        states = T.tensor(states, dtype=T.float).to(self.online_ADN.device)
        actions = T.tensor(actions, dtype=T.float).to(self.online_ADN.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.online_ADN.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.online_ADN.device)
        dones = T.tensor(dones).to(self.online_ADN.device)
        
        target_mu_ = self.target_ADN.forward(states_)
        target_mu_ = target_mu_ + T.clamp(T.tensor(np.random.normal(scale = 0.2)), min = -0.5, max = 0.5)
        target_mu_ = T.clamp(target_mu_, self.action_min[0], self.action_max[0])
        
        target_Q1_ = self.target_CDN1.forward(states_, target_mu_)
        target_Q2_ = self.target_CDN2.forward(states_, target_mu_)
        online_Q1 = self.online_CDN1.forward(states, actions)
        online_Q2 = self.online_CDN2.forward(states, actions)
        
        target_Q1_[dones] = 0.0
        target_Q2_[dones] = 0.0
        target_Q1_ = target_Q1_.view(-1) #9
        target_Q2_ = target_Q2_.view(-1)
        
        target_Q_ = T.min(target_Q1_, target_Q2_)
    
        target = rewards + self.gamma * target_Q_
        target = target.view(self.batch_size, 1)
        
        self.online_CDN1.optimizer.zero_grad()
        self.online_CDN2.optimizer.zero_grad()
        critic_loss1 = F.mse_loss(target, online_Q1)
        critic_loss2 = F.mse_loss(target, online_Q2)
        critic_loss = critic_loss1 + critic_loss2
        critic_loss.backward()
        self.online_CDN1.optimizer.step()
        self.online_CDN2.optimizer.step()
        
        self.learning_counter += 1
        
        if self.learning_counter % self.update_interval != 0:
            return
        
        self.online_ADN.optimizer.zero_grad()
        actor_loss = self.online_CDN1.forward(states, self.online_ADN.forward(states))
        actor_loss = -T.mean(actor_loss)
        actor_loss.backward()
        self.online_ADN.optimizer.step()
        
        self.update_targets()       
        
        
    def update_targets(self, tau = None):
        if tau is None:
            tau = self.tau
        
        online_ADN_params = self.online_ADN.named_parameters()
        online_CDN1_params = self.online_CDN1.named_parameters()
        online_CDN2_params = self.online_CDN2.named_parameters()
        target_ADN_params = self.target_ADN.named_parameters()
        target_CDN1_params = self.target_CDN1.named_parameters()
        target_CDN2_params = self.target_CDN2.named_parameters()
        
        online_ADN_dict = dict(online_ADN_params)
        online_CDN1_dict = dict(online_CDN1_params)
        online_CDN2_dict = dict(online_CDN2_params)
        target_ADN_dict = dict(target_ADN_params)
        target_CDN1_dict = dict(target_CDN1_params)
        target_CDN2_dict = dict(target_CDN2_params)
        
        for name in online_ADN_dict:
            online_ADN_dict[name] = tau * online_ADN_dict[name].clone() + (1 - tau) * target_ADN_dict[name].clone() #10
        
        for name in online_CDN1_dict:
            online_CDN1_dict[name] = tau * online_CDN1_dict[name].clone() + (1 - tau) * target_CDN1_dict[name].clone()
        
        for name in online_CDN2_dict:
            online_CDN2_dict[name] = tau * online_CDN2_dict[name].clone() + (1 - tau) * target_CDN2_dict[name].clone()
        
        self.target_ADN.load_state_dict(online_ADN_dict)
        self.target_CDN1.load_state_dict(online_CDN1_dict)
        self.target_CDN2.load_state_dict(online_CDN2_dict)

    
    