#       In this file, we implement an SAC-based agent which is working based on ADN, CDN, and VDN.
#1:     ADN.eval() (activating the evaluation mode of the system) might not have a significant impact on the results.
#       However, to ensure consistency, reproducibility, and to follow best practices, it’s important to set the network to evaluation mode during inference. 
#       This practice is especially crucial if the network might include other types of layers in the future.
#2:     To send the state in the form of Tensor to the device selected in ADN.
#3:     .clone() method is used to create a copy of a tensor. 
#       This ensures that any subsequent operations on the cloned tensor do not affect the original tensor.
#4:     Check the source paper and README.md to see the logic of self.learn().
#5:     When online_v is computed, it might have a shape that includes an extra dimension, 
#       making it a two-dimensional tensor with shape (batch_size, 1). 
#       For some operations, it’s necessary to have this tensor as a one-dimensional tensor (a vector) with shape (batch_size,). 
#       This ensures that arithmetic operations like addition with rewards and element-wise multiplication with self.gamma work correctly 
#       and that the resulting tensor can be reshaped back to (batch_size, 1) without issues. For example, 
#       online_v = tensor([[0.5], [0.8], [0.3], [1.2]]) afeter .view(-1) would be tensor([0.5, 0.8, 0.3, 1.2]).

import numpy as np
import torch as T
import torch.nn.functional as F
from ADN import ADN
from CDN import CDN
from VDN import VDN
from Memory import Memory

class SAC_Agent():
    def __init__(self, learning_rates, gamma, tau, sizes, files, reward_scaler):
        self.a_lr = learning_rates["a_lr"]
        self.c_lr = learning_rates["c_lr"]
        self.gamma = gamma
        self.tau = tau
        self.actions_max = sizes["actions_max"]
        self.memory_size = sizes["memory_size"]
        self.input_size = sizes["input_size"]
        self.fcl1_size = sizes["fcl1_size"]
        self.fcl2_size = sizes["fcl2_size"]
        self.actions_num = sizes["actions_num"]
        self.batch_size = sizes["batch_size"]
        self.memory = Memory(self.memory_size, self.input_size, self.actions_num)
        self.a_mf = files["a_mf"]
        self.c_mf1 = files["c_mf1"]
        self.c_mf2 = files["c_mf2"]
        self.ov_mf = files["ov_mf"]
        self.tv_mf = files["tv_mf"]
        self.reward_scaler = reward_scaler
        
        self.ADN = ADN(self.a_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.actions_num, self.actions_max, self.a_mf) 
        self.CDN1 = CDN(self.c_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.actions_num, self.c_mf1)
        self.CDN2 = CDN(self.c_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.actions_num, self.c_mf2)
        self.online_VDN = VDN(self.c_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.ov_mf) 
        self.target_VDN = VDN(self.c_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.tv_mf) 
        
        self.update_targets(tau = 1)

    def act(self, state, mode = "train"):
        self.ADN.eval() #1
        
        state = T.tensor(state[np.newaxis, :], dtype = T.float, device = self.ADN.device) #2
        actions, _ = self.ADN.sample_action(state, isReparamEnabled = False)
        
        self.ADN.train()
        
        return actions.cpu().detach().numpy()[0]
    
    def save_models(self):
        self.ADN.save_model()
        self.CDN1.save_model()
        self.CDN2.save_model()
        self.online_VDN.save_model()
        self.target_VDN.save_model()
        
    def load_models(self):
        self.ADN.load_model()
        self.CDN1.load_model()
        self.CDN2.load_model()
        self.online_VDN.load_model()
        self.target_VDN.load_model()
        
    def update_targets(self, tau = None):
        if tau is None:
            tau = self.tau
        
        online_VDN_params = self.online_VDN.named_parameters()
        target_VDN_params = self.target_VDN.named_parameters()
        
        online_VDN_dict = dict(online_VDN_params)
        target_VDN_dict = dict(target_VDN_params)
        
        for name in online_VDN_dict:
            online_VDN_dict[name] = tau * online_VDN_dict[name].clone() + (1 - tau) * target_VDN_dict[name].clone() #3
        
        self.target_VDN.load_state_dict(online_VDN_dict)
        
    def learn(self): #4
        if self.memory.index < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.CDN1.device)
        actions = T.tensor(actions, dtype=T.float).to(self.CDN1.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.CDN1.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.CDN1.device)
        dones = T.tensor(dones).to(self.CDN1.device)

        online_v = self.online_VDN.forward(states).view(-1) #5
        target_v_ = self.target_VDN.forward(states_).view(-1)
        target_v_ = target_v_.clone()
        target_v_[dones] = 0.0

        new_actions, log_probabilities = self.ADN.sample_action(states, isReparamEnabled=False)
        log_probabilities = log_probabilities.view(-1)
        Q1 = self.CDN1.forward(states, new_actions)
        Q2 = self.CDN2.forward(states, new_actions)
        Q = T.min(Q1, Q2)
        Q = Q.view(-1)
        self.online_VDN.optimizer.zero_grad()
        v_target = Q - log_probabilities
        v_loss = 0.5 * F.mse_loss(online_v, v_target)
        v_loss.backward(retain_graph=True)
        self.online_VDN.optimizer.step()

        new_actions, log_probabilities = self.ADN.sample_action(states, isReparamEnabled=True)
        log_probabilities = log_probabilities.view(-1)
        Q1 = self.CDN1.forward(states, new_actions)
        Q2 = self.CDN2.forward(states, new_actions)
        Q = T.min(Q1, Q2)
        Q = Q.view(-1)
        self.ADN.optimizer.zero_grad()
        actor_loss = log_probabilities - Q
        actor_loss = T.mean(actor_loss)
        actor_loss.backward(retain_graph=True)
        self.ADN.optimizer.step()
        
        self.CDN1.optimizer.zero_grad()
        self.CDN2.optimizer.zero_grad()
        Q_ = self.reward_scaler * rewards + self.gamma * target_v_
        Q_ = Q_.clone().detach()
        Q1 = self.CDN1.forward(states, actions).view(-1)
        Q2 = self.CDN2.forward(states, actions).view(-1)
        critic1_loss = 0.5 * F.mse_loss(Q1, Q_)
        critic2_loss = 0.5 * F.mse_loss(Q2, Q_)
        critic_loss = critic1_loss + critic2_loss
        critic_loss.backward()
        self.CDN1.optimizer.step()
        self.CDN2.optimizer.step()

        self.update_targets()    
            