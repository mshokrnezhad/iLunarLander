#       In this file, we implement an SAC-based agent which is working based on ADN, CDN, and VDN.
#1:     ADN.eval() (activating the evaluation mode of the system) might not have a significant impact on the results.
#       However, to ensure consistency, reproducibility, and to follow best practices, it’s important to set the network to evaluation mode during inference. 
#       This practice is especially crucial if the network might include other types of layers in the future.
#2:     To send the state in the form of Tensor to the device selected in ADN.
#3:     .clone() method is used to create a copy of a tensor. 
#       This ensures that any subsequent operations on the cloned tensor do not affect the original tensor.

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
        
        self.ADN = ADN(self.a_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.actions_num, self.a_mf) 
        self.CDN1 = CDN(self.c_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.actions_num, self.c_mf1)
        self.CDN2 = CDN(self.c_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.actions_num, self.c_mf2)
        self.online_VDN = VDN(self.c_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.ov_mf) 
        self.target_VDN = VDN(self.c_lr, self.input_size, self.fcl1_size, self.fcl2_size, self.tv_mf) 
        
        self.update_targets(tau = 1)

    def act(self, state, mode = "train"):
        self.ADN.eval() #1
        
        state = T.tensor([state], dtype = T.float, device = self.ADN.device) #2
        actions = self.ADN.sample_action(state, isReparamEnabled = False)
        
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
            online_VDN_dict[name] = tau * online_VDN_dict[name].clone() + (1 - tau) * online_VDN_dict[name].clone() #3
        
        self.target_VDN.load_state_dict(online_VDN_dict)

    
        
        
        
        