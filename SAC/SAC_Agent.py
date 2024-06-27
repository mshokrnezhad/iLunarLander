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