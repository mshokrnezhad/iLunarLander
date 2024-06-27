import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
#current_dir = "/users/mshokrne/iLunarLander/TD3"
#parent_dir = "/users/mshokrne/iLunarLander"
sys.path.append(parent_dir)
import gym 
import matplotlib.pyplot as plt 
import numpy as np
from SAC_Agent import SAC_Agent
from utils import save_frames_as_gif, plot_learning_curve

if __name__ == "__main__":
    env_name = "LunarLanderContinuous-v2"
    env = gym.make(env_name, render_mode="rgb_array")
    num_games = 1000
    a_lr = 0.0003
    c_lr = 0.0003
    learning_rates = {'a_lr': a_lr, 'c_lr': c_lr}
    gamma=0.99
    tau = 0.005
    actions_max = env.action_space.high
    actions_min = env.action_space.low
    input_size = env.observation_space.shape
    fcl1_size = 256
    fcl2_size = 256
    actions_num = env.action_space.shape[0]
    memory_size = 1000000
    batch_size = 256
    sizes = {
        'actions_max': actions_max, 
        'actions_min': actions_min, 
        'memory_size': memory_size, 
        'input_size': input_size, 
        'fcl1_size': fcl1_size, 
        'fcl2_size': fcl2_size, 
        'actions_num': actions_num, 
        'batch_size': batch_size
    }
    reward_scaler = 2
    file_name = "SAC_" + env_name + "_" + str(a_lr) + "_" + str(c_lr) + "_" + str(num_games)
    scores_plot_file = str(current_dir) + "/plots/" + file_name + ".png"
    final_landing_file = str(current_dir) + "/plots/" + file_name + ".gif"
    actor_file_name = "Actor_SAC_" + env_name + "_" + str(a_lr) + "_" + str(num_games)
    critic_file_name = "Critic_SAC_" + env_name + "_" + str(c_lr) + "_" + str(num_games)
    value_file_name = "Value_SAC_" + env_name + "_" + str(c_lr) + "_" + str(num_games)
    a_mf = str(current_dir) + "/models/" + actor_file_name
    c_mf1 = str(current_dir) + "/models/1_" + critic_file_name
    c_mf2 = str(current_dir) + "/models/2_" + critic_file_name
    ov_mf = str(current_dir) + "/models/Online_" + value_file_name
    tv_mf = str(current_dir) + "/models/Target_" + value_file_name
    files = {'a_mf': a_mf, 'c_mf1': c_mf1, 'c_mf2': c_mf2, 'ov_mf': ov_mf, 'tv_mf': tv_mf}
    agent = SAC_Agent(learning_rates, gamma, tau, sizes, files, reward_scaler)
    mode = "train" # select among {"train", "test"}