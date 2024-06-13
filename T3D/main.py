#1:     Note that you need to replace online_ADN with target_ADN in agent.act().

import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
#current_dir = "/users/mshokrne/iLunarLander/T3D"
#parent_dir = "/users/mshokrne/iLunarLander"
sys.path.append(parent_dir)
import gym 
import matplotlib.pyplot as plt 
import numpy as np
from T3D_Agent import T3D_Agent
from utils import save_frames_as_gif, plot_learning_curve

if __name__ == "__main__":
    env_name = "LunarLanderContinuous-v2"
    env = gym.make(env_name, render_mode="rgb_array")
    num_games = 1000
    a_lr = 0.001
    c_lr = 0.001
    learning_rates = {a_lr, c_lr}
    gamma=0.99
    tau = 0.005
    action_max = env.action_space.high
    action_min = env.action_space.low
    input_size = env.observation_space.shape
    fcl1_size = 400 
    fcl2_size = 300
    actions_num = env.action_space.shape[0]
    memory_size = 1000000
    batch_size = 100
    sizes = {action_max, action_min, memory_size, input_size, fcl1_size, fcl2_size, actions_num, batch_size}
    update_interval = 2
    warmup_interval = 1000
    intervals = {warmup_interval, update_interval}
    file_name = "DDPG_" + env_name + "_" + str(a_lr) + "_" + str(c_lr) + "_" + str(num_games)
    scores_plot_file = str(current_dir) + "/plots/" + file_name + ".png"
    final_landing_file = str(current_dir) + "/plots/" + file_name + ".gif"
    actor_file_name = "Actor_DDPG_" + env_name + "_" + str(a_lr) + "_" + str(num_games)
    critic_file_name = "Critic_DDPG_" + env_name + "_" + str(c_lr) + "_" + str(num_games)
    oa_mf = str(current_dir) + "/models/Online_" + actor_file_name
    oc_mf1 = str(current_dir) + "/models/Online_1" + critic_file_name
    oc_mf2 = str(current_dir) + "/models/Online_2" + critic_file_name
    ta_mf = str(current_dir) + "/models/Target_" + actor_file_name
    tc_mf1 = str(current_dir) + "/models/Target_1" + critic_file_name
    tc_mf2 = str(current_dir) + "/models/Target_2" + critic_file_name
    files = {oa_mf, oc_mf1, oc_mf2, ta_mf, tc_mf1, tc_mf2}
    noise = 0.1
    agent = T3D_Agent(learning_rates, gamma, tau, sizes, files, intervals, noise)
    mode = "train" # select among {"train", "test"}
    
    if(mode == "train"): 
        scores = []
        best_avg_score = -np.inf
        
        for t in range(num_games):
            state, _ = env.reset()
            done, trunc = False, False
            score = 0
            step = 0
            while not (done or trunc):
                step += 1
                action = agent.act(state)
                state_, reward, done, trunc, info = env.step(action)
                terminal = done or trunc
                agent.memory.store(state, action, reward, state_, terminal)
                agent.learn()
                score += reward 
                state = state_
            scores.append(score)
            
            avg_score = np.mean(scores[-100:])
            print("game", t, "steps", step, "- score %.2f" %score, "- avg_score %.2f" %avg_score)
            if avg_score > best_avg_score:
                agent.save_models()
                best_avg_score = avg_score

        plot_learning_curve(scores, scores_plot_file)
    
    if(mode == "test"):
        agent.load_models()
        
        frames = []
        done, trunc = False, False
        score = 0
        state, _ = env.reset()
        step = 0
        agent.noise.reset()
        while not (done or trunc):
            step += 1
            action = agent.act(state) #1
            state_, reward, done, trunc, info = env.step(action)
            terminal = done or trunc
            score += reward 
            state = state_
            frames.append(env.render())

        print("score %.2f" %score)      
        save_frames_as_gif(frames, final_landing_file)            
