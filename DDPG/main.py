import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
#current_dir = "/users/mshokrne/iLunarLander/DDPG"
#parent_dir = "/users/mshokrne/iLunarLander"
sys.path.append(parent_dir)
import gym 
import matplotlib.pyplot as plt 
import numpy as np
from DDPG_Agent import DDPG_Agent
from utils import save_frames_as_gif, plot_learning_curve

if __name__ == "__main__":
    env_name = "LunarLanderContinuous-v2"
    env = gym.make(env_name, render_mode="rgb_array")
    num_games = 1000
    a_lr = 0.0001
    c_lr = 0.001
    gamma=0.99
    tau = 0.001
    input_size = env.observation_space.shape
    fcl1_size = 400
    fcl2_size = 300
    actions_num = env.action_space.shape[0]
    memory_size = 1000000
    batch_size = 64
    file_name = "DDPG_" + env_name + "_" + str(a_lr) + "_" + str(c_lr) + "_" + str(num_games)
    scores_plot_file = str(current_dir) + "/plots/" + file_name + ".png"
    final_landing_file = str(current_dir) + "/plots/" + file_name + ".gif"
    actor_file_name = "Actor_DDPG_" + env_name + "_" + str(a_lr) + "_" + str(num_games)
    critic_file_name = "Critic_DDPG_" + env_name + "_" + str(c_lr) + "_" + str(num_games)
    oa_mf = str(current_dir) + "/models/Online_" + actor_file_name
    oc_mf = str(current_dir) + "/models/Online_" + critic_file_name
    ta_mf = str(current_dir) + "/models/Target_" + actor_file_name
    tc_mf = str(current_dir) + "/models/Target_" + critic_file_name
    agent = DDPG_Agent(a_lr, c_lr, gamma, tau, input_size, fcl1_size, fcl2_size, actions_num, memory_size, batch_size, oa_mf, oc_mf, ta_mf, tc_mf) 
    mode = "train" # select among {"train", "test"}
    
    if(mode == "train"): 
        scores = []
        best_avg_score = -np.inf
        
        for t in range(num_games):
            state, _ = env.reset()
            done, trunc = False, False
            score = 0
            agent.noise.reset()
            while not (done or trunc):
                action = agent.act(state)
                state_, reward, done, trunc, info = env.step(action)
                terminal = done or trunc
                agent.memory.store(state, action, reward, state_, terminal)
                agent.learn()
                score += reward 
                state = state_
            scores.append(score)
            
            avg_score = np.mean(scores[-100:])
            print("game", t, "- score %.2f" %score, "- avg_score %.2f" %avg_score)
            if avg_score > best_avg_score:
                agent.save_models()
                best_avg_score = avg_score

        plot_learning_curve(scores, scores_plot_file)
    
    if(mode == "test"):
        agent.load_models()
        
        frames = []
        done = False
        score = 0
        state = env.reset()[0]
        step = 0
        agent.noise.reset()
        while not done:
            step += 1
            action = agent.act(state)
            state_, reward, done, info, _ = env.step(action)
            score += reward 
            state = state_
            frames.append(env.render())
            if(step >= env_max_num_steps):
                done = True

        print("score %.2f" %score)      
        save_frames_as_gif(frames, final_landing_file)            
