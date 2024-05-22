import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
#current_dir = "/users/mshokrne/iLunarLander/REINFORCE"
#parent_dir = "/users/mshokrne/iLunarLander"
sys.path.append(parent_dir)
import gym 
import matplotlib.pyplot as plt 
import numpy as np
from PG_Agent import PG_Agent
from utils import save_frames_as_gif, plot_learning_curve

if __name__ == "__main__":
    env_name = "LunarLander-v2"
    env = gym.make(env_name, render_mode="rgb_array")
    env_max_num_steps = 1000
    num_games = 3000
    learning_rate = 0.0005
    file_name = "REINFORCE_" + env_name + "_" + str(learning_rate) + "_" + str(num_games)
    scores_plot_file = str(current_dir) + "/plots/" + file_name + ".png"
    final_landing_file = str(current_dir) + "/plots/" + file_name + ".gif"
    model_file = str(current_dir) + "/models/" + file_name
    agent = PG_Agent(learning_rate, 0.99, [8], 4, model_file) 
    mode = "train" # select among {"train", "test"}
    
    if(mode == "train"): 
        scores = []
        best_avg_score = -np.inf
        
        for t in range(num_games):
            done = False
            score = 0
            state = env.reset()[0]
            step = 0
            while not done:
                step += 1
                action = agent.act(state)
                state_, reward, done, info, _ = env.step(action)
                score += reward 
                agent.store_reward(reward)
                state = state_
                if(step >= env_max_num_steps):
                    done = True
            agent.learn()
            scores.append(score)
            
            avg_score = np.mean(scores[-100:])
            print("step", t, "- score %.2f" %score, "- avg_score %.2f" %avg_score)
            if avg_score > best_avg_score:
                agent.PGN.save_model()
                best_avg_score = avg_score
        
        plot_learning_curve(scores, scores_plot_file)       
    
    if(mode == "test"):
        agent.PGN.load_model()
        
        frames = []
        done = False
        score = 0
        state = env.reset()[0]
        step = 0
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
