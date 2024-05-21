import sys
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
print(current_dir, parent_dir)
sys.path.append(parent_dir)
import gym 
import matplotlib.pyplot as plt 
import numpy as np
from PG_Agent import PG_Agent
from utils import save_frames_as_gif, plot_learning_curve


if __name__ == "__main__":
    env_name = "LunarLander-v2"
    env = gym.make(env_name, render_mode="rgb_array")
    num_games = 10
    agent = PG_Agent(0.0005, [8], 4, 0.99) 
    file_name = "REINFORCE_" + env_name + "_" + str(agent.learning_rate) + "_" + str(num_games)
    scores_plot_file = str(current_dir) + "/plots/" + file_name + ".png"
    first_landing_gif_file = str(current_dir) + "/plots/" + file_name + "_first.gif"
    final_landing_gif_file = str(current_dir) + "/plots/" + file_name + "_final.gif"
    
    scores = []
    first_frames = []
    final_frames = []
    
    for t in range(num_games):
        done = False
        score = 0
        state = env.reset()[0]
        while not done:
            action = agent.act(state)
            state_, reward, done, info, _ = env.step(action)
            score += reward 
            if(t == 0):
                first_frames.append(env.render())
            if(t == num_games - 1):
                final_frames.append(env.render())
            agent.store_reward(reward)
            state = state_
        agent.learn()
        scores.append(score)
        
        avg_score_ = np.mean(scores[-100:])
        print("step", t, "- score %.2f" %score, "- avg_score %.2f" %avg_score_)
    
    save_frames_as_gif(first_frames, first_landing_gif_file)
    save_frames_as_gif(final_frames, final_landing_gif_file)
    plot_learning_curve(scores, scores_plot_file)            