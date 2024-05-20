import gym 
import matplotlib.pyplot as plt 
import numpy as np
from PG_Agent import PG_Agent



def plot_learning_curve(scores, file):
    x = [i+1 for i in range(len(scores))]
    avg_scores = np.zeros(len(scores))
    
    for i in range(len(avg_scores)):
        avg_scores[i] = np.mean(scores[max(0, i-100):(i+1)])
        
    plt.plot(x, avg_scores)
    plt.title("average score of previous 100 steps")
    
    plt.savefig(file)
    


if __name__ == "__main__":
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    num_games = 3000
    agent = PG_Agent(0.0005, [8], 4, 0.99) 
    file_name = "REINFORCE_" + env_name + "_" + str(agent.learning_rate) + "_" + str(num_games)
    file = "plots/" + file_name + ".png"
    
    scores = []
    for t in range(num_games):
        done = False
        score = 0
        state = env.reset()[0]
        while not done:
            action = agent.act(state)
            state_, reward, done, info, _ = env.step(action)
            score += reward 
            agent.store_reward(reward)
            state = state_
        agent.learn()
        scores.append(score)
        
        avg_score_ = np.mean(scores[-100:])
        print("step", t, "- score %.2f" %score, "- avg_score %.2f" %avg_score_)
    
    plot_learning_curve(scores, file)            