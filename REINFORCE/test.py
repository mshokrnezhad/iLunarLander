import gym
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode="human")

    state = env.reset()
    done = False
    score = 0
    while not done:
        action = env.action_space.sample()
        state_, reward, done, info, _ = env.step(action)
        score += reward
        env.render()
        print(reward, score)
        
    