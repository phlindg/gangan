

from Env import XEnv
from Agents import PPO
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch

tickers = ["SAND", "ERIC"]
start_date = "2010-01-01"
end_date = "2017-01-01"
num_days = 30
window_length = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = XEnv(tickers, start_date, end_date, num_days, window_length)
#env = gym.make("Pendulum-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = PPO(state_size, action_size, action_std = 0.01, device = "cpu")
def main():
    episodes = 50
    losses = []
    rewards = []
    batch_size = 30
    n_update = 2
    for episode in range(1,episodes+1):
        state = env.reset()
        agent.reset()
        done = False
        tot_reward = 0
        while not done:
            #env.render()
            action = agent.act(state)
            #print(action)
            next_state, reward, done, _ = env.step(action)
            #print(reward.shape)
            agent.memory.rewards.append(reward)
            tot_reward += reward
            
                
            agent.replay()
            agent.memory.clear_memory()

            state = next_state
        if done:
            print("Episode {}/{}, Reward: {}, Loss: {}".format(episode, episodes, tot_reward[0], agent.loss))
            print("Final weights: {}".format(action))
            losses.append(agent.loss)
            rewards.append(tot_reward)
    plt.plot(losses)
    plt.show()

main()