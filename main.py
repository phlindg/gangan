

#from Env import XEnv
from Agents import PPO
from Models import Predictor
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as tdata
import torch.optim as opt
import torch.nn as nn


def main():
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

def ls():
    data = np.random.normal(loc=0.0, scale=0.3, size=(10000,1))
    episodes = 2
    batch_size = 10
    dataloader = tdata.DataLoader(data, batch_size=batch_size)
    input_size = data.shape[1]
    output_size = data.shape[1]
    pred = Predictor(input_size, , batch_size, output_size)
ls()