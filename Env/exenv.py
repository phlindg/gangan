import pandas as pd
import numpy as np
import gym
from Env.Engine import Engine
import matplotlib.pyplot as plt
plt.ion()

#https://github.com/wbaik/gym-stock-exchange/tree/master/gym_exchange
class XEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, tickers, start_date,end_date ,num_days, window_length,
                    render=False, seed = None):
        self.tickers = tickers
        self._seed = seed

        self.env = Engine(tickers, start_date, end_date, num_days, window_length)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape = (len(tickers),), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(tickers),window_length), dtype=np.float32)

        self.total_reward = [0]
        self.fig, self.axis = plt.subplots(1,1, figsize=(5,5))
    def render(self):
        #self.env.render()
        
        self.axis.plot(np.arange(self.env.tickers[0].today+1), self.total_reward)
        plt.pause(0.0001)
    def reset(self):
        self.total_reward = [0]
        self.env.reset()
        return self.env.get_state()
    def step(self, actions):
        actions = np.array(actions).reshape((len(self.tickers),1))
        next_state, reward, done = self.env.step(actions)
        self.total_reward += [reward]
        
        return next_state, reward, done, None

    