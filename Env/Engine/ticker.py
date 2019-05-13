import pandas as pd
import gym
import numpy as np
import matplotlib.pyplot as plt

class Ticker:
    def __init__(self, ticker, start_date, end_date, num_days_iter, window_length):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.num_days_iter = num_days_iter
        self.window_length = window_length

        self.df, self.dates = self._load_df()

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape = (1,), dtype = np.float32)
        self.today = 0

        self.current_position = 0.0
        self.accumulated_pnl = 0.0

    def _load_df(self):
        df = pd.read_csv("D:/fonden/Data/{}.csv".format(self.ticker), header =0, sep=";",
                index_col = 0, parse_dates = True, usecols = ["Date","Closing price"]
               ).iloc[::-1]
        df = df[(df.index >= self.start_date)]
        df.columns = ["close"]
        
        df["returns"] = df.close.pct_change().fillna(0.00001)
        df["pnl"] = np.zeros(len(df.index))
        df["position"] = np.zeros(len(df.index))
        return df, df.index
    
    def get_state(self):
        pos = self.df.position.iat[self.today]
        price = self.df.close.iat[self.today]

        return price
    def step(self, action):
       # print(self.ticker, action)
        assert self.action_space.contains(action)
        self.df.pnl.iat[self.today] = reward = self.current_position * self.df.returns.iat[self.today]
        
        self.accumulated_pnl += reward
        self.df.position.iat[self.today] = action
        self.current_position = action
        self.today += 1

        done = self.today > self.num_days_iter
        return reward, done
    def reset(self):
        self.today = 0
        self.df.position = self.df.pnl = 0.0
        self.current_position = np.array([0.0])
        self.accumulated_pnl = 0.0
    def render(self, axis):
        axis.plot(np.arange(self.today), self.df.close[:self.today])
        plt.pause(0.0001)
    