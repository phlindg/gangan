
import numpy as np
from Env.Engine.ticker import Ticker
import matplotlib.pyplot as plt
import itertools
import functools




class Engine:
    def __init__(self, tickers, start_date, end_date, num_days_iter, window_length):

        self.tickers = self._get_tickers(tickers, start_date, end_date, num_days_iter, window_length)

        ren = True
       # if ren:
       #     fig_height = 3 * len(tickers)
       #     self.fig, self.ax_list = plt.subplots(len(tickers), 1, figsize=(10, fig_height))

    def _get_tickers(self, tickers, start_date, end_date, num_days_iter, window_length):
        ticker_list = [Ticker(ticker, start_date, end_date, num_days_iter, window_length) for ticker in tickers]
        return ticker_list
    def reset(self):
        list(map(lambda ticker: ticker.reset(), self.tickers))
    def get_state(self):
        return np.array(list(map(lambda ticker: ticker.get_state(), self.tickers)))
    def render(self):
        for axis, ticker in zip(self.ax_list, self.tickers):
            ticker.render(axis)
    def step(self, actions):
        assert len(actions) == len(self.tickers)

        rewards, dones = zip(*itertools.starmap(lambda ticker, action: ticker.step(action), zip(self.tickers, actions)))
        # print("ZIP ",rewards)
        # print(np.array(rewards).sum(axis=0))
        reward = functools.reduce(lambda x, y: x + y, rewards)
        done = functools.reduce(lambda x, y: x | y, dones, False)

        return self.get_state(), reward, done

