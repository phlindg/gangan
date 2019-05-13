
import torch as t
import torch.optim as opt

import torch.distributions as td
from Models import Actor, Critic
from collections import deque
from itertools import islice
import numpy as np
class A2CAgent:
    def __init__(self, env):
        self.env = env
        self.action_size = env.action_space.shape
        self.state_size = env.observation_space.shape[0]
        self.device = "cpu"#t.device("cuda" if t.cuda.is_available() else "cpu")
        
        self.critic = Critic(self.action_size, self.state_size).to(self.device)
        self.actor = Actor(self.action_size, self.state_size).to(self.device)

        self.memory = deque(maxlen=2000)
        self.alpha_w = 0.001
        self.alpha_theta = 0.001
        self.gamma = 0.95

        

    def compute_returns(self,v_prime, rewards, dones):
        R = v_prime
        returns = []
        for step in reversed(range(len(rewards))):
            r = rewards[step] + self.gamma*dones[step]*R[step]
            returns.insert(0, r)
        return t.stack(returns)
    def remember(self, state, action, reward, next_state, done, log_prob):
        self.memory.append([state, action, reward, next_state, done, log_prob])
    def reset(self):
        self.loss = 0

    def replay(self, batch_size):
        idx = np.random.randint(low = 0, high = len(self.memory) - batch_size)
        minibatch = list(islice(self.memory, idx, idx+batch_size))

        optim_a = opt.Adam(self.actor.parameters(), lr=self.alpha_theta)
        optim_c = opt.Adam(self.critic.parameters(), lr=self.alpha_w)
        
        states = t.stack([tup[0] for tup in minibatch])
        actions = t.stack([tup[1] for tup in minibatch])
        rewards = t.stack([tup[2] for tup in minibatch])
        next_states = t.stack([tup[3] for tup in minibatch])
        dones = t.stack([tup[4] for tup in minibatch])
        log_probs = t.stack([tup[5] for tup in minibatch])


        v = self.critic(states)

        v_prime = self.critic(next_states)

        returns = self.compute_returns(v_prime, rewards, dones)

        advantage = returns - v
        
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss_func = t.nn.MSELoss()
        critic_loss = critic_loss_func(v, returns)

        self.loss+=critic_loss.data
       

        optim_a.zero_grad()
        optim_c.zero_grad()
        actor_loss.backward(retain_graph=True)
        critic_loss.backward()
        optim_a.step()
        optim_c.step()


    def train(self, episodes, batch_size, render=True):
        
        for episode in range(episodes):
            state = self.env.reset()
            self.reset()
            
            entropy = 0
            tot_reward = 0
            loss = 0
            done = False
            while not done:
                if render:
                    self.env.render()
                
                state = t.tensor(state, dtype=t.float, device = self.device)#.view(self.state_size[0],1)
                #print(state)
                distr = self.actor(state)
                v = self.critic(state)
                action = distr.sample()
               # print(action)
                next_state, reward, done, _ = self.env.step(action.cpu().numpy())
                tot_reward+=reward
                #entropy += distr.entropy().mean()
                
                #rewards.append(t.tensor([reward], dtype=t.float, device = self.device))
                
                log_prob = distr.log_prob(action).unsqueeze(0)
                #print(next_state)
                next_state = t.tensor(next_state, dtype=t.float, device = self.device)#.view(self.state_size[0], 1)
                #print(reward)
                reward = t.tensor(reward, dtype=t.float, device = self.device)
                done = t.tensor(done, dtype=t.float, device = self.device)
                self.remember(state, action, reward, next_state, done, log_prob)
                
                state = next_state
                
                if len(self.memory) > batch_size:
                    self.replay(batch_size)
                if done:
                    print("Episode {}/{}, Reward: {}, Loss: {}".format(episode+1, episodes, tot_reward, self.loss))
        #self.env.close()


                


