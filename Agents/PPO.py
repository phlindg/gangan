
import torch
import torch.optim as opt
import torch.distributions as td
from Models import ActorCritic
from memory import Memory
import numpy as np


class PPO:
    def __init__(self, state_size, action_size, action_std, device="cpu"):
        self.lr = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.gamma = 0.99
        self.device = device
        
        self.eps_clip = 0.2
        self.epochs = 5
        self.loss = 0

        self.policy = ActorCritic(state_size, action_size, action_std).to(self.device)
        self.optimizer = opt.Adam(self.policy.parameters(), lr = self.lr, betas = (self.beta1, self.beta2))
        self.policy_old = ActorCritic(state_size, action_size, action_std).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.memory = Memory()

    
    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action_mean = self.policy_old.actor(state)
        #print(action_mean)
        #distr = td.MultivariateNormal(action_mean, torch.diag(self.policy_old.action_variance).to(self.device))
        distr = td.Dirichlet(action_mean)
        action = distr.sample()
        action_logprob = distr.log_prob(action)
        
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(action_logprob)

        action =  action.detach()

        return action.cpu().data.numpy().flatten()

    def reset(self):
        self.loss = 0

    def _evaluate(self, state, action):
        action_mean = self.policy.actor(state)
        #distr = td.MultivariateNormal(action_mean, torch.diag(self.policy.action_variance))
        distr = td.Dirichlet(action_mean)
        action_logprobs = distr.log_prob(action)
        dist_entropy = distr.entropy()
        state_value = self.policy.critic(state)

        return action_logprobs, state_value.flatten(), dist_entropy
    def replay(self):
        #Monte carlo estiamte of state reward
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            #print(discounted_reward, reward)
            rewards.insert(0, discounted_reward)
        #rewards = np.array(rewards)
        #normalize rewards
        # for i in rewards:
        #     print(i, end = " ")
        # print("==================================================================================")
        # for i in self.memory.rewards:
        #     print(i, end =" ")
        rewards = torch.FloatTensor(rewards).to(self.device).squeeze(1)
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8) #pga zerodivsion

        old_states = torch.stack(self.memory.states)
        old_actions = torch.stack(self.memory.actions)
        old_logprobs = torch.stack(self.memory.logprobs)

        for _ in range(self.epochs):
            logprobs, state_values, dist_entropy = self._evaluate(old_states, old_actions)

            #Find ratio pi_theta / pi_theta_old
            ratios = torch.exp(logprobs - old_logprobs.detach()) #detach gör så att den ej optimeras

            #Find surrogate loss
            #print(rewards.size(), state_values.size())
            advantage = rewards - state_values.detach()
            surr1 = ratios*advantage
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)*advantage
            loss = -torch.min(surr1, surr2) + 0.5*self.mse_loss(state_values, rewards) - 0.1*dist_entropy
            #print("STATE", state_values)
            #print("REW", rewards)
            #print("LOGS", logprobs)


            self.loss += loss.mean()
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        #copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
