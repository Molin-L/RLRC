import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import random
from duelingDQN.models import ConvDuelingDQN, DuelingDQN

random.seed(42)
class Environment:
    class Action_space:
        def __init__(self, eps):
            self.eps = eps
            self.n = 2
            self.actions = [0, 1]
        def sample(self):
            return random.randrange(2)
    def __init__(self, max_len, obs_space, eps=0.5):
        self.max_len = max_len
        self.action_space = Action_space(eps)
        self.observation_space = obs_space

class DuelingAgent:

    def __init__(self, env, learning_rate=3e-4, gamma=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ConvDuelingDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        
      
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, eps=0.5):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        
        if(np.random.randn() > eps):
            return self.env.action_space.sample()
        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        
        return loss

    def update(self, batch):
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()