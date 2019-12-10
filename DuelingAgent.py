import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

from buffers import BasicBuffer, PrioritizedBuffer
from DuelingDQN import DuelingDQN


class DuelingAgent:

    def __init__(self, env, learning_rate, gamma, buffer_size, prioritized):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.prioritized = prioritized

        if self.prioritized == False:
            self.replay_buffer = BasicBuffer(max_size=buffer_size)
        else:
            self.replay_buffer = PrioritizedBuffer(max_size=buffer_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DuelingDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, eps=0.20):
        #state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        
        if(np.random.randn() > eps):
            return self.env.action_space.sample()
        return action


    def tensor_states(self, states_list):
        states_tensor = torch.Tensor()
        
        for state in states_list:
            state_float = torch.FloatTensor(state).to(self.device)
            states_tensor = torch.cat((states_tensor, state_float))

        return states_tensor


    def compute_loss(self, batch):
        
        if self.prioritized == False:
            states, actions, rewards, next_states, dones = batch
        else:
            states, actions, rewards, next_states, dones = batch[0]

        states = self.tensor_states(states)
        next_states = self.tensor_states(next_states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        dones = torch.FloatTensor(dones).to(self.device)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()