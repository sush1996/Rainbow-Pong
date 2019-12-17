import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

from buffers import BasicBuffer, PrioritizedBuffer
from RainbowDQN import RainbowDQN

torch.manual_seed(0)

class RainbowAgent:

    def __init__(self, env, vmin, vmax, num_atoms, lr, gamma, tau, buffer_size, prioritized, dueling, noisy, double, multi_step, distributional):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.vmin = vmin
        self.vmax = vmax
        self.num_atoms = num_atoms

        self.multi_step = multi_step
        self.prioritized = prioritized
        self.double = double
        self.dueling = dueling
        self.noisy = noisy
        self.distributional = distributional
        
        #Prioritized Replay
        if self.prioritized == True:
            self.replay_buffer = PrioritizedBuffer(max_size=buffer_size)
        else:
            self.replay_buffer = BasicBuffer(max_size=buffer_size)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RainbowDQN(env.observation_space.shape, env.action_space.n, self.num_atoms, self.dueling, self.noisy, self.distributional).to(self.device)
        self.target_model = RainbowDQN(env.observation_space.shape, env.action_space.n, self.num_atoms, self.dueling, self.noisy, self.distributional).to(self.device)
        
        #Double DQN
        if double == True:
            for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
                target_param.data.copy_(param)
        else:
            self.target_model = None
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.MSE_loss = nn.MSELoss()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())    
        
    def get_action(self, state, eps=0.20):
    
        if(np.random.randn() > eps):
            return self.env.action_space.sample()
        
        else:    
            if self.distributional == False:
                qvals = self.model.forward(state)
                action = np.argmax(qvals.cpu().detach().numpy())
                return action
            else:
                #state = autograd.Variable(torch.FloatTensor(state).unsqueeze(0), volatile = True)
                dist = self.model.forward(state)#.data().cpu()
                dist = dist*torch.linspace(self.vmin, self.vmax, self.num_atoms)
                action = dist.sum(2).max(1)[1].numpy()[0]
                return action
        
    def tensor_states(self, states_list):
        states_tensor = torch.Tensor()
        
        for state in states_list:
            state_float = torch.cuda.FloatTensor(state).to(self.device)
            states_tensor = torch.cat((states_tensor, state_float))

        return states_tensor

    def projection_distribution(self, next_state, rewards, dones):
        batch_size  = next_state.size(0)
        
        delta_z = float(self.vmax - self.vmin) / (self.num_atoms - 1)
        support = torch.linspace(self.vmin, self.vmax, self.num_atoms)
        
        next_dist   = self.target_model(next_state).data.cpu() * support
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
        next_dist   = next_dist.gather(1, next_action).squeeze(1)
        
        rewards = rewards.unsqueeze(1)#.expand_as(next_dist)
        dones   = dones.unsqueeze(1)#.expand_as(next_dist)
        support = support.unsqueeze(0)#.expand_as(next_dist)
        
        Tz = rewards + (1 - dones) * 0.99 * support
        Tz = Tz.clamp(min=self.vmin, max=self.vmax)
        b  = (Tz - self.vmin) / delta_z
        l  = b.floor().long()
        u  = b.ceil().long()
            
        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long()\
                        .unsqueeze(1).expand(batch_size, self.num_atoms)

        proj_dist = torch.zeros(next_dist.size())    
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
    
        return proj_dist

    def compute_loss(self, batch):
        
        if self.prioritized == False:
            states, actions, rewards, next_states, dones = batch[0]
        else:
            states, actions, rewards, next_states, dones = batch[0]
            idxs = batch[1]
            IS_weights = batch[2]
        
        states = self.tensor_states(states)
        next_states = self.tensor_states(next_states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.cuda.FloatTensor(rewards).to(self.device)
        dones = torch.cuda.FloatTensor(dones).to(self.device)
        
            
        if self.distributional == True:
            proj_dist = self.projection_distribution(next_states, rewards, dones)
        
            dist = self.model.forward(states)
            action = actions
            
            dist = action
            dist.data.clamp_(0.01, 0.99)
            dist = dist.type(torch.cuda.FloatTensor)
            
            loss = - ((proj_dist) * dist.log().view(-1,1)).sum(1).mean()
            loss = autograd.Variable(loss, requires_grad = True)    
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.model.reset_noise()
            self.target_model.reset_noise()
        
        else:
            curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
            curr_Q = curr_Q.squeeze(1)
            next_Q = self.model.forward(next_states)
            max_next_Q = torch.max(next_Q, 1)[0]
            expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q
        
            loss = self.MSE_loss(curr_Q, expected_Q)
            
        if self.prioritized == True:
            IS_weights = torch.cuda.FloatTensor(IS_weights).to(self.device)
            td_errors = loss * IS_weights
            return td_errors, idxs
        else:
            return loss

    def update(self, batch_size):
        
        if self.prioritized == True:
            batch_temp, idxs, IS_weights = self.replay_buffer.sample(batch_size)
            batch = (batch_temp, idxs, IS_weights)
        else:
            batch = self.replay_buffer.sample(batch_size)

        if self.prioritized == True:
            td_loss, idxs = self.compute_loss(batch)
            loss = td_loss.mean()
        else:
            loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.prioritized == True:
            for idx, td_error in zip(idxs, td_loss.cpu().detach().numpy()):
                self.replay_buffer.update_priority(idx, td_error)

        if self.double == True:
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
