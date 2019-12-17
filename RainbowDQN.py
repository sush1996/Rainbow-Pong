import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import random
from torch.nn import init, Parameter
from torch.autograd import Variable
from nstep_buffer import optimize_model

torch.manual_seed(0)

class FactorizedNoisyNet(nn.Linear):
    def __init__(self, input_len, output_len, sigma_init=0.017, noisy = True):
        super(FactorizedNoisyNet, self).__init__(in_features=input_len, out_features=output_len, bias=True)

        self.sigma_init = 0.5/np.sqrt(input_len)
        self.sigma_w = Parameter(torch.Tensor(output_len, input_len)) # requires_grad = True
        self.sigma_b = Parameter(torch.Tensor(output_len)) # requires_grad = True
        self.register_buffer('epsilon_in', torch.zeros(input_len))# requires_grad = False
        self.register_buffer('epsilon_out', torch.zeros(output_len)) # requires_grad = False
        self.noisy = noisy
        self._reset_parameters()
        self.reset_noise()

    def _reset_parameters(self):
        init.uniform_(self.weight, -np.sqrt(1 / self.in_features), np.sqrt(1 / self.in_features))
        init.uniform_(self.bias, -np.sqrt(1 / self.in_features), np.sqrt(1 / self.in_features))
        init.constant_(self.sigma_w, self.sigma_init)
        init.constant_(self.sigma_b, self.sigma_init)

    def sample_noise(self):
        torch.randn(self.epsilon_out.shape, out=self.epsilon_out)
        torch.randn(self.epsilon_in.shape, out=self.epsilon_in)

    def forward(self, input):
        if self.noisy:
            self.sample_noise()
        else:
            self.remove_noise()

        noise_matrix = self.epsilon_out.ger(self.epsilon_in)

        return F.linear(input, self.weight + self.sigma_w * noise_matrix, self.bias + self.sigma_b * self.epsilon_out.clone())

    def remove_noise(self):
        torch.zeros(self.epsilon_in.shape, out=self.epsilon_in)
        torch.zeros(self.epsilon_out.shape, out=self.epsilon_out)

    def reset_noise(self):
        eps_i = torch.randn(self.in_features)
        eps_j = torch.randn(self.out_features)
        self.epsilon_in = eps_i.sign() * (eps_i.abs()).sqrt()
        self.epsilon_out = eps_j.sign() * (eps_j.abs()).sqrt()

class RainbowDQN(nn.Module):
    def __init__(self, in_channels = 4, num_actions = 4, num_atoms = 51, dueling = False, noisy = False, distributional = False):
        super(RainbowDQN, self).__init__()
        
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.reset_noise()

        self.noisy = noisy
        self.dueling = dueling
        self.distributional = distributional

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        if self.dueling == True:
            if self.distributional == True:
                self.fc1_adv = FactorizedNoisyNet(7*7*64, 512)
                self.fc1_val = FactorizedNoisyNet(7*7*64, 512)
                
                self.fc2_adv = FactorizedNoisyNet(512, self.num_actions*self.num_atoms)
                self.fc2_val = FactorizedNoisyNet(512, self.num_atoms)
                
            elif self.noisy == True:
                self.fc1_adv = FactorizedNoisyNet(7*7*64, 512)
                self.fc1_val = FactorizedNoisyNet(7*7*64, 512)
                
                self.fc2_adv = FactorizedNoisyNet(512, self.num_actions)
                self.fc2_val = FactorizedNoisyNet(512, 1)
                
            else:
                self.fc1_adv = nn.Linear(7*7*64, 128)
                self.fc1_val = nn.Linear(7*7*64, 128)
                
                self.fc2_adv = nn.Linear(128, self.num_actions)
                self.fc2_val = nn.Linear(128, 1)

        
        elif self.distributional == True and self.noisy == False and self.dueling == False:
            self.fc1 = FactorizedNoisyNet(7*7*64, 512)
            self.fc2 = FactorizedNoisyNet(512, self.num_actions*self.num_atoms)
        
        elif self.noisy == True and self.distributional == False and self.dueling == False:
            self.fc1 = FactorizedNoisyNet(7*7*64, 512)
            self.fc2 = FactorizedNoisyNet(512, self.num_actions)
            
        else:
            self.fc1 = nn.Linear(7*7*64, 512)
            self.fc2 = nn.Linear(512, self.num_actions)
            
        self.relu = nn.ReLU()

    def forward(self, x):
        
        #x = x.float()/255
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        batch_size = x.size(0)

        if self.dueling == True:
           
            adv = self.relu(self.fc1_adv(x))
            val = self.relu(self.fc1_val(x))
            
            if self.distributional == True:
                adv = self.fc2_adv(adv)
                val = self.fc2_val(val)

                adv = adv.view(batch_size, self.num_actions, self.num_atoms)
                val = val.view(batch_size, 1, self.num_atoms)

                x = val + adv - adv.mean(1, keepdim = True)

            else:
                adv = self.fc2_adv(adv)
                val = self.fc2_val(val).expand(x.size(0), self.num_actions)

                x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        else:
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
        
        if self.distributional == True:
            x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)
        
        return x
    
    def reset_noise(self):
        for name, module in self.named_children():
          if 'fc' in name:
            module.reset_noise()