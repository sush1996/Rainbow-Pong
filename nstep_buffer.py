import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np
import time
from torch.autograd import Variable
import gym
from operator import itemgetter
from atari_wrappers import *
#from models import *
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from pdb import set_trace as debug
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state','done' ))
def nstep_target(idx, policy_net,target_net, memory, steps = 20, device = 'cpu', BATCH_SIZE = 32, GAMMA = 0.99, double_dqn = False):
    range_ = np.arange(0, steps + 1)

    idx_nReward = idx.reshape(-1, 1) + range_


    _batch, _ = memory.sample(idx = idx_nReward.ravel())
    n_batch = Transition(*zip(*_batch))
    non_final_mask_rewards = torch.tensor(
        tuple(map(lambda s: s is not None, n_batch.next_state)),
        device=device, dtype=torch.bool).view(idx_nReward.shape)

    non_final_mask = torch.prod(non_final_mask_rewards[:, :-1], 1).bool()

    non_final_mask_r = non_final_mask_rewards[:, :-1]

    #####
    r23 = non_final_mask_r[:, :-1]
    r23 = r23.t().view(r23.shape[1], r23.shape[0], 1).expand(r23.shape[1], r23.shape[0], r23.shape[1])
    r12 = non_final_mask_r[:, 1:]
    r = torch.prod(r23, 0)*r12.long()
    r_mask = torch.cat([non_final_mask_rewards[:, 0].view(-1, 1).long(), r], 1)
    #####

    rewards = tuple((map(lambda r: torch.tensor([r], device=device), n_batch.reward)))
    n_rewards = torch.cat(rewards).view(idx_nReward.shape)[:, 1:] * r_mask.float()


    gamma_n = np.geomspace(1, GAMMA**(steps - 1), steps)

    discounted_rewards = n_rewards*torch.from_numpy(gamma_n).float().to(device)
    discounted_rewards = torch.sum(discounted_rewards, axis = 1).to(device)

    batch_future, _ = memory.sample(idx + steps - 1)
    batch_ = Transition(*zip(*batch_future))

    # non_final_next_states = torch.cat([s for s in batch_.next_state if s is not None]).to(device)


    next_states_ = [s for s in batch_.next_state]
    non_final_next_states_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_.next_state)),device=device, dtype=torch.bool)

    non_final_mask = non_final_next_states_mask * non_final_mask



    non_final_next_states = torch.cat(itemgetter(*list(torch.where(non_final_mask == 1)[0]))(next_states_)).to(device)


    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    if double_dqn:
        max_action = policy_net(non_final_next_states).max(1, keepdim=True)[1].detach()
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, max_action).detach().squeeze(1)

    else:
        next_state_values[non_final_mask] = target_net(non_final_next_states, double_dqn = double_dqn).max(1)[0].detach()
    # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * (GAMMA**steps)) + discounted_rewards


    return expected_state_action_values

def optimize_model(optimizer,policy_net, target_net, memory, device, GAMMA = 0.99, BATCH_SIZE = 32, n_steps = 20, double_dqn = False):
    torch.autograd.set_detect_anomaly(True)
    if len(memory) < BATCH_SIZE:
        return
    transitions, idx,_ = memory.sample()
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))

    actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to(device)


    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    if n_steps == 1:

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        if double_dqn:

            max_action = policy_net(non_final_next_states).max(1, keepdim = True)[1].detach()
            next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, max_action).squeeze(1).detach()

        else:
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

        # next_state_values.requires_grad = False

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    else:
        expected_state_action_values = nstep_target(idx=idx, policy_net=policy_net,target_net=target_net, steps=n_steps, memory=memory, device=device, double_dqn=double_dqn)

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return policy_net
