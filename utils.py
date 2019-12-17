import torch
import numpy as np
import random
import math

class ExponentialSchedule:
    def __init__(self, decay, final_val = 0.02, initial_val=1.0):
        self.decay = decay
        self.final_val = final_val
        self.initial_val = initial_val

    def schedule_value(self, t):
        fraction = self.final_val + (self.initial_val - self.final_val)* \
                   math.exp(-1. * t / self.decay)
        return fraction

def select_action(state, exploration,steps_done, policy_net, device, noise = False, factorized_noise = False):
    # global steps_done
    sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END)* \
    #                 math.exp(-1. * steps_done / EPS_DECAY)
    eps_threshold = exploration(steps_done)
    # steps_done += 1
    if noise or factorized_noise:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1, 1)
    else:
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state.to(device)).max(1)[1].view(1,1)
        else:
            return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)


def get_state(obs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    state = state.unsqueeze(0)

    return state.float().to(device)