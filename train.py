import gym
from RainbowAgent import RainbowAgent
from atari_wrappers import *
import torch.nn as nn
import torch
from RainbowDQN import RainbowDQN
import numpy as np
from utils import *
from nstep_buffer import optimize_model
from buffers import *
'''
def get_state(obs):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    state = state.unsqueeze(0)
    
    return state.float().to(device)
'''

def mini_batch_train(env, agent, target_steps, initial_memory, memory, n_steps, max_episodes, batch_size, exploration):
    episode_rewards = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        step = 0
        
        episode_rewards_plot = []
        num_steps_plot = []

        done = False

        while not done:
            state_temp = get_state(state)
            #action = agent.get_action(state_temp)
            action = select_action(state_temp, exploration, step, agent.model, device, noise = False, factorized_noise = agent.noisy)
            
            next_state, reward, done, _ = env.step(action)
            #print("episode:", episode, "step", step, "action:", action, "reward:", reward)
            next_state_temp = get_state(next_state)
            
            agent.replay_buffer.push(state_temp, action, reward, next_state_temp, done)
            episode_reward += reward

            if agent.multi_step == True or agent.distributional == True:
                if step > initial_memory:
                   agent.model = optimize_model(agent.optimizer, agent.model, agent.target_model, memory, device, 
                   								GAMMA = 0.99, BATCH_SIZE = 32, n_steps = 20, double_dqn = agent.double)
                
                if step % target_steps == 0:
                   agent.target_model.load_state_dict(agent.model.state_dict())
                   agent.update_target()

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done:# or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break
			
            state = next_state
            step = step + 1

        episode_rewards_plot.append(episode_reward)
        num_steps_plot.append(step)

    return episode_rewards_plot, num_steps_plot

env_id = "PongNoFrameskip-v4"

num_episodes = 1000
batch_size = 32
target_steps = 1000
initial_memory = 10000
memory_size = 10*initial_memory
n_step = 3
lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
env = gym.make(env_id)
env = make_env(env)
agent = RainbowAgent(env, vmin = 10, vmax=-10, num_atoms = 51, lr=lr, gamma=0.99, tau = 0.01, buffer_size=10000,
					prioritized = False, dueling = False, noisy = True, double = True, multi_step = False, distributional = False)

EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 1000000

if agent.prioritized==True:
	memory  = PrioritizedBuffer(max_size = memory_size)
else:
	memory  = BasicBuffer(max_size = memory_size)


Schedule = ExponentialSchedule(EPS_DECAY)
exploration = Schedule.schedule_value
episode_rewards_plot, num_steps_plot = mini_batch_train(env, agent, target_steps, initial_memory, memory, n_step, num_episodes, batch_size, exploration) 