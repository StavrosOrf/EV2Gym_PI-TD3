import copy
import numpy as np
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces import MultiDiscrete, Discrete

torch.autograd.set_detect_anomaly(True)


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class Trajectory_ReplayBuffer(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_episode_length,
                 device=None,
                 max_size=int(1e4)):

        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.max_length = max_episode_length

        self.state = torch.zeros((max_size, max_episode_length, state_dim))
        self.action = torch.zeros((max_size, max_episode_length, action_dim))
        self.rewards = torch.zeros((max_size, max_episode_length))
        self.dones = torch.zeros((max_size, max_episode_length))

        self.device = device

    def add(self, state, action, reward, done):
        self.state[self.ptr, :, :] = state
        self.action[self.ptr, :, :] = action
        self.rewards[self.ptr, :] = reward.squeeze()
        self.dones[self.ptr, :] = done.squeeze()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # Example of the sample method in utils.py
    def sample(self, batch_size, sequence_length):
        ind = np.random.randint(0, self.size, size=batch_size)
        start = np.random.randint(
            0, self.max_length - sequence_length, size=batch_size)
        end = start + sequence_length

        # Ensure ind, start, and end are integers
        ind = ind.astype(int)
        start = start.astype(int)
        end = end.astype(int)

        # Sample states and actions
        states = torch.FloatTensor(self.state[ind, :, :]).to(self.device)
        actions = torch.FloatTensor(self.action[ind, :, :]).to(self.device)
        next_states = torch.FloatTensor(self.state[ind, :, :]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[ind, :]).to(self.device)
        dones = torch.FloatTensor(self.dones[ind, :]).to(self.device)

        states = [states[i, start[i]:end[i], :]
                  for i in range(batch_size)]
        next_states = [next_states[i, start[i]:end[i], :]
                       for i in range(batch_size)]
        actions = [actions[i, start[i]:end[i], :]
                   for i in range(batch_size)]
        rewards = [rewards[i, start[i]:end[i]]
                   for i in range(batch_size)]
        dones = [dones[i, start[i]:end[i]]
                 for i in range(batch_size)]

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)

        return states, actions, next_states, rewards, dones

    def sample_new(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        start = np.random.randint(
            1, self.max_length - 1, size=batch_size)
        # 2, self.max_length - 2, size=batch_size)

        # Ensure ind, start, and end are integers
        ind = ind.astype(int)
        start = start.astype(int)
        # end = end.astype(int)

        # Sample states and actions
        states = torch.FloatTensor(self.state[ind, :, :]).to(self.device)
        actions = torch.FloatTensor(self.action[ind, :, :]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[ind, :]).to(self.device)
        dones = torch.FloatTensor(self.dones[ind, :]).to(self.device)

        states_new = torch.zeros_like(states, device=self.device)
        actions_new = torch.zeros_like(actions, device=self.device)
        rewards_new = torch.zeros_like(rewards, device=self.device)
        dones_new = torch.ones_like(dones, device=self.device)

        for i in range(batch_size):
            # print(f'self.max_length-start[i] {self.max_length-start[i]}')
            # print(f'states[i, start[i]:, :].shape {states[i, start[i]:, :].shape}')

            states_new[i, :self.max_length-start[i],
                       :] = states[i, start[i]:, :]
            actions_new[i, :self.max_length-start[i],
                        :] = actions[i, start[i]:, :]
            rewards_new[i, :self.max_length-start[i]] = rewards[i, start[i]:]
            dones_new[i, :self.max_length-start[i]] = dones[i, start[i]:]

        # print(f'start: {start}')
        # print(f'dones: {dones}')
        # print(f'states: {states.shape}')
        # print(f'dones: {dones.shape}')
        # input(f'states: {states.shape}')
        return states_new.detach(), actions_new.detach(), rewards_new.detach(), dones_new.detach()


class SAPO_Trajectory_ReplayBuffer(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_episode_length,
                 device=None,
                 max_size=int(1e4)):

        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.max_length = max_episode_length

        self.state = torch.zeros((max_size, max_episode_length, state_dim))
        self.action = torch.zeros((max_size, max_episode_length, action_dim))
        self.log_probs = torch.zeros(
            (max_size, max_episode_length, action_dim))
        self.rewards = torch.zeros((max_size, max_episode_length))
        self.dones = torch.zeros((max_size, max_episode_length))

        self.device = device
    
    def add(self, state, action, reward, done, log_probs):
        self.state[self.ptr, :, :] = state
        self.action[self.ptr, :, :] = action
        self.log_probs[self.ptr, :, :] = log_probs
        self.rewards[self.ptr, :] = reward.squeeze()
        self.dones[self.ptr, :] = done.squeeze()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_new(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        start = np.random.randint(
            1, self.max_length - 1, size=batch_size)

        # Ensure ind, start, and end are integers
        ind = ind.astype(int)
        start = start.astype(int)
        # end = end.astype(int)

        # Sample states and actions
        states = torch.FloatTensor(self.state[ind, :, :]).to(self.device)
        actions = torch.FloatTensor(self.action[ind, :, :]).to(self.device)
        log_probs = torch.FloatTensor(
            self.log_probs[ind, :, :]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[ind, :]).to(self.device)
        dones = torch.FloatTensor(self.dones[ind, :]).to(self.device)

        states_new = torch.zeros_like(states, device=self.device)
        actions_new = torch.zeros_like(actions, device=self.device)
        log_probs_new = torch.zeros_like(log_probs, device=self.device)
        rewards_new = torch.zeros_like(rewards, device=self.device)
        dones_new = torch.ones_like(dones, device=self.device)

        for i in range(batch_size):
            # print(f'self.max_length-start[i] {self.max_length-start[i]}')
            # print(f'states[i, start[i]:, :].shape {states[i, start[i]:, :].shape}')

            states_new[i, :self.max_length-start[i],
                       :] = states[i, start[i]:, :]
            actions_new[i, :self.max_length-start[i],
                        :] = actions[i, start[i]:, :]
            log_probs_new[i, :self.max_length-start[i],
                          :] = log_probs[i, start[i]:, :]
            rewards_new[i, :self.max_length-start[i]] = rewards[i, start[i]:]
            dones_new[i, :self.max_length-start[i]] = dones[i, start[i]:]

        # print(f'start: {start}')
        # print(f'dones: {dones}')
        # print(f'states: {states.shape}')
        # print(f'dones: {dones.shape}')
        # input(f'states: {states.shape}')
        return states_new.detach(), actions_new.detach(), rewards_new.detach(), dones_new.detach(), log_probs_new.detach()


class ThreeStep_Action(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """
    Clip the continuous action within the valid :class:`Box` observation space bound.
    """

    def __init__(self, env: gym.Env):
        """
        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

        self.min_action = np.zeros(env.action_space.shape)

        epsilon = 1e-4
        counter = 0
        # for cs in env.charging_stations:
        #     n_ports = cs.n_ports
        #     for i in range(n_ports):
        #         self.min_action[counter] = cs.min_charge_current / \
        #             cs.max_charge_current + epsilon

        #         counter += 1

    def action(self, action: np.ndarray) -> np.ndarray:
        """ 
        If action[i] == 0 then action[i] = 0
        elif action[i] == 1 then action[i] = self.min_action
        else action[i] = 1

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """

        return np.where(action == 0, -1, np.where(action == 1, 0, 1))


class TwoStep_Action(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """
    Clip the continuous action within the valid :class:`Box` observation space bound.
    """

    def __init__(self, env: gym.Env):
        """
        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

        self.min_action = np.zeros(env.action_space.shape)

        # epsilon = 1e-4
        # counter = 0
        # for cs in env.charging_stations:
        #     n_ports = cs.n_ports
        #     for i in range(n_ports):
        #         self.min_action[counter] = cs.min_charge_current / \
        #             cs.max_charge_current + epsilon

        #         counter += 1

    def action(self, action: np.ndarray) -> np.ndarray:
        """ 
        If action[i] == 0 then action[i] = 0
        elif action[i] == 1 then action[i] = self.min_action
        else action[i] = 1

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """

        return np.where(action == 0, -1, 1)
