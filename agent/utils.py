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


class SimpleNN(nn.Module):
    def __init__(self, state_dim,
                 action_dim,
                 max_action,
                 mlp_hidden_dim,
                 final_activation=nn.Tanh()
                 ):
        super(SimpleNN, self).__init__()

        self.l1 = nn.Linear(state_dim, mlp_hidden_dim)
        self.l2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.l3 = nn.Linear(mlp_hidden_dim, action_dim)
        self.final_activation = final_activation

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * self.final_activation(self.l3(a))


class ModelBasedRL(object):

    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action=1,
                 mlp_hidden_dim=256,
                 device=torch.device(
                     "cuda" if torch.cuda.is_available() else "cpu"),
                 loss_fn=None,
                 **kwargs
                 ) -> None:

        self.device = device
        self.max_action = max_action

        self.actor = SimpleNN(state_dim,
                              action_dim,
                              max_action,
                              mlp_hidden_dim).to(device)

        # initialize NN with HE initialization
        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.loss_fn = loss_fn

        self.update_cycles = 3

    def select_action(self, state, exporation_noise=0, **kwargs):

        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        actions = self.actor(state).cpu().data.numpy().flatten()

        if exporation_noise != 0:
            noise = np.random.normal(0, exporation_noise,
                                     size=actions.shape[0])
            actions = (actions + noise).clip(0, self.max_action)

        return actions

    def train(self, replay_buffer, batch_size=256):

        self.actor = self.actor.to(self.device)
        total_loss = 0.

        self.actor.train()
        # print(f'Training with {self.update_cycles} cycles')
        for _ in range(self.update_cycles):
            state, action = replay_buffer.sample(batch_size)

            self.optimizer.zero_grad()
            out = self.actor(state)

            loss = self.loss_fn(action=out,
                                state=state)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() / batch_size

        mean_loss = total_loss / self.update_cycles
        return mean_loss

    def save(self, filename):

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.optimizer.state_dict(),
                   filename + "optimizer")

    def load(self, filename):

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.optimizer.load_state_dict(
            torch.load(filename + "optimizer"))


class ReplayBuffer(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action):
        self.state[self.ptr] = state
        self.action[self.ptr] = action

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
        )

class Trajectory_ReplayBuffer(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_episode_length,
                 max_size=int(1e4)):
        
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.max_length = max_episode_length

        self.state = torch.zeros((max_size, max_episode_length, state_dim))
        self.action = torch.zeros((max_size, max_episode_length, action_dim))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action):
        self.state[self.ptr, :, :] = state
        self.action[self.ptr, :, :] = action

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

        states = [states[i, start[i]:end[i], :]
                  for i in range(batch_size)]
        next_states = [next_states[i, start[i]:end[i], :]
                       for i in range(batch_size)]
        actions = [actions[i, start[i]:end[i], :]
                   for i in range(batch_size)]

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.stack(actions)

        return states, actions, next_states
    
        
    def sample_new(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        start = np.random.randint(
            2, self.max_length - 2, size=batch_size)
        

        # Ensure ind, start, and end are integers
        ind = ind.astype(int)
        start = start.astype(int)
        # end = end.astype(int)

        # Sample states and actions
        states = torch.FloatTensor(self.state[ind, :, :]).to(self.device)
        actions = torch.FloatTensor(self.action[ind, :, :]).to(self.device)
        
        states_new = torch.zeros_like(states, device=self.device)
        dones = torch.zeros((states.shape[0], self.max_length), device=self.device)
                
        for i in range(batch_size):
            # print(f'self.max_length-start[i] {self.max_length-start[i]}')
            # print(f'states[i, start[i]:, :].shape {states[i, start[i]:, :].shape}')
            
            states_new[i, :self.max_length-start[i], :] = states[i, start[i]:, :]            
            actions[i, self.max_length-start[i]:, :] = 0
            dones[i, self.max_length-start[i]-1:] = 1    
            
        # print(f'start: {start}')
        # print(f'dones: {dones}')
        # print(f'states: {states.shape}')
        # print(f'dones: {dones.shape}')
        # input(f'states: {states.shape}')
        return states, dones, actions
    
    

class ActionGNN_ReplayBuffer(object):
    def __init__(self, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # state is a dict of type Data
        self.state = [{} for i in range(max_size)]
        self.action = [{} for i in range(max_size)]
        self.next_state = [{} for i in range(max_size)]
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

        edge_index = []
        ev_indexes = np.array([])
        cs_indexes = np.array([])
        tr_indexes = np.array([])
        env_indexes = np.array([])

        edge_counter = 0
        node_counter = 0

        ev_features = np.concatenate(
            [self.state[i].ev_features for i in ind], axis=0)
        cs_features = np.concatenate(
            [self.state[i].cs_features for i in ind], axis=0)
        tr_features = np.concatenate(
            [self.state[i].tr_features for i in ind], axis=0)
        env_features = np.concatenate(
            [self.state[i].env_features for i in ind], axis=0)
        node_types = np.concatenate(
            [self.state[i].node_types for i in ind], axis=0)

        sample_node_length = [len(self.state[i].node_types) for i in ind]

        for i in ind:
            edge_index.append(self.state[i].edge_index + edge_counter)
            ev_indexes = np.concatenate(
                [ev_indexes, self.state[i].ev_indexes + node_counter], axis=0)
            cs_indexes = np.concatenate(
                [cs_indexes, self.state[i].cs_indexes + node_counter], axis=0)
            tr_indexes = np.concatenate(
                [tr_indexes, self.state[i].tr_indexes + node_counter], axis=0)
            env_indexes = np.concatenate(
                [env_indexes, self.state[i].env_indexes + node_counter], axis=0)

            node_counter += len(self.state[i].node_types)
            if self.state[i].edge_index.shape[1] > 0:
                edge_counter += np.max(self.state[i].edge_index)
            else:
                edge_counter += 1

        edge_index = np.concatenate(edge_index, axis=1)

        state_batch = Data(edge_index=torch.from_numpy(edge_index).to(self.device),
                           ev_features=torch.from_numpy(
                               ev_features).to(self.device).float(),
                           cs_features=torch.from_numpy(
                               cs_features).to(self.device).float(),
                           tr_features=torch.from_numpy(
                               tr_features).to(self.device).float(),
                           env_features=torch.from_numpy(
                               env_features).to(self.device).float(),
                           node_types=torch.from_numpy(
                               node_types).to(self.device).float(),
                           sample_node_length=sample_node_length,
                           ev_indexes=ev_indexes,
                           cs_indexes=cs_indexes,
                           tr_indexes=tr_indexes,
                           env_indexes=env_indexes)

        action_batch = torch.concatenate([self.action[i] for i in ind], axis=0)

        edge_index = []
        ev_indexes = np.array([])
        cs_indexes = np.array([])
        tr_indexes = np.array([])
        env_indexes = np.array([])

        edge_counter = 0
        node_counter = 0
        ev_features = np.concatenate(
            [self.next_state[i].ev_features for i in ind], axis=0)
        cs_features = np.concatenate(
            [self.next_state[i].cs_features for i in ind], axis=0)
        tr_features = np.concatenate(
            [self.next_state[i].tr_features for i in ind], axis=0)
        env_features = np.concatenate(
            [self.next_state[i].env_features for i in ind], axis=0)
        node_types = np.concatenate(
            [self.next_state[i].node_types for i in ind], axis=0)

        sample_node_length = [len(self.next_state[i].node_types) for i in ind]

        for i in ind:
            edge_index.append(self.next_state[i].edge_index + edge_counter)
            ev_indexes = np.concatenate(
                [ev_indexes, self.next_state[i].ev_indexes + node_counter], axis=0)
            cs_indexes = np.concatenate(
                [cs_indexes, self.next_state[i].cs_indexes + node_counter], axis=0)
            tr_indexes = np.concatenate(
                [tr_indexes, self.next_state[i].tr_indexes + node_counter], axis=0)
            env_indexes = np.concatenate(
                [env_indexes, self.next_state[i].env_indexes + node_counter], axis=0)

            node_counter += len(self.next_state[i].node_types)
            if self.next_state[i].edge_index.shape[1] > 0:
                edge_counter += np.max(self.next_state[i].edge_index)
            else:
                edge_counter += 1

        edge_index = np.concatenate(edge_index, axis=1)
        next_state_batch = Data(edge_index=torch.from_numpy(edge_index).to(self.device),
                                ev_features=torch.from_numpy(
                                    ev_features).to(self.device).float(),
                                cs_features=torch.from_numpy(
                                    cs_features).to(self.device).float(),
                                tr_features=torch.from_numpy(
                                    tr_features).to(self.device).float(),
                                env_features=torch.from_numpy(
                                    env_features).to(self.device).float(),
                                node_types=torch.from_numpy(
                                    node_types).to(self.device),
                                sample_node_length=sample_node_length,
                                ev_indexes=ev_indexes,
                                cs_indexes=cs_indexes,
                                tr_indexes=tr_indexes,
                                env_indexes=env_indexes)

        return (
            state_batch,
            action_batch,
            next_state_batch,
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

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

        return np.where(action == 0, -1,1)