import copy
import numpy as np
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
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

        states = [states[i, start[i]:end[i], :] for i in range(batch_size)]
        next_states = [next_states[i, start[i]:end[i], :]
                       for i in range(batch_size)]
        actions = [actions[i, start[i]:end[i], :] for i in range(batch_size)]

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.stack(actions)

        return states, actions, next_states