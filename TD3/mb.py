
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GlobalAttention, GATConv

import torch_geometric.transforms as T


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, mlp_hidden_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, mlp_hidden_dim)
        self.l2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.l3 = nn.Linear(mlp_hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # return self.max_action * torch.sigmoid(self.l3(a))
        return torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, mlp_hidden_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, mlp_hidden_dim)
        self.l2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.l3 = nn.Linear(mlp_hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, mlp_hidden_dim)
        self.l5 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.l6 = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class MB(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            ph_coeff=1,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            mlp_hidden_dim=256,
            loss_fn=None,
            transition_fn=None,
            **kwargs
    ):

        self.actor = Actor(state_dim, action_dim, max_action,
                           mlp_hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, mlp_hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_norm = 0.5

        self.ph_coeff = ph_coeff
        self.loss_fn = loss_fn
        self.transition_fn = transition_fn

        self.total_it = 0
        self.loss_dict = {
            'critic_loss': 0,
            'physics_loss': 0,
            'actor_loss': 0
        }

    def select_action(self, state, **kwargs):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(
        #     self.critic.parameters(), max_norm=self.max_norm)
        self.critic_optimizer.step()

        self.loss_dict['critic_loss'] = critic_loss.item()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            if self.transition_fn is not None:

                if False:
                    # test if loss_fn is working properly
                    reward_test = self.loss_fn.profit_maxV2(state=state,
                                               action=action)
                    reward_diff = torch.abs(
                        reward.view(-1) - reward_test.view(-1))
                    if reward_diff.mean() > 0.01:

                        print(f'Reward diff: {reward_diff.mean()}')
                        print(f'Reward: {reward}')
                        print(f'Reward Test: {reward_test}')
                        input("Error in reward calculation")

                    next_state_test = self.transition_fn(state,
                                                         next_state,
                                                         action)
                    state_diff = torch.abs(next_state - next_state_test)
                    if state_diff.mean() > 0.001:
                        print(f'State diff: {state_diff.mean()}')
                        input("Error in state transition")

                action_vector = self.actor(state)
                next_state_pred = self.transition_fn(state,
                                                     next_state,
                                                     action_vector)
                                                     
                reward_pred = self.loss_fn.profit_maxV2(state=state,
                                                             action=action_vector)

                # with torch.no_grad():
                next_action = self.actor(next_state_pred)

                actor_loss = - (reward_pred + self.discount *
                                self.critic.Q1(next_state_pred, next_action)).mean()

                self.loss_dict['physics_loss'] = reward_pred.mean().item()
                self.loss_dict['actor_loss'] = actor_loss.item()
                # print(f'Physics loss: {reward_pred.mean().item()}')
                # print(f'Actor loss: {actor_loss.item()}')

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            # torch.nn.utils.clip_grad_norm_(
            #     self.actor.parameters(), max_norm=self.max_norm)

            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            # input()

        return self.loss_dict

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
