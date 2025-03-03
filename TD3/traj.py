
from torchviz import make_dot
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, mlp_hidden_dim, dropout=0.1):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, mlp_hidden_dim)
        self.l2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.l3 = nn.Linear(mlp_hidden_dim, action_dim)
        self.dropout = nn.Dropout(dropout)

        self.max_action = max_action

    def forward(self, state):

        a = F.relu(self.l1(state))
        a = self.dropout(a)
        a = F.relu(self.l2(a))
        a = self.dropout(a)

        return self.l3(a)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, mlp_hidden_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, mlp_hidden_dim)
        self.l2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.l3 = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class Traj(object):
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
            sequence_length=3,
            lr=3e-4,
            dropout=0.1,
            loss_fn=None,
            transition_fn=None,
            **kwargs
    ):

        self.actor = Actor(state_dim,
                           action_dim,
                           max_action,
                           mlp_hidden_dim,
                           dropout=dropout
                           ).to(device)
        # self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim, mlp_hidden_dim).to(device)
        # self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr)

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
        self.sequence_length = sequence_length

        self.total_it = 0
        self.loss_dict = {
            # 'critic_loss': 0,
            'physics_loss': 0
            # 'actor_loss': 0
        }

    def select_action(self, state, **kwargs):

        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            actions = self.actor(state).cpu().data.numpy().flatten()
            # apply tanh to actions
            actions = np.tanh(actions)
            return actions

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state = replay_buffer.sample(
            batch_size, self.sequence_length)

        if False:
            # test if loss_fn is working properly
            reward_test = self.loss_fn(state=state,
                                       action=action)
            reward_diff = torch.abs(
                reward.view(-1) - reward_test.view(-1))
            if reward_diff.mean() > 0.001:

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
        
        self.actor.train()
        
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            # noise = (
            #     torch.randn_like(action) * self.policy_noise
            # ).clamp(-self.noise_clip, self.noise_clip)

            # next_action = (
            #     self.actor_target(next_state) + noise
            # ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            # target_Q1 = self.critic_target(next_state, next_action)
            # target_Q = torch.min(target_Q1, target_Q2)
            # target_Q1 = reward #+ not_done * self.discount * target_Q
            state_new = state[:, 0, :]
            for i in range(self.sequence_length-1):
                # state_new = state[:, i, :]
                action_pred = self.actor(state_new)
                reward = self.loss_fn.profit_max(state=state_new,
                                                action=action_pred)

                if i == 0:
                    total_reward = -reward
                else:
                    total_reward -= reward

                state_new = self.transition_fn(state=state_new,
                                            new_state=next_state[:,i+1, :].detach(),
                                            action=action_pred)
                
                if i == 0:
                    action_pred_first = action_pred
            
        # Get current Q estimates
        current_Q = self.critic(state[:, 0, :], action_pred_first)

        print(f'Current Q: {current_Q.shape}')
        print(f'Current Q: {current_Q}')
        print(f'Total Reward: {total_reward.shape}')
        print(f'Total Reward: {total_reward}')
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, total_reward)

        self.loss_dict['critic_loss'] = critic_loss.item()
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(
        #     self.critic.parameters(), max_norm=self.max_norm)
        self.critic_optimizer.step()        

        

        # state_new = state[:, 0, :]
        # for i in range(self.sequence_length-1):
        #     # state_new = state[:, i, :]
        #     action_pred = self.actor(state_new)
        #     reward = self.loss_fn.profit_max(state=state_new,
        #                                      action=action_pred)

        #     if i == 0:
        #         total_reward = -reward
        #     else:
        #         total_reward -= reward

        #     state_new = self.transition_fn(state=state_new,
        #                                    new_state=next_state[:,
        #                                                         i+1, :].detach(),
        #                                    action=action_pred)
        
        

        self.loss_dict['physics_loss'] = total_reward.sum().item()

        # total_reward = -total_reward.mean()
        total_reward = total_reward.sum()
        
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

        self.loss_dict['actor_loss'] = actor_loss.item()
        # dot = make_dot(total_reward)
        # dot.render("autograd_graph", format="png")
        # exit()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        total_reward.backward()

        total_norm = 0.0
        for param in self.actor.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.loss_dict['actor_grad_norm'] = total_norm

        self.actor_optimizer.step()

        return self.loss_dict

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
