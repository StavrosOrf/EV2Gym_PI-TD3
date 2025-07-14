
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class PI_DDPG(object):
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
            look_ahead=2,
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

        assert look_ahead >= 1, 'Look ahead should be greater than 1'
        self.look_ahead = look_ahead

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
        states, actions, rewards, dones = replay_buffer.sample_new(
            batch_size)

        with torch.no_grad():
            next_state = states[:, 1, :]
            not_done = 1 - dones[:, 0]
            reward = rewards[:, 0]

            next_action = self.actor_target(next_state)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, next_action)

            target_Q = reward + self.discount * not_done * target_Q.view(-1)

        # Get current Q estimates
        current_Q = self.critic(states[:, 0, :],
                                actions[:, 0, :])

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q.view(-1), target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(
        #     self.critic.parameters(), max_norm=self.max_norm)
        self.critic_optimizer.step()

        self.loss_dict['critic_loss'] = critic_loss.item()


        if False:
            # test if loss_fn is working properly
            reward_test = self.loss_fn(state=states[:, 0, :],
                                        action=actions[:, 0, :])
            reward_diff = torch.abs(
                rewards[:, 0].view(-1) - reward_test.view(-1))
            if reward_diff.mean() > 0.01:

                print(f'Reward diff: {reward_diff.mean()}')
                print(f'Reward: {reward}')
                print(f'Reward Test: {reward_test}')
                input("Error in reward calculation")

            next_state_test = self.transition_fn(states[:, 0, :],
                                                    states[:, 1, :],
                                                    actions[:, 0, :])
            state_diff = torch.abs(states[:, 1, :] - next_state_test)
            if state_diff.mean() > 0.001:
                print(f'State diff: {state_diff.mean()}')
                input("Error in state transition")

        state_pred = states[:, 0, :]

        for i in range(0, self.look_ahead):

            done = dones[:, i]

            discount = self.discount**i

            action_vector = self.actor(state_pred)

            reward_pred = self.loss_fn(state=state_pred,
                                        action=action_vector)

            state_pred = self.transition_fn(state=state_pred,
                                            new_state=states[:, i+1, :],
                                            action=action_vector)

            if i == 0:
                actor_loss = - reward_pred
            else:
                # print(f'Actor Loss: {actor_loss.shape}')
                # print(f'Reward Pred: {reward_pred.shape}')
                # print(f'Done: {done.shape}')
                actor_loss += - discount * reward_pred * \
                    (torch.ones_like(done) - done)

        next_action = self.actor(state_pred)

        actor_loss += - discount * self.discount * \
            self.critic(state_pred, next_action).view(-1) *\
            (torch.ones_like(done) - dones[:, self.look_ahead])

        actor_loss = actor_loss.mean()

        self.loss_dict['physics_loss'] = actor_loss.item()
        self.loss_dict['actor_loss'] = actor_loss.item()


        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

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

    def load(self, filename, map_location=None):
        self.critic.load_state_dict(torch.load(filename + "_critic", weights_only=True, map_location=map_location))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer", weights_only=True, map_location=map_location))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor", weights_only=True, map_location=map_location))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer", weights_only=True, map_location=map_location))
        self.actor_target = copy.deepcopy(self.actor)
