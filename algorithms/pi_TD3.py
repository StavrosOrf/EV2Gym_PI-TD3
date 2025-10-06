
import copy
import numpy as np
from algorithms.utils import td_lambda_forward_view, compute_target_values
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.TD3 import Actor
from algorithms.TD3 import Critic


class PI_TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            device,
            ph_coeff=1,
            discount=0.99,
            tau=0.005,
            lambda_=0.95,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            mlp_hidden_dim=256,
            loss_fn=None,
            transition_fn=None,
            look_ahead=2,
            critic_enabled=True,
            lookahead_critic_reward=1,
            td_lambda_horizon=5,
            **kwargs
    ):

        self.actor = Actor(state_dim, action_dim, max_action,
                           mlp_hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        # self.actor_optimizer = torch.optim.Adam(
        #     self.actor.parameters(), lr=3e-4)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=2e-3, betas=(0.7, 0.95))

        self.critic = Critic(state_dim, action_dim, mlp_hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        # self.critic_optimizer = torch.optim.Adam(
        #     self.critic.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=5e-4, betas=(0.7, 0.95))

        assert look_ahead >= 1, 'Look ahead should be greater than 1'
        self.look_ahead = look_ahead

        self.max_action = max_action
        self.discount = discount
        self.lambda_ = lambda_
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_norm = 0.5

        self.ph_coeff = ph_coeff
        self.loss_fn = loss_fn
        self.transition_fn = transition_fn
        self.critic_enabled = critic_enabled
        self.lookahead_critic_reward = lookahead_critic_reward
        self.td_lambda_horizon = td_lambda_horizon

        self.total_it = 0
        self.loss_dict = {
            'critic_loss': 0,
            'physics_loss': 0,
            'actor_loss': 0
        }
        self.device = device

    def select_action(self, state, **kwargs):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        states, actions, rewards, dones = replay_buffer.sample_new(
            batch_size)

        if self.critic_enabled:            
            # Select action according to policy and add clipped noise

            if self.lookahead_critic_reward == 0:

                with torch.no_grad():
                    noise = (
                        torch.randn_like(
                            actions[:, 0, :], device=self.device) * self.policy_noise
                    ).clamp(-self.noise_clip, self.noise_clip)

                    next_state = states[:, self.look_ahead, :]  # +1
                    not_done = 1 - dones[:, self.look_ahead - 1]  # without -1
                    discount_vector = torch.tensor(
                        [self.discount**(i-1) for i in range(self.look_ahead)],
                        device=states.device).view(1, -1)
                    rewards = rewards[:, :self.look_ahead] * discount_vector
                    reward = torch.sum(rewards[:, :self.look_ahead], dim=1)

                    next_action = (
                        self.actor_target(next_state) + noise
                    ).clamp(-self.max_action, self.max_action)

                    # Compute the target Q value
                    target_Q1, target_Q2 = self.critic_target(
                        next_state, next_action)
                    target_Q = torch.min(target_Q1, target_Q2)

                    target_Q = reward + self.discount**(self.look_ahead) * \
                        not_done * target_Q.view(-1)

            elif self.lookahead_critic_reward == 1:
                with torch.no_grad():
                    noise = (
                        torch.randn_like(
                            actions[:, :self.look_ahead, :], device=self.device) * self.policy_noise
                    ).clamp(-self.noise_clip, self.noise_clip)

                    state_pred = states[:, 0, :]
                    total_reward = torch.zeros(
                        states.shape[0], device=states.device)

                    for i in range(0, self.look_ahead):

                        done = dones[:, i]

                        discount = self.discount**i

                        # action_vector = self.actor_target(state_pred)
                        action_vector = (
                            self.actor_target(state_pred) + noise[:, i, :]
                        ).clamp(-self.max_action, self.max_action)

                        reward_pred = self.loss_fn(state=state_pred,
                                                action=action_vector)

                        state_pred = self.transition_fn(state=state_pred,
                                                        new_state=states[:,
                                                                        i+1, :],
                                                        action=action_vector)

                        total_reward += discount * reward_pred * \
                            (torch.ones_like(done, device=self.device) - done)

                    next_state = state_pred
                    not_done = 1 - dones[:, self.look_ahead - 1]
                    noise = noise[:, -1, :]

                    next_state = states[:, 1, :]
                    not_done = 1 - dones[:, 0]
                    reward = rewards[:, 0]

                    next_action = (
                        self.actor_target(next_state) + noise
                    ).clamp(-self.max_action, self.max_action)

                    # Compute the target Q value
                    target_Q1, target_Q2 = self.critic_target(
                        next_state, next_action)
                    target_Q = torch.min(target_Q1, target_Q2)

                    target_Q = total_reward + self.discount**(self.look_ahead) * \
                        not_done * target_Q.view(-1)

            elif self.lookahead_critic_reward == 3:
                """
                Uses the forward-view TD(lambda) target calculation.
                """
                with torch.no_grad():
                    target_Q = td_lambda_forward_view(
                        rewards=rewards,
                        dones=dones,
                        states=states,
                        actions=actions,
                        critic=self.critic_target,
                        gamma=self.discount,
                        lambda_=self.lambda_,
                        horizon=self.look_ahead,
                    )

            elif self.lookahead_critic_reward == 4:
                # Compute targets without gradients
                temp_states = states[:, :self.look_ahead, :]
                temp_actions = actions[:, :self.look_ahead, :]
                temp_rewards = rewards[:, :self.look_ahead]
                temp_dones = dones[:, :self.look_ahead]
                
                with torch.no_grad():                    
                    
                    qf1_next_target, qf2_next_target = self.critic_target(
                        temp_states.reshape(-1, temp_states.shape[-1]),
                        temp_actions.reshape(-1, temp_actions.shape[-1])
                    )
                    # next_values = (qf1_next_target + qf2_next_target) / 2.0
                    next_values = torch.min(qf1_next_target, qf2_next_target)
                    
                    target_values = compute_target_values(
                        temp_rewards,
                        next_values.view(batch_size, -1),
                        temp_dones,
                        gamma=self.discount,
                        lam=self.lambda_,
                        device=self.device
                    )

                # Compute predictions with gradients
                current_Q1, current_Q2 = self.critic(
                    temp_states.reshape(-1, temp_states.shape[-1]),
                    temp_actions.reshape(-1, temp_actions.shape[-1])
                )

                critic_loss = F.mse_loss(current_Q1.view(-1), target_values.view(-1)) + \
                            F.mse_loss(current_Q2.view(-1), target_values.view(-1))

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(
                #     self.critic.parameters(), max_norm=self.max_norm)
                self.critic_optimizer.step()
                self.loss_dict['critic_loss'] = critic_loss.item()

            else:
                
                with torch.no_grad():
                    noise = (
                        torch.randn_like(
                            actions[:, 0, :], device=self.device) * self.policy_noise
                    ).clamp(-self.noise_clip, self.noise_clip)
                    next_state = states[:, 1, :]
                    not_done = 1 - dones[:, 0]
                    reward = rewards[:, 0]

                    next_action = (
                        self.actor_target(next_state) + noise
                    ).clamp(-self.max_action, self.max_action)

                    # Compute the target Q value
                    target_Q1, target_Q2 = self.critic_target(
                        next_state, next_action)
                    target_Q = torch.min(target_Q1, target_Q2)

                    target_Q = reward + self.discount * \
                        not_done * target_Q.view(-1)

            if self.lookahead_critic_reward <= 3:
                # Get current Q estimates
                current_Q1, current_Q2 = self.critic(states[:, 0, :],
                                                        actions[:, 0, :])
                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1.view(-1), target_Q) +\
                    F.mse_loss(current_Q2.view(-1), target_Q)

                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # torch.nn.utils.clip_grad_norm_(
                #     self.critic.parameters(), max_norm=self.max_norm)
                self.critic_optimizer.step()
                self.loss_dict['critic_loss'] = critic_loss.item()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

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

            for i in range(0, self.look_ahead-1 ):

                done = dones[:, i]

                discount = self.discount**i

                action_vector = self.actor(state_pred)

                reward_pred = self.loss_fn(state=state_pred,
                                           action=action_vector)

                state_pred = self.transition_fn(state=state_pred,
                                                new_state=states[:, i, :],
                                                action=action_vector)

                if i == 0:
                    actor_loss = - reward_pred
                else:
                    actor_loss += - discount * reward_pred * \
                        (torch.ones_like(done, device=self.device
                                         ) - done)

            # with torch.no_grad():
            next_action = self.actor(state_pred)

            if self.critic_enabled:
                actor_loss += - discount * self.discount * \
                    self.critic_target.Q1(state_pred, next_action).view(-1)
                #     *\
                # (torch.ones_like(done, device=self.device) -
                #  dones[:, self.look_ahead])

            actor_loss = actor_loss.mean()

            self.loss_dict['physics_loss'] = actor_loss.item()
            self.loss_dict['actor_loss'] = actor_loss.item()

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
        # torch.save(self.critic.state_dict(), filename + "_critic")
        # torch.save(self.critic_optimizer.state_dict(),
        #            filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        # torch.save(self.actor_optimizer.state_dict(),
        #            filename + "_actor_optimizer")

    def load(self, filename, map_location=None):
        # self.critic.load_state_dict(torch.load(filename + "_critic", weights_only=True))
        # self.critic_optimizer.load_state_dict(
        #     torch.load(filename + "_critic_optimizer"))
        # self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor", weights_only=True, map_location=map_location))
        # self.actor_optimizer.load_state_dict(
        #     torch.load(filename + "_actor_optimizer"))
        # self.actor_target = copy.deepcopy(self.actor)
