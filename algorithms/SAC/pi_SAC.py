import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from algorithms.SAC.model import soft_update, hard_update
from algorithms.SAC.model import GaussianPolicy, QNetwork, DeterministicPolicy
from algorithms.utils import td_lambda_forward_view


class PI_SAC(object):

    def __name__(self):
        return "PI_SAC"

    def __init__(self,
                 num_inputs,
                 action_space,
                 args):

        self.gamma = args['discount']
        self.tau = args['tau']
        self.alpha = args['alpha']
        self.lambda_ = args['lambda_']
        self.td_lambda_horizon = args['td_lambda_horizon']

        self.look_ahead = args['look_ahead']
        self.critic_enabled = args['critic_enabled']

        self.loss_fn = args['loss_fn']
        self.transition_fn = args['transition_fn']

        self.policy_type = args['policy']
        self.target_update_interval = args['target_update_interval']
        self.automatic_entropy_tuning = args['automatic_entropy_tuning']

        self.device = args['device']
        self.hidden_size = args['hidden_size']
        self.lr = args['lr']
        self.lookahead_critic_reward = args['lookahead_critic_reward']
        self.max_norm = 0.5

        self.critic = QNetwork(
            num_inputs, action_space.shape[0], self.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(
            num_inputs, action_space.shape[0], self.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = - \
                    torch.prod(torch.Tensor(
                        action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(
                    1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

            self.policy = GaussianPolicy(
                num_inputs, action_space.shape[0], self.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(
                num_inputs, action_space.shape[0], self.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

    def select_action(self, state, evaluate=True, **kwargs):
        # if state is numpy array, convert to torch tensor
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def compute_target_values(self, rewards, next_values, dones, gamma=0.99, lam=0.95):
        batch_size, horizon = rewards.shape
        target_values = torch.zeros_like(rewards).to(self.device)

        Ai = torch.zeros(batch_size).to(self.device)
        Bi = torch.zeros(batch_size).to(self.device)
        lam_tensor = torch.ones(batch_size).to(self.device)

        for t in reversed(range(horizon)):
            done_mask = 1. - dones[:, t]
            lam_tensor = lam_tensor * lam * done_mask + (1. - done_mask)

            Ai = done_mask * (lam * gamma * Ai + gamma *
                              next_values[:, t] + (1. - lam_tensor) / (1. - lam) * rewards[:, t])
            Bi = gamma * \
                (next_values[:, t] * (1. - done_mask) +
                 Bi * done_mask) + rewards[:, t]

            target_values[:, t] = (1. - lam) * Ai + lam_tensor * Bi

        return target_values

    def train(self, memory, batch_size, updates, **kwargs):
        # Sample a batch from memory
        # state_batch, action_batch, next_state_batch, reward_batch, not_dones = memory.sample(
        #     batch_size=batch_size)

        states, actions, rewards, dones = memory.sample_new(
            batch_size)

        state_batch = states[:, 0, :]
        action_batch = actions[:, 0, :]
        next_state_batch = states[:, 1, :]
        reward_batch = rewards[:, 0].view(-1, 1)
        not_dones = (torch.ones_like(
            dones[:, 0], device=self.device) - dones[:, 0]).view(-1, 1)

        if self.critic_enabled:
            if self.lookahead_critic_reward == 2:
                with torch.no_grad():
                    next_state_action, next_state_log_pi, _ = self.policy.sample(
                        next_state_batch)

                    qf1_next_target, qf2_next_target = self.critic_target(
                        next_state_batch, next_state_action)
                    min_qf_next_target = torch.min(
                        qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                    next_q_value = reward_batch + not_dones * \
                        self.gamma * (min_qf_next_target)
                # Two Q-functions to mitigate positive bias in the policy improvement step
                qf1, qf2 = self.critic(state_batch, action_batch)
                # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                qf1_loss = F.mse_loss(qf1, next_q_value)
                # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
                qf2_loss = F.mse_loss(qf2, next_q_value)
                qf_loss = qf1_loss + qf2_loss

            elif self.lookahead_critic_reward == 3:
                target_Q = td_lambda_forward_view(
                    rewards=rewards,
                    dones=dones,
                    states=states,
                    actions=actions,
                    critic=self.critic_target,
                    gamma=self.gamma,
                    lambda_=self.lambda_,
                    horizon=self.td_lambda_horizon
                )

                # Get current Q estimates
                current_Q1, current_Q2 = self.critic(states[:, 0, :],
                                                     actions[:, 0, :])
                # Compute critic loss
                qf_loss = F.mse_loss(current_Q1.view(-1), target_Q) +\
                    F.mse_loss(current_Q2.view(-1), target_Q)

            elif self.lookahead_critic_reward == 4:
                # Compute estimated returns
                with torch.no_grad():
                    qf1_next_target, qf2_next_target = self.critic_target(
                        states.view(-1, states.shape[-1]),
                        actions.view(-1, actions.shape[-1]))
                    
                    next_values = (qf1_next_target + qf2_next_target) / 2.0
                    target_values = self.compute_target_values(rewards,
                                                               next_values.view(batch_size, -1),
                                                               dones,
                                                               gamma=self.gamma,
                                                               lam=self.lambda_)

                # Value update
                current_Q1, current_Q2 = self.critic(
                    states.view(-1, states.shape[-1]),
                    actions.view(-1, actions.shape[-1]))
                
                qf_loss = F.mse_loss(current_Q1.view(-1), target_values.view(-1)) +\
                    F.mse_loss(current_Q2.view(-1), target_values.view(-1))

            self.critic_optim.zero_grad()
            qf_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), max_norm=self.max_norm)
            self.critic_optim.step()

        # replace min_qf_pi with the rolled out value
        state_pred = states[:, 0, :]

        total_reward = torch.zeros(states.size(0), device=self.device)
        
        for i in range(0, self.look_ahead):

            done = dones[:, i]

            discount = self.gamma**i

            # action_vector = self.actor(state_pred)
            action_vector, log_pi_vec, _ = self.policy.sample(state_pred)

            reward_pred = self.loss_fn(state=state_pred,
                                       action=action_vector)

            state_pred = self.transition_fn(state=state_pred,
                                            new_state=states[:, i+1, :],
                                            action=action_vector)

            # if i == 0:
            #     actor_loss = - reward_pred
            #     log_pi = log_pi_vec
            # else:
            #     actor_loss += - discount * reward_pred * \
            #         (torch.ones_like(done, device=self.device
            #                          ) - done)
            #     log_pi += log_pi_vec
                
            normalized_entropy = log_pi_vec.squeeze() / self.target_entropy
            total_reward += discount * (reward_pred - self.log_alpha.exp() * normalized_entropy) * (1.0 - done)

        # with torch.no_grad():
        # next_action = self.actor(state_pred)
        next_action, _, _ = self.policy.sample(state_pred)

        if self.critic_enabled:
            qf1_pi, qf2_pi = self.critic_target(state_pred, next_action)
            qf = (qf1_pi + qf2_pi) / 2

            # actor_loss += - discount * self.gamma * \
            #     qf.view(-1) *\
            #     (torch.ones_like(done, device=self.device) -
            #         dones[:, self.look_ahead])

            policy_loss = -(total_reward + discount * self.gamma * qf.squeeze()).mean()

        # actor_loss = actor_loss.mean()
        # print("Actor loss: ", actor_loss.shape)
        # print(f'alpha: {self.alpha}, log_pi: {log_pi.shape}')

        # log_pi = log_pi / self.look_ahead

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        # policy_loss = ((self.alpha * log_pi) + actor_loss).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), max_norm=self.max_norm)
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi_vec +
                           self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        loss_dict = {
            'critic_loss': qf_loss.item() if self.critic_enabled else 0,
            'actor_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha_tlogs': alpha_tlogs.item()
        }
        return loss_dict

    # Save model parameters
    def save(self, save_path):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        print('Saving models to {}'.format(save_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, save_path)

    # Load model parameters
    def load(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(
                checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(
                checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(
                checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
