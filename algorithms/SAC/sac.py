import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from algorithms.SAC.model import soft_update, hard_update
from algorithms.SAC.model import GaussianPolicy, QNetwork, DeterministicPolicy
from algorithms.sapo import Actor

class SAC(object):

    def __name__(self):
        return "SAC"

    def __init__(self, num_inputs,
                 action_space,
                 args):

        self.gamma = args['discount']
        self.tau = args['tau']
        self.alpha = args['alpha']

        self.policy_type = args['policy']
        self.target_update_interval = args['target_update_interval']
        self.automatic_entropy_tuning = args['automatic_entropy_tuning']

        self.device = args['device']
        self.hidden_size = args['hidden_size']
        self.lr = args['lr']

        self.critic = QNetwork(
            num_inputs, action_space.shape[0], self.hidden_size).to(device=self.device)
        # self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.critic_optim = AdamW(
            self.critic.parameters(), lr=5e-4, betas=(0.7, 0.95))

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
                # self.alpha_optim = Adam([self.log_alpha], lr=self.lr)
                self.alpha_optim = AdamW(
                    [self.log_alpha], lr=5e-3, betas=(0.7, 0.95))

            # self.policy = GaussianPolicy(
            #     num_inputs, action_space.shape[0], self.hidden_size, action_space).to(self.device)
            self.policy = Actor(state_dim=num_inputs,
                                action_dim=action_space.shape[0],
                                max_action=action_space.high[0],
                                mlp_hidden_dim=self.hidden_size,
                                ).to(self.device)
            
            # self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)
            self.policy_optim = AdamW(
                self.policy.parameters(), lr=2e-3, betas=(0.7, 0.95))

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


        action, _ = self.policy(state)

        return action.detach().cpu().numpy()[0]

    def train(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, next_state_batch, reward_batch, not_dones = memory.sample(
            batch_size=batch_size)

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.policy(
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

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi = self.policy(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi +
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
            'critic_loss': qf1_loss.item(),
            'critic_loss2': qf2_loss.item(),
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
