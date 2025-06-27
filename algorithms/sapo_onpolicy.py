import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.sapo import Actor, Critic

class SAPO_OnPolicy:
    def __init__(self,
                 mlp_hidden_dim,
                 state_dim,
                 action_dim,
                 max_action,
                 action_space,
                 transition_fn=None,
                 loss_fn=None,
                 discount=0.99,
                 look_ahead=32,
                 device='cpu',
                 **kwargs):

        self.device = device
        self.lambda_td = kwargs.get('lambda_td', 0.95)
        self.num_mini_epochs = kwargs.get('num_mini_epochs', 8)

        self.actor = Actor(state_dim, action_dim, max_action,
                           mlp_hidden_dim).to(device)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=2e-3, betas=(0.7, 0.95))

        self.critics = [Critic(state_dim, mlp_hidden_dim).to(device),
                        Critic(state_dim, mlp_hidden_dim).to(device)]
        self.critics_optimizer = torch.optim.AdamW(
            [param for critic in self.critics for param in critic.parameters()],
            lr=5e-4, betas=(0.7, 0.95))

        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.AdamW(
            [self.log_alpha], lr=5e-3, betas=(0.7, 0.95))

        self.transition_fn = transition_fn
        self.reward_fn = loss_fn
        self.discount = discount
        self.horizon = look_ahead
        self.entropy_target = - \
            torch.prod(torch.Tensor(
                action_space.shape).to(self.device)).item()
        self.device = device

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, log_prob = self.actor(state)
        if evaluate:
            return action.cpu().data.numpy().flatten()
        else:
            return action.cpu().data.numpy().flatten(), log_prob.cpu().data.numpy().flatten()

    def compute_actor_loss(self, states, dones):
        state_pred = states[:, 0, :]
        total_reward = torch.zeros(states.size(0), device=self.device)
        # gamma = torch.ones(states.size(0), device=self.device)
        gamma = self.discount
        for t in range(self.horizon-1):

            done = dones[:, t]
            action_pred, log_prob = self.actor(state_pred)

            next_state_pred = self.transition_fn(state=state_pred,
                                                 new_state=states[:, t, :],
                                                 action=action_pred)

            reward_pred = self.reward_fn(state=state_pred,
                                         action=action_pred)

            normalized_entropy = log_prob.squeeze() / self.entropy_target
            total_reward += gamma * \
                (reward_pred - self.log_alpha.exp()
                 * normalized_entropy) * (1.0 - done)

            gamma *= self.discount
            state_pred = next_state_pred

        # Compute values for all critics
        q1 = self.critics[0](state_pred)
        q2 = self.critics[1](state_pred)
        v_next = (q1 + q2) / 2.0  # Average over critics

        actor_loss = -(total_reward + gamma * v_next.squeeze()).mean()
        return actor_loss, log_prob.mean()

    def compute_soft_td_lambda(self, values, rewards, dones, log_probs, entropy_target, gamma=0.99, lam=0.95):
        # values: [ensemble, batch, H+1]
        # rewards, dones, log_probs: [batch, H]
        ensemble, batch_size, H1 = values.shape
        H = H1 - 1

        soft_returns = torch.zeros(
            (ensemble, batch_size, H), device=values.device)
        # h_norm = log_probs / entropy_target
        normalized_entropy = log_probs / entropy_target

        with torch.no_grad():
            h_norm = self.log_alpha.exp() * normalized_entropy

        for k in range(ensemble):
            next_value = values[k, :, -1]
            g = next_value
            for t in reversed(range(H)):
                g = rewards[:, t] - h_norm[:, t] + gamma * \
                    ((1 - dones[:, t]) *
                     ((1 - lam) * values[k, :, t] + lam * g))
                soft_returns[k, :, t] = g
        return soft_returns  # [ensemble, batch, H]

    def compute_critic_targets(self, states, rewards, dones, log_probs):
        # states: [batch, H+1, state_dim], rewards, dones, log_probs: [batch, H]
        batch_size, H1, state_dim = states.shape
        H = H1 - 1
        # Compute values for all critics and time steps
        state_batches = states.transpose(0, 1)  # [H+1, batch, state_dim]
        v_ensemble = []
        for critic in self.critics:
            v = torch.stack([critic(state_batches[t])
                            for t in range(H1)], dim=1)  # [batch, H+1]
            v_ensemble.append(v.squeeze(2))  # [batch, H+1]
        # [num_critics, batch, H+1]
        v_ensemble = torch.stack(v_ensemble, dim=0)
        # Soft TD(λ)
        soft_targets = self.compute_soft_td_lambda(
            v_ensemble, rewards, dones, log_probs, self.entropy_target,
            gamma=self.discount, lam=self.lambda_td
        )  # [num_critics, batch, H]
        # Min across critics for value target, as in SAPO
        target_v = torch.min(soft_targets, dim=0)[0].detach()  # [batch, H]
        return target_v

    def update_critics(self, replay_buffer, batch_size, K=8):
        # K: number of mini-epochs (batches)

        for _ in range(K):
            states, _, rewards, dones, log_probs = replay_buffer.sample(
                batch_size)

            batch_size, H1, state_dim = states.shape
            H = H1 - 1
            target_v = self.compute_critic_targets(
                states, rewards, dones, log_probs)  # [batch, H]

            idx = torch.randint(0, batch_size, (batch_size,))
            loss = 0
            for i, critic in enumerate(self.critics):
                pred = []
                for t in range(H):
                    pred.append(critic(states[idx, t]))
                pred = torch.stack(pred, dim=1).squeeze(2)  # [batch, H]
                # print(f"pred shape: {pred.shape}, target_v shape: {target_v[idx].shape}")
                loss += F.mse_loss(pred, target_v[idx])
            loss /= len(self.critics)
            self.critics_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for c in self.critics for p in c.parameters()], 0.5)
            self.critics_optimizer.step()
        return loss.item()

    def train(self, replay_buffer, batch_size):
        
        states = replay_buffer.state.detach().to(self.device)
        dones = replay_buffer.dones.detach().to(self.device)

        # Policy update
        self.actor_optimizer.zero_grad()
        actor_loss, log_prob = self.compute_actor_loss(states, dones)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        # Entropy update
        alpha_loss = -(self.log_alpha.exp() * (log_prob +
                       self.entropy_target).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Critic update: mini-epoch K
        critic_loss = self.update_critics(
            replay_buffer, batch_size, K=self.num_mini_epochs)

        return {
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'critic_loss': critic_loss,
            'alpha': self.log_alpha.exp().item()
        }

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")
