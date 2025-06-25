import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, mlp_hidden_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, mlp_hidden_dim)
        self.l2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.l3_mu = nn.Linear(mlp_hidden_dim, action_dim)
        self.l3_log_std = nn.Linear(mlp_hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.silu(self.l1(state))
        a = F.silu(self.l2(a))
        mu = self.l3_mu(a)
        log_std = self.l3_log_std(a).clamp(-5, 2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        action = torch.tanh(normal.rsample()) * self.max_action
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        log_prob -= torch.log((1 - action.pow(2)).clamp(min=1e-6)).sum(dim=-1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, mlp_hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, mlp_hidden_dim)
        self.ln1 = nn.LayerNorm(mlp_hidden_dim)
        self.l2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.ln2 = nn.LayerNorm(mlp_hidden_dim)
        self.l3 = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, state):
        x = F.silu(self.ln1(self.l1(state)))
        x = F.silu(self.ln2(self.l2(x)))
        return self.l3(x).squeeze(-1)  # [batch]


def compute_soft_td_lambda(values, rewards, dones, log_probs, entropy_target, gamma=0.99, lam=0.95):
    # values: [ensemble, batch, H+1]
    # rewards, dones, log_probs: [batch, H]
    ensemble, batch_size, H1 = values.shape
    H = H1 - 1
    soft_returns = torch.zeros((ensemble, batch_size, H), device=values.device)
    h_norm = log_probs / entropy_target
    for k in range(ensemble):
        next_value = values[k, :, -1]
        g = next_value
        for t in reversed(range(H)):
            g = rewards[:, t] - h_norm[:, t] + gamma * ((1 - dones[:, t]) * ((1 - lam) * values[k, :, t] + lam * g))
            soft_returns[k, :, t] = g
    return soft_returns  # [ensemble, batch, H]

class SAPO:
    def __init__(self,
                 mlp_hidden_dim,
                 state_dim,
                 action_dim,
                 max_action,
                 transition_fn=None,
                 reward_fn=None,
                 discount=0.99,
                 horizon=32,
                 entropy_target=-0.5,
                 actor_lr=2e-3,
                 critics_lr=2e-3,
                 alpha_lr=5e-3,
                 device='cpu'):

        self.actor = Actor(state_dim, action_dim, max_action, mlp_hidden_dim).to(device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr, betas=(0.7, 0.95))

        self.critic1 = Critic(state_dim, mlp_hidden_dim).to(device)
        self.critic2 = Critic(state_dim, mlp_hidden_dim).to(device)
        self.critics_optimizer = torch.optim.AdamW(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critics_lr, betas=(0.7, 0.95))

        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=alpha_lr, betas=(0.7, 0.95))

        self.transition_fn = transition_fn
        self.reward_fn = reward_fn
        self.discount = discount
        self.horizon = horizon
        self.entropy_target = entropy_target
        self.device = device

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, _ = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def compute_actor_loss(self, states, dones):
        state_pred = states[:, 0, :]
        total_reward = torch.zeros(states.size(0), device=self.device)
        gamma = torch.ones(states.size(0), device=self.device)

        for t in range(self.horizon):
            done = dones[:, t]
            action_pred, log_prob = self.actor(state_pred)

            next_state_pred = self.transition_fn(state=state_pred,
                                                 new_state=states[:, t + 1, :],
                                                 action=action_pred)

            reward_pred = self.reward_fn(state=state_pred, action=action_pred)

            normalized_entropy = log_prob.squeeze() / self.entropy_target

            total_reward += gamma * (reward_pred - self.log_alpha.exp() * normalized_entropy) * (1.0 - done)
            gamma *= self.discount
            state_pred = next_state_pred

        critic_values = torch.stack([critic(state_pred) for critic in self.critics], dim=0)
        v_next = critic_values.mean(dim=0)
        actor_loss = -(total_reward + gamma * v_next.squeeze()).mean()
        return actor_loss, log_prob.mean()
    
    def compute_critic_targets(self, states, rewards, dones, log_probs):
        # states: [batch, H+1, state_dim], rewards, dones, log_probs: [batch, H]
        batch_size, H1, state_dim = states.shape
        H = H1 - 1
        # Compute values for all critics and time steps
        state_batches = states.transpose(0, 1)  # [H+1, batch, state_dim]
        v_ensemble = []
        for critic in self.critics:
            v = torch.stack([critic(state_batches[t]) for t in range(H1)], dim=1)  # [batch, H+1]
            v_ensemble.append(v)
        v_ensemble = torch.stack(v_ensemble, dim=0)  # [num_critics, batch, H+1]
        # Soft TD(λ)
        soft_targets = compute_soft_td_lambda(
            v_ensemble, rewards, dones, log_probs, self.entropy_target,
            gamma=self.discount, lam=self.lambda_td
        )  # [num_critics, batch, H]
        # Min across critics for value target, as in SAPO
        target_v = torch.min(soft_targets, dim=0)[0].detach()  # [batch, H]
        return target_v

    def update_critics(self, states, rewards, dones, log_probs, K=8):
        # K: number of mini-epochs (batches)
        batch_size, H1, state_dim = states.shape
        H = H1 - 1
        target_v = self.compute_critic_targets(states, rewards, dones, log_probs)  # [batch, H]

        for _ in range(K):
            idx = torch.randint(0, batch_size, (batch_size,))
            loss = 0
            for i, critic in enumerate(self.critics):
                pred = []
                for t in range(H):
                    pred.append(critic(states[idx, t]))
                pred = torch.stack(pred, dim=1)  # [batch, H]
                loss += F.mse_loss(pred, target_v[idx])
            loss /= len(self.critics)
            self.critics_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for c in self.critics for p in c.parameters()], 0.5)
            self.critics_optimizer.step()
        return loss.item()

    def train(self, replay_buffer, batch_size=2048):
        states, _, rewards, dones, log_probs = replay_buffer.sample_new(batch_size)

        # Policy update
        self.actor_optimizer.zero_grad()
        actor_loss, log_prob = self.compute_actor_loss(states, dones)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        # Entropy update
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.entropy_target).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

                # Critic update: mini-epoch K
        critic_loss = self.update_critics(states, rewards, dones, log_probs, K=self.num_mini_epochs)

        return {
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'critic_loss': critic_loss,
            'alpha': self.log_alpha.exp().item()
        }

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
