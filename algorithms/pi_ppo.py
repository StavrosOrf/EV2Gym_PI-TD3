import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.sapo import Actor, Critic
from algorithms.utils import ActorWithEntropy
import torch.optim as optim
from algorithms.utils import compute_target_values

torch.autograd.set_detect_anomaly(True)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class PhysicsInformedPPO:
    def __init__(self,
                 mlp_hidden_dim,
                 state_dim,
                 action_dim,
                 max_action,
                 action_space,
                 transition_fn,
                 loss_fn,
                 look_ahead=32,
                 discount=0.99,
                 lambda_=0.95,
                 device='cpu',
                 critic_update_method="td_lambda",  # soft_td_lambda or td_lambda
                 entropy_enabled=False,
                 epsilon=0.2,
                 entropy_coef=0.01,
                 value_loss_coef=0.5,
                 max_grad_norm=0.5,
                 policy_epochs=1,
                 actor_update_steps=1,
                 critic_update_steps=8,
                 **kwargs):

        self.device = device
        self.discount = discount
        self.horizon = look_ahead
        self.critic_update_method = critic_update_method
        self.lam = lambda_
        self.entropy_enabled = entropy_enabled
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.policy_epochs = actor_update_steps
        self.critic_epochs = critic_update_steps

        # Actor with entropy output
        self.actor = ActorWithEntropy(
            state_dim, action_dim, max_action, mlp_hidden_dim).to(device)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=2e-3, betas=(0.7, 0.95))

        self.old_actor = ActorWithEntropy(
            state_dim, action_dim, max_action, mlp_hidden_dim).to(device)
        hard_update(self.old_actor, self.actor)  # Initialize old actor for PPO

        # Critic network
        self.critic = Critic(state_dim, mlp_hidden_dim).to(device)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=5e-4, betas=(0.7, 0.95))
        
        # Initialize log_alpha for entropy regularization
        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.AdamW(
            [self.log_alpha], lr=5e-3, betas=(0.7, 0.95))

        # Physics-informed functions
        self.transition_fn = transition_fn
        self.reward_fn = loss_fn

        self.entropy_target = - \
            torch.prod(torch.Tensor(
                action_space.shape).to(self.device)).item()

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, log_prob, entropy = self.actor(state)
        if evaluate:
            return action.cpu().data.numpy().flatten()
        return action.detach().cpu().numpy().flatten(), log_prob.detach().cpu().numpy().flatten(), entropy.detach().cpu().numpy().flatten()

    def compute_gae(self, rewards, values, dones):
        batch_size, horizon = rewards.shape
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0

        # values = torch.cat([values, torch.zeros(batch_size, 1, device=self.device)], dim=1)

        for t in reversed(range(horizon-1)):
            # print(f"Computing GAE at timestep {t}")
            delta = rewards[:, t] + self.discount * \
                values[:, t + 1] * (1 - dones[:, t]) - values[:, t]
            gae = delta + self.discount * self.lam * (1 - dones[:, t]) * gae
            advantages[:, t] = gae

        # returns = advantages + values[:, :-1]
        return advantages  # , returns

    def compute_actor_loss(self, states, dones):
        state_pred = states[:, 0, :]

        rewards = []  # Store rewards for GAE
        values = torch.zeros(states.size(0), self.horizon,
                             device=self.device)  # Store values
        # Store old log probabilities
        old_log_probs = torch.zeros(states.size(
            0), self.horizon, device=self.device)
        log_probs = []

        for t in range(self.horizon):

            # done = dones[:, t]
            action_pred, log_prob, _ = self.actor(state_pred)

            next_state_pred = self.transition_fn(state=state_pred,
                                                 new_state=states[:, t, :],
                                                 action=action_pred)

            reward_pred = self.reward_fn(state=state_pred,
                                         action=action_pred)

            # Critic value prediction
            values[:, t] = self.critic(state_pred).squeeze(-1)
            rewards.append(reward_pred)
            log_probs.append(log_prob.squeeze(-1))  # Store log probabilities

            _, old_log_prob, _ = self.old_actor(
                state_pred)  # Get old log probabilities
            old_log_probs[:, t] = old_log_prob.squeeze(-1)
            state_pred = next_state_pred

        # Detach old log probabilities to avoid gradient flow
        old_log_probs = old_log_probs.detach()
        values = values.detach()  # Detach critic values to avoid gradient flow

        # Compute GAE advantages
        rewards = torch.stack(rewards, dim=1)  # Shape: [N, K]
        log_probs = torch.stack(log_probs, dim=1)  # Shape: [N, K]

        # print(f"Rewards shape: {rewards.shape}, Values shape: {values.shape}, Dones shape: {dones.shape} - {dones.shape} - Old Log Probs shape: {old_log_probs.shape}, Log Probs shape: {log_probs.shape}")
        advantages = self.compute_gae(rewards, values, dones)  # Shape: [N, K]
        # Normalize advantages
        advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / \
            (advantages.std(dim=1, keepdim=True) + 1e-8)

        ratio = torch.exp(log_probs - old_log_probs)  # Shape: [N, K]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 +
                            self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        if self.entropy_enabled:
            # Compute entropy target
            entropy_target = self.entropy_target
            # Compute actor loss with entropy
            normalized_entropy = log_probs / entropy_target
            actor_loss += self.entropy_coef * self.log_alpha.exp() * normalized_entropy.mean()

        return actor_loss, log_probs.mean()

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
        # for critic in self.critics:
        v = torch.stack([self.critic(state_batches[t])
                        for t in range(H1)], dim=1)  # [batch, H+1]
        v_ensemble.append(v.squeeze(2))  # [batch, H+1]
        # [num_critics, batch, H+1]
        v_ensemble = torch.stack(v_ensemble, dim=0)
        # Soft TD(λ)
        soft_targets = self.compute_soft_td_lambda(
            v_ensemble, rewards, dones, log_probs, self.entropy_target,
            gamma=self.discount, lam=self.lam
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
            # for i, critic in enumerate(self.critics):
            pred = []
            for t in range(H):
                pred.append(self.critic(states[idx, t]))
            pred = torch.stack(pred, dim=1).squeeze(2)  # [batch, H]
            # print(f"pred shape: {pred.shape}, target_v shape: {target_v[idx].shape}")
            loss += F.mse_loss(pred, target_v[idx])
            # loss /= len(self.critics)
            self.critic_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.critic.parameters()], 0.5)
            self.critic_optimizer.step()
        return loss

    def train(self, replay_buffer, batch_size):
        # Extract data
        states = replay_buffer.state.to(self.device)      # [N, K+1, D]
        dones = replay_buffer.dones.to(self.device)      # [N, K]

        states = replay_buffer.state.detach().to(self.device)
        dones = replay_buffer.dones.detach().to(self.device)

        hard_update(self.old_actor, self.actor)  # Update old actor for PPO

        for epoch in range(self.policy_epochs):
            # print(f"Policy update epoch {epoch + 1}/{self.policy_epochs}")
            # Policy update
            self.actor_optimizer.zero_grad()
            actor_loss, log_prob = self.compute_actor_loss(states, dones)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()

            if self.entropy_enabled:
                # Entropy update
                alpha_loss = -(self.log_alpha.exp() * (log_prob +
                                                       self.entropy_target).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

        # Critic update: mini-epoch K

        if self.critic_update_method == "soft_td_lambda":
            critic_loss = self.update_critics(
                replay_buffer, batch_size, K=self.critic_epochs)

        elif self.critic_update_method == "td_lambda":
            for _ in range(self.critic_epochs):

                states, actions, rewards, dones, _ = replay_buffer.sample(
                    batch_size)

                # Compute estimated returns
                with torch.no_grad():
                    next_values = self.critic(
                        states.reshape(-1, states.shape[-1])).view(batch_size, -1)

                    target_values = compute_target_values(rewards,
                                                          next_values,
                                                          dones,
                                                          gamma=self.discount,
                                                          lam=self.lam,
                                                          device=self.device)

                predicted_values = self.critic(
                    states.reshape(-1, states.shape[-1])).squeeze(-1)
                critic_loss = (
                    (predicted_values - target_values.view(-1)) ** 2).mean()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            # 'alpha_loss': alpha_loss.item(),
            'critic_loss': critic_loss.item(),
            # 'alpha': self.log_alpha.exp().item()
        }

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")
