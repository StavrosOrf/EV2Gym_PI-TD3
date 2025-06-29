import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.sapo import Actor, Critic


class PhysicsInformedPPO:
    def __init__(self,
                 mlp_hidden_dim,
                 state_dim,
                 action_dim,
                 max_action,
                 action_space,
                 transition_fn,
                 loss_fn,
                 discount=0.99,
                 lam=0.95,
                 device='cpu',
                 epsilon=0.2,
                 entropy_coef=0.01,
                 value_loss_coef=0.5,
                 max_grad_norm=0.5,
                 actor_lr=3e-4,
                 critic_lr=1e-3,
                 **kwargs):

        self.device = device
        self.discount = discount
        self.lam = lam
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        self.actor = Actor(state_dim, action_dim, max_action,
                           mlp_hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, mlp_hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.transition_fn = transition_fn
        self.reward_fn = loss_fn

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action, log_prob = self.actor(state)
        if evaluate:
            return action.detach().cpu().data.numpy().flatten()
        else:
            return action.detach().cpu().data.numpy().flatten(), log_prob.cpu().data.numpy().flatten()

    def physics_informed_rollout(self, states, dones):
        state_pred = states[:, 0, :]
        rewards, values, log_probs = [], [], []

        for t in range(states.shape[1] - 1):
            action_pred, log_prob = self.actor(state_pred)
            next_state_pred = self.transition_fn(state=state_pred,
                                                 new_state=states[:, t, :],
                                                 action=action_pred)

            reward_pred = self.reward_fn(state=state_pred,
                                         action=action_pred)

            value_pred = self.critic(state_pred)

            rewards.append(reward_pred)
            values.append(value_pred.squeeze(-1))
            log_probs.append(log_prob.squeeze(-1))

            state_pred = next_state_pred

        rewards = torch.stack(rewards, dim=1)
        values = torch.stack(values, dim=1)
        log_probs = torch.stack(log_probs, dim=1)

        advantages, returns = self.compute_gae(rewards, values, dones[:, :-1])

        return advantages, returns, log_probs, values

    def compute_gae(self, rewards, values, dones):
        batch_size, horizon = rewards.shape
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0

        values = torch.cat([values, torch.zeros(batch_size, 1, device=self.device)], dim=1)

        for t in reversed(range(horizon)):
            delta = rewards[:, t] + self.discount * values[:, t + 1] * (1 - dones[:, t]) - values[:, t]
            gae = delta + self.discount * self.lam * (1 - dones[:, t]) * gae
            advantages[:, t] = gae

        returns = advantages + values[:, :-1]
        return advantages, returns

    def train(self, replay_buffer, epochs=4):
        states = replay_buffer.state.to(self.device).detach()
        dones = replay_buffer.dones.to(self.device).detach()
        old_log_probs = replay_buffer.log_probs.to(self.device).detach()

        rtg = self.compute_rtgs(replay_buffer.rewards.to(self.device).detach())

        advantages, returns, old_log_probs, _ = self.physics_informed_rollout(states, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()
        
        old_log_prob_batch = old_log_probs.reshape(-1)
        advantage_batch = advantages.reshape(-1).float()

        for _ in range(epochs):

            _, _, log_probs, values = self.physics_informed_rollout(states, dones)
            
            log_prob_batch = log_probs[:, :-1].reshape(-1).float()
            # state_batch = states[:, :-1].reshape(-1, states.shape[-1]).float()
                        
            rtg_batch = rtg.reshape(-1).float()
            
            # action_batch, log_prob_batch = self.actor(state_batch)
            # log_prob_batch = log_prob_batch.reshape(-1)

            ratio = torch.exp(log_prob_batch - old_log_prob_batch)

            surr1 = ratio * advantage_batch
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage_batch

            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * log_prob_batch.mean()
            critic_loss = F.mse_loss(values, rtg_batch) * self.value_loss_coef

            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    