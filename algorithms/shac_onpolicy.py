import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from algorithms.utils import compute_target_values
from algorithms.shac import Actor, Critic


class SHAC_OnPolicy:
    def __init__(self,
                 mlp_hidden_dim,
                 state_dim,
                 action_dim,
                 max_action,
                 loss_fn=None,
                 transition_fn=None,
                 discount=0.99,
                 lambda_p=0.95,
                 look_ahead=2,
                 tau=0.005,
                 device='cpu',
                 **kwargs):

        self.actor = Actor(state_dim, action_dim, max_action,
                           mlp_hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, mlp_hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=3e-4)
        
        self.critic_update_steps = kwargs.get('critic_update_steps', 3)

        self.transition_fn = transition_fn
        self.loss_fn = loss_fn
        self.discount = discount
        self.lambda_p = lambda_p
        self.horizon = look_ahead
        self.tau = tau
        self.device = device

    def select_action(self, state, **kwargs):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def compute_policy_loss(self, states, dones):

        state_pred = states[:, 0, :]
        total_reward = 0

        for t in range(self.horizon-1):
            done = dones[:, t]

            action_pred = self.actor(state_pred)

            next_state_pred = self.transition_fn(state=state_pred,
                                                 new_state=states[:, t+1, :],
                                                 action=action_pred)

            reward_pred = self.loss_fn(state=state_pred,
                                       action=action_pred)

            total_reward += (self.discount ** t) * reward_pred * (torch.ones_like(done, device=self.device
                                                                                  ) - done)
            state_pred = next_state_pred

        v_next = self.critic_target(state_pred)
        policy_loss = -(total_reward + (self.discount **
                        (self.horizon-1)) * v_next).mean()

        return policy_loss

    def train(self, replay_buffer, batch_size=64):

        states = replay_buffer.state.detach().to(self.device)
        dones = replay_buffer.dones.detach().to(self.device)

        # Policy update
        self.actor_optimizer.zero_grad()
        policy_loss = self.compute_policy_loss(states, dones)
        policy_loss.backward()
        self.actor_optimizer.step()

        for _ in range(self.critic_update_steps):
            
            states, actions, rewards, dones, _ = replay_buffer.sample(
                batch_size)

            # Compute estimated returns
            with torch.no_grad():
                next_values = self.critic_target(
                    states.reshape(-1, states.shape[-1])).view(batch_size, -1)

                target_values = compute_target_values(rewards,
                                                      next_values,
                                                      dones,
                                                      gamma=self.discount,
                                                      lam=self.lambda_p,
                                                      device=self.device)

            predicted_values = self.critic(
                states.reshape(-1, states.shape[-1])).squeeze(-1)
            value_loss = (
                (predicted_values - target_values.view(-1)) ** 2).mean()
            
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()}

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")
