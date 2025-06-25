import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return torch.tanh(self.l3(a))


class BPTT:
    def __init__(self,
                 mlp_hidden_dim,
                 state_dim,
                 action_dim,
                 max_action,
                 loss_fn=None,
                 transition_fn=None,
                 discount=0.99,
                 horizon=5,
                 actor_lr=3e-4,
                 device='cpu',
                 **kwargs):

        self.actor = Actor(state_dim, action_dim, max_action, mlp_hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.loss_fn = loss_fn
        self.transition_fn = transition_fn
        self.discount = discount
        self.horizon = horizon
        self.device = device

    def select_action(self, state, sample=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def compute_actor_loss(self, states, dones):
        state_pred = states[:, 0, :]
        total_reward = torch.zeros(states.size(0), device=self.device)
        gamma = torch.ones(states.size(0), device=self.device)

        for t in range(self.horizon):
            done = dones[:, t]

            action_pred = self.actor(state_pred)

            next_state_pred = self.transition_fn(state=state_pred,
                                                 new_state=states[:, t + 1, :],
                                                 action=action_pred)

            reward_pred = self.loss_fn(state=state_pred, action=action_pred)

            total_reward += gamma * reward_pred * (1.0 - done)

            gamma *= self.discount
            state_pred = next_state_pred

        actor_loss = -total_reward.mean()
        return actor_loss

    def train(self, replay_buffer, batch_size=64):
        states, actions, rewards, dones = replay_buffer.sample_new(batch_size)

        # Policy update
        self.actor_optimizer.zero_grad()
        actor_loss = self.compute_actor_loss(states, dones)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        return {'actor_loss': actor_loss.item()}

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
