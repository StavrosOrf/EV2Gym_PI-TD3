import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable
import math




def normal(x, mu, sigma_sq, device):
    pi = Variable(torch.FloatTensor([math.pi])).to(device)
    x = Variable(x).to(device)
    a = (-1 * (x - mu).pow(2) / (2 * sigma_sq)).exp()
    b = 1 / (2 * sigma_sq * pi.expand_as(sigma_sq)).sqrt()
    return a * b


class Policy(nn.Module):
    def __init__(self, hidden_size, state_dim, action_dim):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_dim)
        self.linear2_ = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        mu = self.linear2(x)
        sigma_sq = F.softplus(self.linear2_(x))
        return mu, sigma_sq


class Reinforce:
    def __init__(self, state_dim, action_dim, max_action, mlp_hidden_dim=256,
                 gamma=0.99, lr=1e-3, device='cpu', **kwargs):

        self.device = device
        self.gamma = gamma
        self.max_action = max_action

        self.actor = Policy(mlp_hidden_dim, state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.actor.train()

    def select_action(self, state, **kwargs):
        pi = Variable(torch.FloatTensor([math.pi])).to(self.device)
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        mu, sigma_sq = self.actor(state)
        eps = torch.randn_like(mu, device=self.device)
        action = mu + sigma_sq.sqrt() * eps
        prob = normal(action, mu, sigma_sq, self.device)
        entropy = -0.5 * ((sigma_sq + 2 * pi.expand_as(sigma_sq)).log() + 1)
        log_prob = prob.log()
        return action.detach().cpu().numpy().flatten(), log_prob, entropy

    def train(self, rewards, log_probs, entropies):
        print("Training Reinforce with {} rewards".format(len(rewards)))
        R = torch.zeros(1).to(self.device)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[i]
            loss -= (log_probs[i] * R).sum() + (0.0001 * entropies[i]).sum()
        loss = loss / len(rewards)

        self.actor_optimizer.zero_grad()
        loss.backward()
        # utils.clip_grad_norm_(self.actor.parameters(), 40)
        self.actor_optimizer.step()
        return {'policy_loss': loss.item()}
    
    def save(self, path):
        torch.save(self.actor.state_dict(), path + '_reinforce_actor.pth')
        print(f"Model saved to {path}_reinforce_actor.pth")