
# from torchviz import make_dot
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool

from torch_geometric.nn import global_mean_pool

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, mlp_hidden_dim, dropout=0.1):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, mlp_hidden_dim)
        self.l2 = nn.Linear(mlp_hidden_dim, 2*mlp_hidden_dim)
        self.l3 = nn.Linear(2*mlp_hidden_dim, 3*mlp_hidden_dim)
        self.l4 = nn.Linear(3*mlp_hidden_dim, mlp_hidden_dim)
        self.l5 = nn.Linear(mlp_hidden_dim, action_dim)        
        
        self.ln1 = nn.LayerNorm(mlp_hidden_dim)
        self.ln2 = nn.LayerNorm(2*mlp_hidden_dim)
        self.ln3 = nn.LayerNorm(3*mlp_hidden_dim)
        self.ln4 = nn.LayerNorm(mlp_hidden_dim)

        self.max_action = max_action

    def forward(self, state):

        a = F.relu(self.l1(state))
        # add batch normalization
        a = self.ln1(a)
        # a = self.dropout(a)
        a = F.relu(self.l2(a))
        a = self.ln2(a)
        a = F.relu(self.l3(a))
        a = self.ln3(a)
        a = F.relu(self.l4(a))
        a = self.ln4(a)
        # a = self.dropout(a)

        return torch.tanh(self.l5(a))
        return self.l3(a)

class ActorGNN(nn.Module):
    def __init__(self,
                 max_action,
                 fx_node_sizes,
                 feature_dim=8,
                 GNN_hidden_dim=32,
                 num_gcn_layers=3,
                 discrete_actions=1,
                 device=torch.device('cpu')):
        
        super(Actor, self).__init__()

        self.device = device
        self.feature_dim = feature_dim
        self.discrete_actions = discrete_actions
        self.num_gcn_layers = num_gcn_layers

        # Node-specific embedding layers
        self.ev_embedding = nn.Linear(fx_node_sizes['ev'], feature_dim)
        self.cs_embedding = nn.Linear(fx_node_sizes['cs'], feature_dim)
        self.tr_embedding = nn.Linear(fx_node_sizes['tr'], feature_dim)
        self.env_embedding = nn.Linear(fx_node_sizes['env'], feature_dim)

        # GCN layers to extract features with a unified edge index
        self.gcn_conv = GCNConv(feature_dim, GNN_hidden_dim)
        
        if num_gcn_layers == 3:
            self.gcn_layers = nn.ModuleList(
                [GCNConv(GNN_hidden_dim, feature_dim)])

        elif num_gcn_layers == 4:
            self.gcn_layers = nn.ModuleList([GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim),
                                             GCNConv(2*GNN_hidden_dim, feature_dim)])

        elif num_gcn_layers == 5:
            self.gcn_layers = nn.ModuleList([GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim),
                                             GCNConv(2*GNN_hidden_dim, GNN_hidden_dim),
                                             GCNConv(GNN_hidden_dim, feature_dim)])
        elif num_gcn_layers == 6:
            self.gcn_layers = nn.ModuleList([GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim),
                                             GCNConv(2*GNN_hidden_dim, 3*GNN_hidden_dim),
                                             GCNConv(3*GNN_hidden_dim, 2*GNN_hidden_dim),
                                             GCNConv(2*GNN_hidden_dim, feature_dim)])
        else:
            raise ValueError(
                f"Number of Actor GCN layers not supported, use 3, 4, 5, or 6!")

        self.gcn_last = GCNConv(feature_dim, discrete_actions)

        self.max_action = max_action

    def forward(self, state, return_mapper=False):

        if isinstance(state.env_features, np.ndarray):
            ev_features = torch.from_numpy(
                state.ev_features).float().to(self.device)
            cs_features = torch.from_numpy(
                state.cs_features).float().to(self.device)
            tr_features = torch.from_numpy(
                state.tr_features).float().to(self.device)
            env_features = torch.from_numpy(
                state.env_features).float().to(self.device)
            edge_index = torch.from_numpy(
                state.edge_index).long().to(self.device)
        else:
            ev_features = state.ev_features
            cs_features = state.cs_features
            tr_features = state.tr_features
            env_features = state.env_features
            edge_index = state.edge_index

        # edge_index = to_undirected(edge_index)

        total_nodes = ev_features.shape[0] + cs_features.shape[0] + \
            tr_features.shape[0] + env_features.shape[0]

        embedded_x = torch.zeros(
            total_nodes, self.feature_dim, device=self.device).float()

        # Apply embeddings to the corresponding segments
        if len(state.ev_indexes) != 0:
            embedded_x[state.ev_indexes] = self.ev_embedding(ev_features)
            embedded_x[state.cs_indexes] = self.cs_embedding(cs_features)
            embedded_x[state.tr_indexes] = self.tr_embedding(tr_features)

        embedded_x[state.env_indexes] = self.env_embedding(env_features)

        embedded_x = embedded_x.reshape(-1, self.feature_dim)
        embedded_x = F.relu(embedded_x)

        # Apply GCN layers with the unified edge index
        x = self.gcn_conv(embedded_x, edge_index)
        x = F.relu(x)

        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)

        # Residual connection
        # x = embedded_x + x

        x = self.gcn_last(x, edge_index)

        # Bound output to action space
        x = self.max_action * torch.tanh(x)
        # x = self.max_action * torch.sigmoid(x)

        # apply action mask
        # valid_action_indexes = torch.where(node_types == 3, 1, 0)
        if self.discrete_actions > 1:
            x = torch.nn.functional.softmax(x, dim=1)

        x = x.reshape(-1)
        # input()
        # x = x * valid_action_indexes
        if return_mapper:
            return x, None, state.ev_indexes
        else:
            return x


class Traj(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            # fx_node_sizes,
            ph_coeff=1,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            mlp_hidden_dim=256,
            sequence_length=3,
            lr=3e-4,
            loss_fn=None,
            transition_fn=None,
            # fx_dim=8,
            # fx_GNN_hidden_dim=32,
            # discrete_actions=1,
            # actor_num_gcn_layers = 3,  
            **kwargs
    ):

        self.actor = Actor(state_dim,
                           action_dim,
                           max_action,
                           mlp_hidden_dim,
                           ).to(device)
        # self.actor = Actor(max_action,
        #                    feature_dim=fx_dim,
        #                    GNN_hidden_dim=fx_GNN_hidden_dim,
        #                    fx_node_sizes=fx_node_sizes,
        #                    discrete_actions=discrete_actions,
        #                    num_gcn_layers = actor_num_gcn_layers,
        #                    device=self.device
        #                    ).to(self.device)
        
        # self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr)


        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_norm = 0.5

        self.ph_coeff = ph_coeff
        self.loss_fn = loss_fn
        self.transition_fn = transition_fn
        self.sequence_length = sequence_length

        self.total_it = 0
        self.loss_dict = {
            # 'critic_loss': 0,
            'physics_loss': 0
            # 'actor_loss': 0
        }

    def select_action(self, state, **kwargs):

        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            actions = self.actor(state).cpu().data.numpy().flatten()
            # apply tanh to actions
            # actions = np.tanh(actions)
            return actions

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, dones, actions = replay_buffer.sample_new(batch_size)

        # for i in range(100):
        #     state_new = self.transition_fn(state=state[:, i, :],
        #                                    new_state=state[:, i+1, :],
        #                                    action=actions[:, i, :])
        #     print(f'stat_new: {state_new.shape}')
        #     if (state_new*(1-dones[:,i]).unsqueeze(1) -\
        #         state[:, i+1, :]*(1-dones[:,i]).unsqueeze(1)).mean() > 0.001:
        #         print(f'diff: {(state_new - state[:, i+1, :]).mean()}')
        #         input('Error in state transition')

        self.actor.train()

        # with torch.no_grad():
        state_new = state[:, 0, :]
        i = 0
        while True:
        # for i in range(1):
            discount = 0.99  # self.discount ** (i+1)
            # discount = 1  # self.discount ** (i+1)

            if i <= 300:
                noise = (
                    torch.randn_like(actions[:, 0, :]) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)

                action_pred = (
                    self.actor(state_new) + noise
                ).clamp(-self.max_action, self.max_action)

                reward = self.loss_fn.profit_maxV2(state=state_new,
                                                    action=action_pred)

                if i == 0:
                    total_reward = + reward
                else:
                    total_reward += discount * reward * (1-dones[:, i])

                state_new = self.transition_fn(state=state_new,
                                            new_state=state[:,
                                                            i+1, :].detach(),
                                            action=action_pred)
            else:
                with torch.no_grad():
                    action_pred = self.actor(state_new)    
                    
                    reward = self.loss_fn.smooth_profit_max(state=state_new,
                                                    action=action_pred)

                    if i == 0:
                        total_reward = + reward
                    else:
                        total_reward += discount * reward * (1-dones[:, i])

                    state_new = self.transition_fn(state=state_new,
                                                new_state=state[:,
                                                                i+1, :].detach(),
                                                action=action_pred)

            # reward = self.loss_fn.profit_max(state=state_new,
            #                                     action=action_pred)
            

            if sum(dones[:, i]) == dones.shape[0]:
                break

            # if i == 0:
            #     action_pred_first = action_pred.detach()

            i += 1

        total_reward = -total_reward.mean()
        self.loss_dict['physics_loss'] = total_reward.item()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        total_reward.backward()

        total_norm = 0.0
        for param in self.actor.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.loss_dict['actor_grad_norm'] = total_norm

        self.actor_optimizer.step()

        return self.loss_dict

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
