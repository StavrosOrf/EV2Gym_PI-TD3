'''
This file is used to load the model
'''
import torch
import numpy as np
import yaml
from DT.models.decision_transformer import DecisionTransformer

def load_GNN_act_emb_DecisionTransformer_model(model_path, max_ep_len, env, config,  device):
        
    # load model config yaml file
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model_path = f"saved_models/{model_path}"
    vars_path = f"{model_path}/vars.yaml"

    vars = yaml.load(open(vars_path), Loader=yaml.FullLoader)
    
    if "act_GNN_hidden_dim" not in vars.keys():
        vars['act_GNN_hidden_dim'] = 32
        vars['num_act_gcn_layers'] = 3

        
    model = GNN_act_emb_DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=vars['K'],
        max_ep_len=max_ep_len,
        hidden_size=vars['embed_dim'],
        n_layer=vars['n_layer'],
        n_head=vars['n_head'],
        n_inner=4*vars['embed_dim'],
        activation_function=vars['activation_function'],
        n_positions=1024,
        resid_pdrop=vars['dropout'],
        attn_pdrop=vars['dropout'],
        action_tanh=True,
        action_masking=vars['action_masking'],
        fx_node_sizes={'ev': 5, 'cs': 4, 'tr': 2, 'env': 6},
        feature_dim=vars['feature_dim'],
        GNN_hidden_dim=vars['GNN_hidden_dim'],
        num_gcn_layers=vars['num_gcn_layers'],
        act_GNN_hidden_dim=vars['act_GNN_hidden_dim'],
        num_act_gcn_layers=vars['num_act_gcn_layers'],
        config=config,
        device=device,
    )
    weights = torch.load(f"{model_path}/model.best")
    drop_keys = [
        "predict_action.0.weight",
        "predict_action.0.bias",
        "predict_state.weight",
        "predict_state.bias"
    ]
    weights = {k: v for k, v in weights.items() if k not in drop_keys}
    model.load_state_dict(weights)
    return model

def load_DT_model(model_path, max_ep_len, env,  device):
    '''
    Load the Decision Transformer model using the model path and device'''

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    load_path = f"saved_models/{model_path}"
    state_mean = np.load(f'{load_path}/state_mean.npy')
    state_std = np.load(f'{load_path}/state_std.npy')

    load_model_path = f"{load_path}/model.best"

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model_path = f"saved_models/{model_path}"
    vars_path = f"{model_path}/vars.yaml"

    vars = yaml.load(open(vars_path), Loader=yaml.FullLoader)

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=vars['K'],
        max_ep_len=max_ep_len,
        hidden_size=vars['embed_dim'],
        n_layer=vars['n_layer'],
        n_head=vars['n_head'],
        n_inner=4*vars['embed_dim'],
        activation_function='relu',
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )

    model.load_state_dict(torch.load(load_model_path))
    model.to(device=device)

    return model, state_mean, state_std

def load_QT_model(model_path, max_ep_len, env,  device):
    '''
    Load the Q-Decision Transformer model using the model path and device'''

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    load_path = f"saved_models/{model_path}"

    load_model_path = f"{load_path}/model.best"
    load_critic_path = f"{load_path}/critic.best"

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model_path = f"saved_models/{model_path}"
    vars_path = f"{model_path}/vars.yaml"

    vars = yaml.load(open(vars_path), Loader=yaml.FullLoader)

    model = QT_DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=vars['K'],
        max_ep_len=max_ep_len,
        hidden_size=vars['embed_dim'],
        n_layer=vars['n_layer'],
        n_head=vars['n_head'],
        n_inner=4*vars['embed_dim'],
        activation_function='relu',
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )
    
    critic = Critic(
        state_dim, act_dim, hidden_dim=vars['embed_dim']
    )   
    critic.load_state_dict(torch.load(load_critic_path))
    
    model.load_state_dict(torch.load(load_model_path))
    model.to(device=device)
    critic = critic.to(device=device)

    return model, critic