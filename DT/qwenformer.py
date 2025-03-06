
from qwen import Qwen2Model
from qwen import Qwen2Config

import argparse

import torch
import torch.nn as nn

def load_model(vars):
    config = Qwen2Config(
        # vocab_size=1,
        # n_embd=vars['embed_dim'],
        # state_dim=state_dim,
        # act_dim=act_dim,
        # max_length=50,
        # max_ep_len=96,
        # hidden_size=vars['embed_dim'],
        # n_layer=vars['n_layer'],
        # n_head=vars['n_head'],
        # n_inner=4*vars['embed_dim'],
        # activation_function=vars['activation_function'],
        # action_masking=vars['action_masking'],
        # n_positions=1024,
        # resid_pdrop=vars['dropout'],
        # attn_pdrop=vars['dropout'],
        
        vocab_size=1,
        hidden_size=vars['embed_dim'],
        intermediate_size=2*vars['embed_dim'],
        num_hidden_layers=vars['n_layer'],
        num_attention_heads=vars['n_head'],
        num_key_value_heads=vars['n_head'],
        hidden_act="silu",
        max_position_embeddings=1024, #get rid of it
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        attention_dropout=0.0,
    )

    model = Qwen2Model(config)
    model.to("cuda")
    print(model)
    #calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')    
    
    batch_size = 2
    seq_length = 10
    max_ep_len = 96
        
    # embed_timestep = nn.Embedding(max_ep_len, vars['embed_dim'])
    
    inputs_embeds= torch.randn(batch_size, seq_length, vars['embed_dim'], device="cuda")
    attention_mask=None
    if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.long)

    output = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask
    )
    print(output)
    print(f'output shape: {output.last_hidden_state.shape}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PST_V2G_ProfixMax')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--group_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)

    # medium, medium-replay, medium-expert, expert
    parser.add_argument('--dataset', type=str, default='random_1000')
    # normal for standard setting, delayed for sparse
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=3)
    # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--model_type', type=str,
                        default='dt')  # dt, gnn_dt, gnn_in_out_dt, bc, gnn_act_emb
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=15)
    parser.add_argument('--n_head', type=int, default=32)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--num_steps_per_iter', type=int, default=10)  # 1000
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--config_file', type=str,
                        default="v2g_grid_150.yaml")

    parser.add_argument('--num_eval_episodes', type=int, default=2)
    parser.add_argument('--eval_replay_path', type=str,
                        default="./replay/v2g_grid_150_100evals/")
    
    args = parser.parse_args()

    load_model(vars=vars(args))