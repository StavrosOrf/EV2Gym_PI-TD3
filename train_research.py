import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import wandb
import yaml
import random
import time
from tqdm import tqdm
import pickle
import pandas as pd

from agent.state import V2G_grid_state, V2G_grid_state_ModelBasedRL
from agent.reward import V2G_grid_reward, V2G_grid_simple_reward
from agent.loss import VoltageViolationLoss, V2G_Grid_StateTransition
from agent.utils import Trajectory_ReplayBuffer

from ev2gym.models.ev2gym_env import EV2Gym

# from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads
# from GNN.state import PublicPST_GNN, V2G_ProfitMax_with_Loads_GNN

# from GNN.state import PublicPST_GNN_no_position_encoding, PublicPST_GNN_full_graph

# from TD3.TD3_GNN import TD3_GNN
# from TD3.TD3_ActionGNN import TD3_ActionGNN
from TD3.TD3 import TD3
from TD3.Traj import Traj

from TD3.replay_buffer import GNN_ReplayBuffer, ReplayBuffer, ActionGNN_ReplayBuffer

# from SAC.sac import SAC
# from SAC.actionSAC import SAC_ActionGNN

# from GF.action_wrapper import BinaryAction, ThreeStep_Action, Rescale_RepairLayer
# from GF.noise_wrappers import FailedActionCommunication

from gymnasium import Space
from torch_geometric.data import Data


class PyGDataSpace(Space):
    def __init__(self):
        super().__init__((), None)

    def sample(self):
        # Implement this method to generate a random Data object
        pass

    def contains(self, x):
        return isinstance(x, Data)

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment


def eval_policy(policy,
                args,
                eval_config,
                config_file=None,
                ):

    eval_episodes = len(eval_config['eval_replays'])
    
    policy.actor.eval()

    avg_reward = 0.
    stats_list = []
    eval_stats = {}
    for replay in tqdm(eval_config['eval_replays']):
        replay = f'{eval_config["eval_path"]}{replay}'
        eval_env = EV2Gym(config_file=config_file,
                          load_from_replay_path=replay,
                          state_function=eval_config['state_function'],
                          reward_function=eval_config['reward_function'],
                          )

        state, _ = eval_env.reset()
        done = False
        while not done:
            action = policy.select_action(state, return_mapped_action=True)
            state, reward, done, _, stats = eval_env.step(action)
            avg_reward += reward

        stats_list.append(stats)

    # # get the mean and std of the stats
    # for key in stats.keys():
    #     eval_stats['eval_metrics/'+key +
    #                '_mean'] = np.mean([x[key] for x in stats_list])
    #     eval_stats['eval_metrics/'+key +
    #                '_std'] = np.std([x[key] for x in stats_list])

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward, eval_stats


if __name__ == "__main__":

    # log run time
    run_timer = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3") #TD3, Traj
    parser.add_argument("--name", default="base")
    parser.add_argument("--project_name", default="EVs4Grid")
    parser.add_argument("--env", default="EV2Gym")
    parser.add_argument("--config", default="v2g_grid_150.yaml")
    # parser.add_argument("--config", default="v2g_grid_3.yaml")
    parser.add_argument("--seed", default=9, type=int)
    parser.add_argument("--max_timesteps", default=1e7, type=int)  # 1e7
    parser.add_argument("--load_model", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument('--group_name', type=str, default='')

    parser.add_argument("--time_limit_hours", default=200, type=float)  # 1e7

    DEVELOPMENT = False

    if DEVELOPMENT:
        parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
        parser.add_argument("--eval_episodes", default=2, type=int)
        parser.add_argument("--start_timesteps", default=600,
                            type=int)
        parser.add_argument('--eval_freq', default=700, type=int)
        parser.add_argument("--batch_size", default=4, type=int)  # 256
        print(f'!!!!!!!!!!!!!!!! DEVELOPMENT MODE !!!!!!!!!!!!!!!!')
        print(f' Switch to production mode by setting DEVELOPMENT = False')
    else:
        parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
        parser.add_argument("--eval_episodes", default=100, type=int)
        parser.add_argument("--start_timesteps", default=2500,
                            type=int)  # original 25e5
        parser.add_argument("--eval_freq", default=2250,
                            type=int)  # in episodes
        parser.add_argument("--batch_size", default=256, type=int)  # 256

    parser.add_argument("--discount", default=0.99,
                        type=float)     # Discount factor
    # Target network update rate
    parser.add_argument("--tau", default=0.005, type=float)
    # Noise added to target policy during critic update

    # TD3 parameters #############################################
    parser.add_argument("--expl_noise", default=0.1, type=float)  # 0.1
    parser.add_argument("--policy_noise", default=0.2)  # 0.2
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # Save model and optimizer parameters
    parser.add_argument("--save_replay_buffer", action="store_true")
    parser.add_argument("--delete_replay_buffer", action="store_true")
    parser.add_argument("--exp_prefix", default="")
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--replay_buffer_size", default=1e6, type=int)

    # DT parameters #############################################
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    
    parser.add_argument('--activation_function', type=str, default='relu')


    # SAC parameters #############################################
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--policy_SAC', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')

    # GNN Feature Extractor Parameters #############################################
    parser.add_argument('--fx_dim', type=int, default=8)
    parser.add_argument('--fx_GNN_hidden_dim', type=int, default=32)
    parser.add_argument('--fx_num_heads', type=int, default=2)
    parser.add_argument('--mlp_hidden_dim', type=int, default=256)
    parser.add_argument('--discrete_actions', type=int, default=1)
    parser.add_argument('--actor_num_gcn_layers', type=int, default=3)
    parser.add_argument('--critic_num_gcn_layers', type=int, default=3)

    parser.add_argument('--no_positional_encoding', type=bool, default=False)
    parser.add_argument('--full_graph', type=bool, default=False)

    parser.add_argument('--PST_correction_layer', type=bool, default=False)
    parser.add_argument('--noisy_communication', type=float, default=0)

    # Physics loss #############################################
    parser.add_argument('--ph_coeff', type=float, default=10e-5)
    
    parser.add_argument('--K', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=3e-4)

    scale = 1
    args = parser.parse_args()

    if args.discrete_actions > 1 and args.policy != "TD3_ActionGNN":
        raise ValueError(f"{args.policy} does not support discrete actions.")

    device = args.device
    device = device if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    replay_buffer_size = int(args.replay_buffer_size)

    config_file = f"./config_files/{args.config}"

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -")
    print(f'Config File: {config_file}')
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    group_name = "150_SB3_tests"
    reward_function = V2G_grid_simple_reward
    state_function = V2G_grid_state_ModelBasedRL

    config = yaml.load(open(config_file, 'r'),
                       Loader=yaml.FullLoader)

    gym.envs.register(id='evs-v1', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                      kwargs={'config_file': config_file,
                              'generate_rnd_game': True,
                              'reward_function': reward_function,
                              'state_function': state_function,
                              })

    env = gym.make('evs-v1')

    # =========================================================================
    problem_name = config_file.split('/')[-1].split('.')[0]
    eval_replay_path = f'./replay/{problem_name}_{args.eval_episodes}evals/'
    print(f'Looking for replay files in {eval_replay_path}')
    try:
        eval_replay_files = [f for f in os.listdir(
            eval_replay_path) if os.path.isfile(os.path.join(eval_replay_path, f))]
        print(
            f'Found {len(eval_replay_files)} replay files in {eval_replay_path}')

        replays_exist = True

    except:
        replays_exist = False

    def generate_replay(evaluation_name):
        env = EV2Gym(config_file=config_file,
                     generate_rnd_game=True,
                     save_replay=True,
                     replay_save_path=f"{evaluation_name}/",
                     )

        replay_path = f"{evaluation_name}/replay_{env.sim_name}.pkl"

        for _ in range(env.simulation_length):
            actions = np.ones(env.cs)

            new_state, reward, done, truncated, _ = env.step(
                actions, visualize=False)  # takes action

            if done:
                break

        return replay_path

    if not replays_exist:
        eval_replay_files = [generate_replay(
            eval_replay_path) for _ in range(args.eval_episodes)]

    eval_config = {
        'eval_path': eval_replay_path,
        'eval_replays': eval_replay_files,
        'state_function': state_function,
        'reward_function': reward_function,
    }
    # =========================================================================

    global_target_return = 0

    exp_prefix = args.exp_prefix
    if exp_prefix != "":
        load_path = f"saved_models/{exp_prefix}"
    else:
        load_path = None

    loss_fn = VoltageViolationLoss(K=env.get_wrapper_attr('grid').net._K_,
                                   L=env.get_wrapper_attr('grid').net._L_,
                                   s_base=env.get_wrapper_attr(
                                       'grid').net.s_base,
                                   num_buses=env.get_wrapper_attr(
                                       'grid').net.nb,
                                   device=device,
                                   verbose=False,
                                   )

    transition_fn = V2G_Grid_StateTransition(verbose=False,
                                             device=device,
                                             num_buses=env.get_wrapper_attr(
                                                 'grid').net.nb,
                                             )
    
    # transition_fn = None
    
    # Set seeds
    # env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    simulation_length = config["simulation_length"]


    if "SAC" in args.policy:
        group_name = f'{args.group_name}GNN_SAC_{number_of_charging_stations}cs_{n_transformers}tr'
    elif "TD3" in args.policy:
        group_name = f'{args.group_name}GNN_TD3_{number_of_charging_stations}cs_{n_transformers}tr'
    elif "Traj" in args.policy:
        group_name = f'{args.group_name}Traj_{number_of_charging_stations}cs_{n_transformers}tr'
    else:
        raise ValueError("Policy not recognized.")

    if args.load_model == "":
        exp_prefix = f'{args.name}-{random.randint(int(1e5), int(1e6) - 1)}'
    else:
        exp_prefix = args.load_model
    print(f'group_name: {group_name}, exp_prefix: {exp_prefix}')

    save_path = f'./saved_models/{exp_prefix}/'
    # create folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save config file
    with open(f'{save_path}/config.yaml', 'w') as file:
        yaml.dump(config, file)

    # np.save(f'{save_path}/state_mean.npy', state_mean.cpu().numpy())
    # np.save(f'{save_path}/state_std.npy', state_std.cpu().numpy())

    if args.log_to_wandb:

        if args.load_model != "":
            resume_mode = "must"
        else:
            resume_mode = "never"

        wandb.init(
            name=exp_prefix,
            group=group_name,
            id=exp_prefix,
            project=args.project_name,
            entity='stavrosorf',
            save_code=True,
            config=config,
            resume=resume_mode,
        )

        wandb.run.log_code(".")

    kwargs = {
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "fx_dim": args.fx_dim,
        "fx_GNN_hidden_dim": args.fx_GNN_hidden_dim,
        "fx_num_heads": args.fx_num_heads,
        "actor_num_gcn_layers": args.actor_num_gcn_layers,
        "critic_num_gcn_layers": args.critic_num_gcn_layers,
    }

    # # Initialize policy
    # if args.policy == "TD3_GNN" or args.policy == "TD3_ActionGNN":
    #     # Target policy smoothing is scaled wrt the action scale
    #     kwargs["policy_noise"] = args.policy_noise * max_action
    #     kwargs["noise_clip"] = args.noise_clip * max_action
    #     kwargs["policy_freq"] = args.policy_freq
    #     kwargs["device"] = device
    #     kwargs["lr"] = args.lr

    #     kwargs['load_path'] = load_path
    #     kwargs['discrete_actions'] = args.discrete_actions

    #     # if statefunction has attribute node_sizes
    #     if hasattr(state_function, 'node_sizes'):
    #         kwargs['fx_node_sizes'] = state_function.node_sizes

    #     # Save kwargs to local path
    #     with open(f'{save_path}/kwargs.yaml', 'w') as file:
    #         yaml.dump(kwargs, file)

    #     if args.policy == "TD3_GNN":
    #         policy = TD3_GNN(**kwargs)
    #         replay_buffer = GNN_ReplayBuffer(action_dim=action_dim,
    #                                          max_size=replay_buffer_size,)
    #         # save the TD3_GNN.py file using cp
    #         os.system(f'cp TD3/TD3_GNN.py {save_path}')

    #     elif args.policy == "TD3_ActionGNN":
    #         policy = TD3_ActionGNN(**kwargs)
    #         replay_buffer = ActionGNN_ReplayBuffer(action_dim=action_dim,
    #                                                max_size=replay_buffer_size,)
    #         os.system(f'cp TD3/TD3_ActionGNN.py {save_path}')

    if args.policy == "TD3":
        state_dim = env.observation_space.shape[0]
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        kwargs["device"] = device
        kwargs['state_dim'] = state_dim
        kwargs['load_path'] = load_path

        kwargs['loss_fn'] = loss_fn
        kwargs['ph_coeff'] = args.ph_coeff
        
        kwargs['transition_fn'] = transition_fn
        
        kwargs['loss_fn'] = None
        kwargs['transition_fn'] = None

        # kwargs['loss_fn'] = None
        # Save kwargs to local path
        with open(f'{save_path}/kwargs.yaml', 'w') as file:
            yaml.dump(kwargs, file)
        
        os.system(f'cp TD3/TD3.py {save_path}')
        
        policy = TD3(**kwargs)
        replay_buffer = ReplayBuffer(state_dim, action_dim)

    elif args.policy == "Traj":
        
        state_dim = env.observation_space.shape[0]
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        kwargs["device"] = device
        kwargs['state_dim'] = state_dim
        kwargs['load_path'] = load_path
        
        kwargs['loss_fn'] = loss_fn
        kwargs['ph_coeff'] = args.ph_coeff        
        kwargs['transition_fn'] = transition_fn
        kwargs['sequence_length'] = args.K
        kwargs['lr'] = args.lr
        kwargs['dropout'] = args.dropout

        # Save kwargs to local path
        with open(f'{save_path}/kwargs.yaml', 'w') as file:
            yaml.dump(kwargs, file)
        
        os.system(f'cp TD3/Traj.py {save_path}')
        
        policy = Traj(**kwargs)
        replay_buffer = Trajectory_ReplayBuffer(state_dim,
                                                action_dim,
                                                max_episode_length=simulation_length,)
    
    # elif "SAC" in args.policy:

    #     kwargs["device"] = device
    #     kwargs["alpha"] = args.alpha
    #     kwargs["automatic_entropy_tuning"] = args.automatic_entropy_tuning
    #     kwargs["updates_per_step"] = args.updates_per_step
    #     kwargs["target_update_interval"] = args.target_update_interval
    #     kwargs["discount"] = args.discount
    #     kwargs["tau"] = args.tau
    #     kwargs['policy'] = args.policy_SAC
    #     kwargs['lr'] = args.lr
    #     kwargs['hidden_size'] = args.mlp_hidden_dim

    #     if hasattr(state_function, 'node_sizes'):
    #         fx_node_sizes = state_function.node_sizes

        # if args.policy == "SAC_GNN":

        #     policy = SAC(num_inputs=-1,
        #                  action_space=env.action_space,
        #                  args=kwargs,
        #                  fx_node_sizes=fx_node_sizes,
        #                  GNN_fx=True)
        #     replay_buffer = GNN_ReplayBuffer(action_dim=action_dim,
        #                                      max_size=replay_buffer_size,)
        #     os.system(f'cp SAC/sac.py {save_path}')

        # elif args.policy == "SAC_ActionGNN":
        #     policy = SAC_ActionGNN(action_space=env.action_space,
        #                            fx_node_sizes=fx_node_sizes,
        #                            args=kwargs,)
        #     replay_buffer = ActionGNN_ReplayBuffer(action_dim=action_dim,
        #                                            max_size=replay_buffer_size,)
        #     os.system(f'cp SAC/actionSAC.py {save_path}')

        # elif args.policy == "SAC":
        #     state_dim = env.observation_space.shape[0]
        #     policy = SAC(num_inputs=state_dim,
        #                  action_space=env.action_space,
        #                  args=kwargs)
        #     replay_buffer = ReplayBuffer(state_dim, action_dim)
        #     os.system(f'cp SAC/sac.py {save_path}')

    else:
        raise ValueError("Policy not recognized.")

    if args.load_model != "":
        # load using pickle
        with open(f'replay_buffers/{args.load_model}/replay_buffer.pkl', 'rb') as f:
            replay_buffer = pickle.load(f)
        print(f'Loaded replay buffer with {replay_buffer.size} samples.')

        # load the timestep
        with open(f'replay_buffers/{args.load_model}/params.yaml', 'r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
            start_timestep_training = params['timestep']
            print(
                f'Starting training from timestep: {start_timestep_training}')
            best_reward = params['best_reward']
            episode_num = params['episode_num']
    else:
        best_reward = -np.Inf
        start_timestep_training = 0
        episode_num = -1

    # save kwargs to save_path
    with open(f'{save_path}/kwargs.yaml', 'w') as file:
        yaml.dump(kwargs, file)

    if args.load_model != "":
        policy.load(f"./saved_models/{args.load_model}/model.last")

    print(
        f'action_dim: {action_dim}, replay_buffer_size: {replay_buffer_size}')
    print(f'max_episode_length: {simulation_length}')

    # Evaluate untrained policy

    evaluations = []

    updates = 0

    episode_timesteps = -1
    episode_reward = 0

    state, _ = env.reset()
    ep_start_time = time.time()

    time_limit_minutes = int(args.time_limit_hours * 60)
    
    action_traj = torch.zeros((simulation_length, action_dim)).to(device)
    state_traj = torch.zeros((simulation_length, state_dim)).to(device)
    done_traj = torch.zeros((simulation_length, 1)).to(device)
    reward_traj = torch.zeros((simulation_length, 1)).to(device)    
    

    for t in range(start_timestep_training, int(args.max_timesteps)):

        if time.time() - run_timer > time_limit_minutes * 60:
            print(f"Time limit reached. Exiting...")
            break

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps and args.policy != "TD3_ActionGNN" and args.policy != "SAC_ActionGNN":
            action = env.action_space.sample()
            next_state, reward, done, _, stats = env.step(action)
        else:

            if args.policy == "TD3_ActionGNN":
                mapped_action, action = policy.select_action(
                    state, expl_noise=args.expl_noise)

                # Perform action
                next_state, reward, done, _, stats = env.step(mapped_action)

            elif args.policy == "SAC_ActionGNN":
                mapped_action, action = policy.select_action(state,
                                                             evaluate=False,
                                                             return_mapped_action=True)

                # Perform action
                next_state, reward, done, _, stats = env.step(mapped_action)
            elif "SAC" in args.policy:
                action = policy.select_action(state, evaluate=False)
                # Perform action
                next_state, reward, done, _, stats = env.step(action)

            elif args.policy == "TD3" or args.policy == "TD3_GNN" or args.policy == "Traj":
                # Select action randomly or according to policy + add noise
                action = (
                    policy.select_action(state)
                    + np.random.normal(0, max_action *
                                       args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
                # Perform action
                next_state, reward, done, _, stats = env.step(action)
                

        if args.policy != "Traj":
            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, float(done))
        else:
            action_traj[episode_timesteps] = torch.FloatTensor(action).to(device)
            state_traj[episode_timesteps] = torch.FloatTensor(state).to(device)
            done_traj[episode_timesteps] = torch.FloatTensor([done]).to(device)
            reward_traj[episode_timesteps] = torch.FloatTensor([reward]).to(device)
        
        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:

            start_time = time.time()
            if 'SAC' in args.policy:
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = policy.train(
                    replay_buffer, args.batch_size, updates)
                updates += 1

                if args.log_to_wandb:
                    wandb.log({'train/critic_loss': critic_1_loss,
                               'train/critic2_loss': critic_2_loss,
                               'train/actor_loss': policy_loss,
                               'train/ent_loss': ent_loss,
                               'train/alpha': alpha,
                               'train/time': time.time() - start_time, },
                              step=t)

            else:
                loss_dict = policy.train(
                    replay_buffer, args.batch_size)

                if args.log_to_wandb:
                    
                    # log all loss_dict keys, but add train/ in front of their name
                    for key in loss_dict.keys():
                        wandb.log({f'train/{key}': loss_dict[key]},
                                  step=t)
                    wandb.log({
                            #    'train/physics_loss': loss_dict['physics_loss'],
                               'train/time': time.time() - start_time, },
                              step=t)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}" +
                f" Time: {time.time() - ep_start_time:.3f}")
            # Reset environment
            state, _ = env.reset()
            ep_start_time = time.time()
            done = False
            
            if args.policy == "Traj":
                # Store trajectory in replay buffer
                replay_buffer.add(state_traj, action_traj)
                action_traj = torch.zeros((simulation_length, action_dim)).to(device)
                state_traj = torch.zeros((simulation_length, state_dim)).to(device)
                done_traj = torch.zeros((simulation_length, 1)).to(device)
                reward_traj = torch.zeros((simulation_length, 1)).to(device)

            episode_num += 1

            if args.log_to_wandb:
                wandb.log({'train_ep/episode_reward': episode_reward,
                           'train_ep/episode_num': episode_num},
                          step=t)

            episode_reward = 0
            episode_timesteps = -1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:

            avg_reward, eval_stats = eval_policy(policy=policy,
                                                 args=args,
                                                 eval_config=eval_config,
                                                 config_file=config_file,
                                                 )
            evaluations.append(avg_reward)

            if evaluations[-1] > best_reward:
                best_reward = evaluations[-1]

                policy.save(f'saved_models/{exp_prefix}/model.best')

            if args.log_to_wandb:
                wandb.log({'eval/mean_reward': evaluations[-1],
                           'eval/best_reward': best_reward, },
                          step=t)

                wandb.log(eval_stats)

    if args.log_to_wandb:
        wandb.finish()

    policy.save(f'saved_models/{exp_prefix}/model.last')

    # if 'runs_logger.csv' exists and run_name is in the dataframe, update the completeion status

    # open as dataframe

    runs_logger = pd.read_csv('runs_logger.csv', index_col=0)
    runs_logger.index = runs_logger.index.astype(str)
    # update field complete of row with index [run_name] to True

    if exp_prefix in runs_logger.index:
        run_name = exp_prefix
        print(f'Updating run {run_name} to complete...')
        runs_logger.loc[runs_logger.index ==
                        run_name, 'finished_training'] = True

        already_done = runs_logger.loc[runs_logger.index ==
                                       run_name, 'train_hours_done'].values
        runs_logger.loc[runs_logger.index == run_name,
                        'train_hours_done'] = already_done + args.time_limit_hours
    else:
        run_name = exp_prefix.split('-')[0]
        if run_name in runs_logger.index:

            print(f'Updating run {run_name} to complete...')
            runs_logger.loc[runs_logger.index ==
                            run_name, 'finished_training'] = True

            already_done = runs_logger.loc[runs_logger.index ==
                                           run_name, 'train_hours_done'].values
            runs_logger.loc[runs_logger.index == run_name,
                            'train_hours_done'] = already_done + args.time_limit_hours

            # create a new row with index name run_name and the other columns from the old row
            runs_logger.loc[exp_prefix] = runs_logger.loc[runs_logger.index ==
                                                          run_name].values[0]
            # drop the old row
            runs_logger.drop(
                runs_logger.index[runs_logger.index == run_name], inplace=True)

    # save the dataframe
    runs_logger.to_csv('runs_logger.csv')

    if args.save_replay_buffer:
        print("Saving replay buffer for future training...")
        if not os.path.exists(f'replay_buffers/{exp_prefix}'):
            os.makedirs(f'replay_buffers/{exp_prefix}')

        with open(f'replay_buffers/{exp_prefix}/replay_buffer.pkl', 'wb') as f:
            pickle.dump(replay_buffer, f)

        # save a yaml file with timestep size
        with open(f'replay_buffers/{exp_prefix}/params.yaml', 'w') as file:
            yaml.dump({'timestep': t,
                       'best_reward': float(best_reward),
                       'episode_num': episode_num}, file)

    if args.delete_replay_buffer:
        print("Deleting replay buffer...")
        if os.path.exists(f'replay_buffers/{exp_prefix}'):
            os.system(f'rm -r replay_buffers/{exp_prefix}')

    print(f'Best reward: {best_reward}')
    print(
        f'Total run-time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - run_timer))}')

    # run the batch_runer_continue.py script through os.system
    os.system('python batch_runer_continue.py')
