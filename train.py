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
import resource

from agent.state import V2G_grid_state_ModelBasedRL
from agent.reward import Grid_V2G_profitmaxV2, V2G_profitmaxV2, V2G_costs_simple, pst_V2G_profitmaxV2
from agent.transition_fn import V2G_Grid_StateTransition
from agent.loss_fn import V2GridLoss

from agent.utils import (Trajectory_ReplayBuffer,
                         ThreeStep_Action,
                         TwoStep_Action,
                         ReplayBuffer,
                         SAPO_Trajectory_ReplayBuffer,
                         ParallelEnvs_ReplayBuffer)

from ev2gym.models.ev2gym_env import EV2Gym

from algorithms.SAC.sac import SAC
from algorithms.SAC.pi_SAC import PI_SAC
from algorithms.ppo import PPO
from algorithms.TD3 import TD3
from algorithms.pi_TD3 import PI_TD3
from algorithms.pi_DDPG import PI_DDPG
from algorithms.shac import SHAC
from algorithms.shac_onpolicy import SHAC_OnPolicy
from algorithms.reinforce import Reinforce
from algorithms.sapo import SAPO
from algorithms.sapo_onpolicy import SAPO_OnPolicy


def eval_policy(policy,
                args,
                eval_config,
                config_file=None,
                ):

    eval_episodes = len(eval_config['eval_replays'])

    avg_reward = 0.
    stats_list = []
    for replay in tqdm(eval_config['eval_replays']):
        replay = f'{eval_config["eval_path"]}{replay}'
        eval_env = EV2Gym(config_file=config_file,
                          load_from_replay_path=replay,
                          state_function=eval_config['state_function'],
                          reward_function=eval_config['reward_function'],
                          )

        if args.discrete_actions == 3:
            eval_env = ThreeStep_Action(eval_env)
        elif args.discrete_actions == 2:
            eval_env = TwoStep_Action(eval_env)

        state, _ = eval_env.reset()
        done = False
        while not done:
            action = policy.select_action(state, evaluate=True)
            if len(action) == 3:
                action = action[0]
            state, reward, done, _, stats = eval_env.step(action)
            avg_reward += reward

        stats_list.append(stats)

    keys_to_keep = [
        'total_profits',
        'total_energy_charged',
        'total_energy_discharged',
        'average_user_satisfaction',
        # 'min_user_satisfaction',
        'voltage_violation',
        'power_tracker_violation',
    ]

    stats = {}
    for key in stats_list[0].keys():
        if "opt" in key:
            key_name = "opt/" + key.split("opt_")[1]
            if key.split("opt_")[1] not in keys_to_keep:
                continue
        else:
            if key not in keys_to_keep:
                continue
            key_name = "eval/" + key
        stats[key_name] = np.mean([stats_list[i][key]
                                   for i in range(len(stats_list))])

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward, stats


if __name__ == "__main__":

    # log run time
    run_timer = time.time()

    parser = argparse.ArgumentParser()

    # sac, pi_sac
    # reinforce, ppo
    # td3, pi_td3
    # pi_ddpg
    # shac
    parser.add_argument("--policy", default="sapo_op")
    parser.add_argument("--name", default="base")
    parser.add_argument("--scenario", default="pst_v2g_profitmax")
    parser.add_argument("--project_name", default="EVs4Grid")
    parser.add_argument("--env", default="EV2Gym")
    parser.add_argument("--config", default="PST_V2G_ProfixMax_150_300.yaml")
    # parser.add_argument("--config", default="v2g_grid_3.yaml")
    parser.add_argument("--seed", default=9, type=int)
    parser.add_argument("--max_timesteps", default=1e7, type=int)  # 1e7
    parser.add_argument("--load_model", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument('--group_name', type=str, default='Full_problem_')

    parser.add_argument("--time_limit_hours", default=200, type=float)  # 1e7
    parser.add_argument('--disable_development_mode', action='store_true',
                        default=False)

    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    parser.add_argument("--eval_episodes", default=50, type=int)
    parser.add_argument("--start_timesteps", default=5000,
                        type=int)  # original 25e5
    parser.add_argument("--eval_freq", default=960,  # 2250
                        type=int)  # in episodes
    parser.add_argument("--batch_size", default=64, type=int)  # 256

    parser.add_argument("--discount", default=0.99,
                        type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)
    # TD3 parameters #############################################
    parser.add_argument("--expl_noise", default=0.1, type=float)  # 0.1
    parser.add_argument("--policy_noise", default=0.2)  # 0.2
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=10, type=int)
    # Save model and optimizer parameters
    parser.add_argument("--save_replay_buffer", action="store_true")
    parser.add_argument("--delete_replay_buffer", action="store_true")
    parser.add_argument("--exp_prefix", default="")
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--replay_buffer_size", default=1e6, type=int)

    # SAC parameters #############################################
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--policy_SAC', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')

    # SHAC parameters #############################################
    parser.add_argument('--N_agents', type=int, default=8, metavar='N',
                        help='Number of parallel environments (default: 12)')

    # PPO parameters #############################################
    parser.add_argument('--eps_clip', type=float, default=0.2)
    parser.add_argument('--update_freq_PPO', type=int, default=4)  # in epochs
    parser.add_argument('--action_std', type=float, default=0.6)
    parser.add_argument('--train_updates_PPO', type=float, default=80)
    parser.add_argument('--action_std_decay_rate', type=float, default=0.05)
    parser.add_argument('--min_action_std', type=float, default=0.1)
    parser.add_argument('--action_std_decay_freq',
                        type=int, default=int(2.5e5))

    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--lr_critic', type=float, default=3e-4)
    # add bollean argument to enable/disable critic
    parser.add_argument('--disable_critic', action='store_true',
                        default=False,
                        help='Enable critic in the policy.')
    parser.add_argument('--lookahead_critic_reward', type=int, default=4)
    parser.add_argument('--lambda_', type=float, default=0.95)
    parser.add_argument('--td_lambda_horizon', type=int, default=30)
    parser.add_argument('--critic_update_steps', type=int, default=3)

    # Parameters #############################################
    parser.add_argument('--mlp_hidden_dim', type=int, default=128)
    parser.add_argument('--discrete_actions', type=int, default=1)

    scale = 1
    args = parser.parse_args()

    if not args.disable_development_mode:
        args.log_to_wandb = False
        args.eval_episodes = 1
        args.start_timesteps = 301
        args.eval_freq = 96*5
        args.batch_size = 3

    device = args.device

    replay_buffer_size = int(args.replay_buffer_size)

    config_file = f"./config_files/{args.config}"

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -")
    print(f'device: {device}')
    print(f'Config File: {config_file}')
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.scenario == "v2g":
        reward_function = V2G_costs_simple
    elif args.scenario == "v2g_profitmax":
        reward_function = V2G_profitmaxV2
    elif args.scenario == "grid_v2g_profitmax":
        reward_function = Grid_V2G_profitmaxV2
    elif args.scenario == 'pst_v2g_profitmax':
        reward_function = pst_V2G_profitmaxV2
    else:
        raise ValueError("Scenario not recognized.")

    state_function = V2G_grid_state_ModelBasedRL

    config = yaml.load(open(config_file, 'r'),
                       Loader=yaml.FullLoader)

    gym.envs.register(id='evs-v1', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                      kwargs={'config_file': config_file,
                              'reward_function': reward_function,
                              'state_function': state_function,
                              #   'load_from_replay_path': replay_path,
                              })

    env = gym.make('evs-v1')

    if args.discrete_actions == 3:
        env = ThreeStep_Action(env)
    elif args.discrete_actions == 2:
        env = TwoStep_Action(env)
    elif args.discrete_actions != 1:
        raise ValueError(
            "Discrete action number not recognized. Only support 1 or 3 at the moment!")

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

    if "pst" in args.scenario:
        loss_fn = V2GridLoss(K=np.zeros(1),
                             L=np.zeros(1),
                             s_base=-1,
                             num_buses=34,
                             device=device,
                             verbose=False,
                             )

        transition_fn = V2G_Grid_StateTransition(verbose=False,
                                                 device=device,
                                                 num_buses=34,
                                                 )
    else:
        loss_fn = V2GridLoss(K=env.get_wrapper_attr('grid').net._K_,
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

    if args.scenario == "v2g":
        loss_fn = loss_fn.V2G_simpleV2
    elif args.scenario == "v2g_profitmax":
        loss_fn = loss_fn.V2G_profit_maxV2
    elif args.scenario == "grid_v2g_profitmax":
        loss_fn = loss_fn.grid_profit_maxV2
    elif args.scenario == "pst_v2g_profitmax":
        loss_fn = loss_fn.pst_V2G_profit_maxV2

    # Set seeds
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    simulation_length = config["simulation_length"]

    group_name = f'{args.group_name}_{args.scenario}_{number_of_charging_stations}cs_{n_transformers}tr'

    if args.load_model == "":
        exp_prefix = f'{args.name}-{random.randint(int(1e5), int(1e6) - 1)}'
    else:
        exp_prefix = args.load_model
    # print(f'group_name: {group_name}, exp_prefix: {exp_prefix}')

    save_path = f'./saved_models/{exp_prefix}/'
    # create folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save config file
    with open(f'{save_path}/config.yaml', 'w') as file:
        yaml.dump(config, file)

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

    state_dim = env.observation_space.shape[0]
    kwargs = {
        "action_dim": action_dim,
        "state_dim": state_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "hidden_size": args.mlp_hidden_dim,
        "device": device,
        "seed": args.seed,
        "loss_fn": loss_fn,
        "transition_fn": transition_fn,
        "alpha": args.alpha,
        "look_ahead": args.K,
        "critic_enabled": not args.disable_critic,
        "lookahead_critic_reward": args.lookahead_critic_reward,
        "lr": args.lr,
        "lr_critic": args.lr_critic,
        "alpha": args.alpha,
        "automatic_entropy_tuning": args.automatic_entropy_tuning,
        "updates_per_step": args.updates_per_step,
        "target_update_interval": args.target_update_interval,
        "policy_freq": args.policy_freq,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "has_continuous_action_space": True if args.discrete_actions == 1 else False,
        "eps_clip": args.eps_clip,
        "action_std": args.action_std,
        "action_std_decay_rate": args.action_std_decay_rate,
        "min_action_std": args.min_action_std,
        "train_updates_PPO": args.train_updates_PPO,
        'td_lambda_horizon': args.td_lambda_horizon,
        "lambda_": args.lambda_,
        'N_agents': args.N_agents,
        'action_space': env.action_space,
        'critic_update_steps': args.critic_update_steps,
    }

    # Save kwargs to local path
    with open(f'{save_path}/kwargs.yaml', 'w') as file:
        yaml.dump(kwargs, file)

    if args.policy == "td3":
        os.system(f'cp algorithms/TD3.py {save_path}')
        policy = TD3(**kwargs)
        replay_buffer = ReplayBuffer(state_dim, action_dim)

    elif args.policy == 'sac':
        kwargs['policy'] = args.policy_SAC
        policy = SAC(num_inputs=state_dim,
                     action_space=env.action_space,
                     args=kwargs)
        replay_buffer = ReplayBuffer(state_dim, action_dim)
        os.system(f'cp algorithms/SAC/sac.py {save_path}')

    elif args.policy == 'pi_sac':

        kwargs['policy'] = args.policy_SAC
        policy = PI_SAC(num_inputs=state_dim,
                        action_space=env.action_space,
                        args=kwargs)
        replay_buffer = Trajectory_ReplayBuffer(state_dim,
                                                action_dim,
                                                device=device,
                                                max_episode_length=simulation_length,)
        os.system(f'cp algorithms/SAC/pi_SAC.py {save_path}')

    elif args.policy == "pi_td3":
        os.system(f'cp algorithms/pi_TD3.py {save_path}')
        policy = PI_TD3(**kwargs)
        replay_buffer = Trajectory_ReplayBuffer(state_dim,
                                                action_dim,
                                                device=device,
                                                max_episode_length=simulation_length,)

    elif args.policy == "shac":

        os.system(f'cp algorithms/shac.py {save_path}')
        policy = SHAC(**kwargs)
        replay_buffer = Trajectory_ReplayBuffer(state_dim,
                                                action_dim,
                                                device=device,
                                                max_episode_length=simulation_length,)
    elif args.policy == "sapo":

        os.system(f'cp algorithms/sapo.py {save_path}')
        policy = SAPO(**kwargs)
        replay_buffer = SAPO_Trajectory_ReplayBuffer(state_dim,
                                                     action_dim,
                                                     device=device,
                                                     max_episode_length=simulation_length,)

    elif args.policy == "shac_op":

        os.system(f'cp algorithms/shac_onpolicy.py {save_path}')
        policy = SHAC_OnPolicy(**kwargs)
        replay_buffer = ParallelEnvs_ReplayBuffer(state_dim,
                                                  action_dim,
                                                  device=device,
                                                  max_episode_length=args.K,
                                                  max_size=args.N_agents,)
    elif args.policy == "sapo_op":

        os.system(f'cp algorithms/sapo_onpolicy.py {save_path}')
        policy = SAPO_OnPolicy(**kwargs)
        replay_buffer = ParallelEnvs_ReplayBuffer(state_dim,
                                                  action_dim,
                                                  device=device,
                                                  max_episode_length=args.K,
                                                  max_size=args.N_agents,)

    elif args.policy == "reinforce":

        os.system(f'cp algorithms/reinforce.py {save_path}')
        policy = Reinforce(**kwargs)
        replay_buffer = Trajectory_ReplayBuffer(state_dim,
                                                action_dim,
                                                device=device,
                                                max_episode_length=simulation_length,)

    elif args.policy == "ppo":
        os.system(f'cp algorithms/ppo.py {save_path}')
        policy = PPO(**kwargs)

    elif args.policy == "pi_DDPG":
        os.system(f'cp algorithms/pi_DDPG.py {save_path}')
        policy = PI_DDPG(**kwargs)
        replay_buffer = Trajectory_ReplayBuffer(state_dim,
                                                action_dim,
                                                max_episode_length=simulation_length,)

    else:
        raise ValueError("Policy not recognized.")

    best_reward = -np.Inf
    start_timestep_training = 0
    episode_num = -1

    # save kwargs to save_path
    with open(f'{save_path}/kwargs.yaml', 'w') as file:
        yaml.dump(kwargs, file)

    if args.load_model != "":
        policy.load(f"./saved_models/{args.load_model}/model.last")

    print("---------------------------------------")
    print(f'action_dim: {action_dim}')
    print(f'max_episode_length: {simulation_length}')
    print("---------------------------------------")

    evaluations = []

    updates = 0

    episode_timesteps = -1
    episode_reward = 0

    shac_trained = False

    state, _ = env.reset()
    ep_start_time = time.time()

    time_limit_minutes = int(args.time_limit_hours * 60)

    if args.policy in ["pi_td3", "pi_DDPG", "shac", 'reinforce', 'pi_sac',
                       'sapo']:
        action_traj = torch.zeros((simulation_length, action_dim)).to(device)
        state_traj = torch.zeros((simulation_length, state_dim)).to(device)
        done_traj = torch.zeros((simulation_length, 1)).to(device)
        reward_traj = torch.zeros((simulation_length, 1)).to(device)

        if args.policy == "sapo":
            log_probs_traj = torch.zeros(
                (simulation_length, 1)).to(device)

        if args.policy == "reinforce":
            log_probs_traj = torch.zeros(
                (simulation_length, action_dim)).to(device)
            entropy_traj = torch.zeros(
                (simulation_length, action_dim)).to(device)

    if args.policy in ['shac_op', 'sapo_op']:

        print(f'Using {args.N_agents} parallel environments.')
        envs = [gym.make('evs-v1') for _ in range(args.N_agents)]
        states = [env.reset()[0] for env in envs]
        rewards = np.zeros(args.N_agents)
        episode_num = 0
        
        print(f'RAM usage: {os.getpid()} {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB')

        print(f'Starting training...')
        for t in range(start_timestep_training, int(args.max_timesteps)):
            
            start_time = time.time()
            #  exploration phase
            for n, env in enumerate(envs):
                for step in range(args.K):

                    if n == 0:
                        episode_timesteps += 1

                    if args.policy == "shac_op":
                        action = policy.select_action(
                            states[n], evaluate=False)

                    elif args.policy == "sapo_op":
                        action, log_prob = policy.select_action(
                            states[n], evaluate=False)

                    next_state, reward, done, _, stats = env.step(action)

                    rewards[n] += reward

                    replay_buffer.state[n][step] = torch.FloatTensor(
                        states[n]).to(device)
                    replay_buffer.action[n][step] = torch.FloatTensor(
                        action).to(device)
                    replay_buffer.rewards[n][step] = torch.FloatTensor(
                        [reward]).to(device)
                    replay_buffer.dones[n][step] = torch.FloatTensor(
                        [done]).to(device)

                    if args.policy == "sapo_op":
                        replay_buffer.log_probs[n][step] = torch.FloatTensor(
                            log_prob).to(device)

                    if done:
                        states[n], _ = env.reset()
                    else:
                        states[n] = next_state
            
            #  training phase            
            loss_dict = policy.train(
                replay_buffer, args.batch_size)

            if done:

                print(
                    f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} AvgReward: {np.mean(rewards):.3f}" +
                    f" Time: {time.time() - start_time:.3f}")
                
                if args.log_to_wandb:
                    wandb.log({'train_ep/episode_reward': np.mean(rewards),
                               'train_ep/episode_num': episode_num},
                              step=t)

                    for key in loss_dict.keys():
                        wandb.log({f'train/{key}': loss_dict[key]},
                                  step=t)
                    wandb.log({
                        'train/time': time.time() - start_time, },
                        step=t)

                episode_num += 1
                episode_timesteps = -1
                rewards = np.zeros(args.N_agents)

            # Evaluate episode
            if (episode_num) % (args.eval_freq // simulation_length) == 0 and done:

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
                    wandb.log({'eval_a/mean_reward': evaluations[-1],
                               'eval_a/best_reward': best_reward, },
                              step=t)

                    wandb.log(eval_stats,
                              step=t)

    else:
        for t in range(start_timestep_training, int(args.max_timesteps)):

            episode_timesteps += 1

            if args.policy in ["sac", "shac", "pi_sac", "shac_op"]:
                action = policy.select_action(state, evaluate=False)
                next_state, reward, done, _, stats = env.step(action)
            elif args.policy in ["sapo"]:
                action, log_prob = policy.select_action(state, evaluate=False)
                next_state, reward, done, _, stats = env.step(action)

            elif args.policy == "reinforce":
                action, log_prob, entropy = policy.select_action(state)
                next_state, reward, done, _, stats = env.step(action)

            elif args.policy == "ppo":
                action = policy.select_action(state, evaluate=False)
                next_state, reward, done, _, stats = env.step(action)

            elif args.policy in ['td3', 'pi_td3', 'pi_DDPG']:
                # Select action randomly or according to policy + add noise
                action = (
                    policy.select_action(state)
                    + np.random.normal(0, max_action *
                                       args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
                # Perform action
                next_state, reward, done, _, stats = env.step(action)
            else:
                raise ValueError("Policy not recognized.")

            if args.policy == "ppo":
                policy.buffer.rewards.append(reward)
                policy.buffer.is_terminals.append(done)

            elif args.policy not in ["pi_td3", "pi_DDPG", "shac", 'reinforce', 'pi_sac', 'shac_op', 'sapo']:
                # Store data in replay buffer
                replay_buffer.add(state, action, next_state,
                                  reward, float(done))
            else:
                action_traj[episode_timesteps] = torch.FloatTensor(
                    action).to(device)
                state_traj[episode_timesteps] = torch.FloatTensor(
                    state).to(device)
                done_traj[episode_timesteps] = torch.FloatTensor(
                    [done]).to(device)
                reward_traj[episode_timesteps] = torch.FloatTensor(
                    [reward]).to(device)
                if args.policy == "sapo":
                    # Store log probabilities in trajectory
                    log_probs_traj[episode_timesteps] = torch.FloatTensor(
                        log_prob).to(device)

                if args.policy == "reinforce":
                    # print(f'log_prob: {log_prob}, entropy: {entropy} step: {episode_timesteps}')
                    log_probs_traj[episode_timesteps] = log_prob.to(device)
                    entropy_traj[episode_timesteps] = entropy.to(device)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:

                start_time = time.time()
                if args.policy == 'sac' or args.policy == 'pi_sac':

                    if t % args.policy_freq == 0:
                        loss_dict = policy.train(
                            replay_buffer, args.batch_size, updates)
                        updates += 1
                    else:
                        loss_dict = None
                elif args.policy in ['shac', 'sapo']:
                    if t % args.policy_freq == 0:
                        loss_dict = policy.train(
                            replay_buffer, args.batch_size)

                elif args.policy == "reinforce":
                    pass

                elif args.policy == "ppo":
                    if t % (args.update_freq_PPO * simulation_length) == 0:
                        loss_dict = policy.train()
                    else:
                        loss_dict = None

                else:
                    loss_dict = policy.train(
                        replay_buffer, args.batch_size)

                if args.log_to_wandb and policy != "reinforce" and loss_dict is not None:

                    for key in loss_dict.keys():
                        wandb.log({f'train/{key}': loss_dict[key]},
                                  step=t)
                    wandb.log({
                        'train/time': time.time() - start_time, },
                        step=t)

            if done:

                if args.policy in ["pi_td3", "pi_DDPG", "shac", 'reinforce', 'pi_sac', 'shac_op', 'sapo']:
                    # Store trajectory in replay buffer

                    if args.policy == "sapo":
                        # Store log probabilities and entropy in trajectory
                        replay_buffer.add(state_traj,
                                          action_traj,
                                          reward_traj,
                                          done_traj,
                                          log_probs=log_probs_traj,
                                          )
                        log_probs_traj = torch.zeros(
                            (simulation_length, 1)).to(device)
                    else:
                        replay_buffer.add(state_traj,
                                          action_traj,
                                          reward_traj,
                                          done_traj)

                    action_traj = torch.zeros(
                        (simulation_length, action_dim)).to(device)
                    state_traj = torch.zeros(
                        (simulation_length, state_dim)).to(device)
                    done_traj = torch.zeros((simulation_length, 1)).to(device)
                    reward_traj = torch.zeros(
                        (simulation_length, 1)).to(device)

                if args.policy == "reinforce":
                    start_time = time.time()
                    loss_dict = policy.train(
                        replay_buffer, args.batch_size)

                    action_traj.zero_()
                    state_traj.zero_()
                    reward_traj.zero_()
                    done_traj.zero_()
                    log_probs_traj.detach_()
                    entropy_traj.detach_()
                    log_probs_traj.zero_()
                    entropy_traj.zero_()

                    if args.log_to_wandb:

                        # log all loss_dict keys, but add train/ in front of their name
                        for key in loss_dict.keys():
                            wandb.log({f'train/{key}': loss_dict[key]},
                                      step=t)
                        wandb.log({
                            #    'train/physics_loss': loss_dict['physics_loss'],
                            'train/time': time.time() - start_time, },
                            step=t)

                elif args.policy == "shac_op" and episode_num % args.N_agents == 0 and episode_num > 0:
                    print(
                        f"Training SHAC on-policy at timestep {t} for episode {episode_num}...")
                    start_time = time.time()
                    loss_dict = policy.train(replay_buffer, args.batch_size)
                    shac_trained = True

                    if args.log_to_wandb:

                        # log all loss_dict keys, but add train/ in front of their name
                        for key in loss_dict.keys():
                            wandb.log({f'train/{key}': loss_dict[key]},
                                      step=t)
                        wandb.log({
                            'train/time': time.time() - start_time, },
                            step=t)

                print(
                    f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}" +
                    f" Time: {time.time() - ep_start_time:.3f}")
                # Reset environment
                state, _ = env.reset()
                ep_start_time = time.time()
                done = False

                episode_num += 1

                if args.log_to_wandb:
                    wandb.log({'train_ep/episode_reward': episode_reward,
                               'train_ep/episode_num': episode_num},
                              step=t)

                episode_reward = 0
                episode_timesteps = -1

            # Evaluate episode
            if ((t + 1) % args.eval_freq == 0 and t + 100 >= args.start_timesteps) or shac_trained:
                if shac_trained:
                    shac_trained = False
                elif args.policy == "shac_op":
                    continue

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
                    wandb.log({'eval_a/mean_reward': evaluations[-1],
                               'eval_a/best_reward': best_reward, },
                              step=t)

                    wandb.log(eval_stats,
                              step=t)

    if args.log_to_wandb:
        wandb.finish()

    policy.save(f'saved_models/{exp_prefix}/model.last')

    # if 'runs_logger.csv' exists and run_name is in the dataframe, update the completeion status

    # open as dataframe

    # runs_logger = pd.read_csv('runs_logger.csv', index_col=0)
    # runs_logger.index = runs_logger.index.astype(str)
    # # update field complete of row with index [run_name] to True

    # if exp_prefix in runs_logger.index:
    #     run_name = exp_prefix
    #     print(f'Updating run {run_name} to complete...')
    #     runs_logger.loc[runs_logger.index ==
    #                     run_name, 'finished_training'] = True

    #     already_done = runs_logger.loc[runs_logger.index ==
    #                                    run_name, 'train_hours_done'].values
    #     runs_logger.loc[runs_logger.index == run_name,
    #                     'train_hours_done'] = already_done + args.time_limit_hours
    # else:
    #     run_name = exp_prefix.split('-')[0]
    #     if run_name in runs_logger.index:

    #         print(f'Updating run {run_name} to complete...')
    #         runs_logger.loc[runs_logger.index ==
    #                         run_name, 'finished_training'] = True

    #         already_done = runs_logger.loc[runs_logger.index ==
    #                                        run_name, 'train_hours_done'].values
    #         runs_logger.loc[runs_logger.index == run_name,
    #                         'train_hours_done'] = already_done + args.time_limit_hours

    #         # create a new row with index name run_name and the other columns from the old row
    #         runs_logger.loc[exp_prefix] = runs_logger.loc[runs_logger.index ==
    #                                                       run_name].values[0]
    #         # drop the old row
    #         runs_logger.drop(
    #             runs_logger.index[runs_logger.index == run_name], inplace=True)

    # save the dataframe
    # runs_logger.to_csv('runs_logger.csv')

    # if args.save_replay_buffer:
    #     print("Saving replay buffer for future training...")
    #     if not os.path.exists(f'replay_buffers/{exp_prefix}'):
    #         os.makedirs(f'replay_buffers/{exp_prefix}')

    #     with open(f'replay_buffers/{exp_prefix}/replay_buffer.pkl', 'wb') as f:
    #         pickle.dump(replay_buffer, f)

    #     # save a yaml file with timestep size
    #     with open(f'replay_buffers/{exp_prefix}/params.yaml', 'w') as file:
    #         yaml.dump({'timestep': t,
    #                    'best_reward': float(best_reward),
    #                    'episode_num': episode_num}, file)

    # if args.delete_replay_buffer:
    #     print("Deleting replay buffer...")
    #     if os.path.exists(f'replay_buffers/{exp_prefix}'):
    #         os.system(f'rm -r replay_buffers/{exp_prefix}')

    # print(f'Best reward: {best_reward}')
    # print(
    #     f'Total run-time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - run_timer))}')

    # # run the batch_runer_continue.py script through os.system
    # os.system('python batch_runer_continue.py')
