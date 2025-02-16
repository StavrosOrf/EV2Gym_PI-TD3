
from agent.state import V2G_grid_state, V2G_grid_state_ModelBasedRL
from agent.reward import V2G_grid_reward, V2G_grid_simple_reward
from agent.loss import VoltageViolationLoss
from agent.utils import ReplayBuffer, ModelBasedRL

import numpy as np
import torch
import yaml
import random
import os
import math

import gymnasium as gym
from tqdm import tqdm

import argparse
import time
import wandb
import ev2gym


def eval_policy(policy,
                args,
                eval_episodes=30,
                ):

    eval_env = gym.make('evs-v1')

    avg_reward = 0.
    # use tqdm to show progress bar
    stats_list = []
    eval_stats = {}

    for _ in tqdm(range(eval_episodes)):
        state, _ = eval_env.reset()
        done = False
        state_list = np.zeros((args.K, env.observation_space.shape[0]))
        while not done:
            state_list[:-1] = state_list[1:]
            state_list[-1] = state

            action = policy.select_action(state, exporation_noise=0)
            state, reward, done, _, stats = eval_env.step(action)
            avg_reward += reward

        stats_list.append(stats)

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward, eval_stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    DEVELOPMENT = False

    if DEVELOPMENT:
        parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
        parser.add_argument("--eval_episodes", default=2, type=int)
        parser.add_argument("--start_timesteps", default=10,
                            type=int)
        parser.add_argument('--eval_freq', default=300, type=int)
        parser.add_argument("--batch_size", default=2, type=int)  # 256
        print(f'!!!!!!!!!!!!!!!! DEVELOPMENT MODE !!!!!!!!!!!!!!!!')
        print(f' Switch to production mode by setting DEVELOPMENT = False')
    else:
        parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
        parser.add_argument("--eval_episodes", default=100, type=int)
        parser.add_argument("--start_timesteps", default=2500,
                            type=int)  # original 25e5
        parser.add_argument("--eval_freq", default=2250,
                            type=int)  # in episodes
        parser.add_argument("--batch_size", default=64, type=int)  # 256

    parser.add_argument("--max_timesteps", default=10_000_000, type=int)
    parser.add_argument("--name", default="ModelBasedRL", type=str)
    parser.add_argument('--group_name', type=str, default='')

    parser.add_argument("--K", default=10, type=int,
                        help="Past timesteps of state")

    args = parser.parse_args()

    # state_function = PublicPST
    # reward_function = SimpleReward

    group_name = "150_SB3_tests"
    reward_function = V2G_grid_simple_reward
    state_function = V2G_grid_state_ModelBasedRL

    config_file = "./config_files/v2g_grid.yaml"

    config = yaml.load(open(config_file, 'r'),
                       Loader=yaml.FullLoader)
    gym.envs.register(id='evs-v1', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                      kwargs={'config_file': config_file,
                              'generate_rnd_game': True,
                              'reward_function': reward_function,
                              'state_function': state_function,
                              })

    env = gym.make('evs-v1')

    exp_prefix = f'{args.name}-ModelBasedRL-{random.randint(int(1e5), int(1e6) - 1)}'

    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    simulation_length = config["simulation_length"]

    # group_name = f'{args.group_name}_ModelBasedRL_V2G_ProfitMax{number_of_charging_stations}cs_{n_transformers}tr'
    if args.log_to_wandb:

        wandb.init(
            name=exp_prefix,
            group=group_name,
            id=exp_prefix,
            entity='stavrosorf',
            project="EVs4Grid",
            save_code=True,
            config=config,
        )
        wandb.run.log_code(".")

    save_path = f'./saved_models/{exp_prefix}/'
    # create folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    state, _ = env.reset()

    print(f"State size: {env.observation_space.shape[0]}")
    print(f"Action size: {env.action_space.shape[0]}")
    print(
        f"Min and Max action: {env.action_space.low[0]} and {env.action_space.high[0]}")
    # print(f'Sequence length: {args.K}')

    # Replay buffer
    replay_buffer = ReplayBuffer(state_dim=env.observation_space.shape[0],
                                 action_dim=env.action_space.shape[0],
                                 max_size=int(1e6))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = VoltageViolationLoss(K=env.grid.net._K_,
                                   L=env.grid.net._L_,
                                   s_base=env.grid.net.s_base,
                                   num_buses=env.grid.net.nb,
                                   device=device,
                                   verbose=False,
                                   )

    # initialize model based RL algorithm
    policy = ModelBasedRL(state_dim=env.observation_space.shape[0],
                          action_dim=env.action_space.shape[0],
                          max_action=env.action_space.high[0],
                          update_cycles=1,
                          final_activation=torch.nn.Tanh(),
                          loss_fn=loss_fn,
                          mlp_hidden_dim=128,
                          )

    episode_timesteps = 0
    episode_reward = 0
    episode_num = -1
    best_reward = -np.inf
    evaluations = []
    ep_start_time = time.time()

    state_list = np.zeros((simulation_length, env.observation_space.shape[0]))
    action_list = np.zeros((simulation_length, env.action_space.shape[0]))

    for t in range(args.max_timesteps):

        # Select action randomly or according to policy + add noise
        action = policy.select_action(state, exporation_noise=0.2)

        state_list[episode_timesteps] = state
        action_list[episode_timesteps] = action

        episode_timesteps += 1
        next_state, reward, done, _, stats = env.step(action)

        replay_buffer.add(state, action)
        #######################################################################################################
        # calculate loss and compare to the reward
        # loss = loss_fn(torch.tensor(action.reshape(1, -1)),
        #                torch.tensor(state.reshape(1, -1)))

        # traffo_overload, user_sat = 0,0
        # for tr in env.transformers:
        #     traffo_overload -= 100 * tr.get_how_overloaded()
        #     print(f'Transformer {tr}')

        # print(f'loss: {loss} | reward: {reward} ({traffo_overload})')
        # input('Press Enter to continue...')
        #######################################################################################################

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            # print(f'Training at timestep {t}')
            start_time = time.time()
            loss = policy.train(replay_buffer,
                                args.batch_size)

            if args.log_to_wandb:
                wandb.log({'train/loss': loss,
                           'train/time': time.time() - start_time, },
                          step=t)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}" +
                f" Time: {time.time() - ep_start_time:.3f}")

            # replay_buffer.add(state_list, action_list)
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
            episode_timesteps = 0

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:

            avg_reward, eval_stats = eval_policy(policy,
                                                 args,
                                                 eval_episodes=args.eval_episodes
                                                 )
            evaluations.append(avg_reward)

            if evaluations[-1] > best_reward:
                best_reward = evaluations[-1]

                print(f"Saving best model with reward {best_reward:.3f}")
                policy.save(f'saved_models/{exp_prefix}/model.best')

            if args.log_to_wandb:
                wandb.log({'eval/mean_reward': evaluations[-1],
                           'eval/best_reward': best_reward, },
                          step=t)

                wandb.log(eval_stats)

    if args.log_to_wandb:
        wandb.finish()
