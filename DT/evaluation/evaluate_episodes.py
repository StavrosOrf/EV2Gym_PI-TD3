import numpy as np
import torch
from ev2gym.models import ev2gym_env
import os
import tqdm

from ev2gym.models.ev2gym_env import EV2Gym
from utils import PST_V2G_ProfitMax_reward, PST_V2G_ProfitMaxGNN_state, PST_V2G_ProfitMax_state

from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives, profit_maximization, SimpleReward


def evaluate_episode(
        test_env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    test_rewards = []
    test_stats = []

    for test_cycle in tqdm.tqdm(range(len(test_env))):
        env = test_env[test_cycle]
        state, _ = env.reset()

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(
            1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        target_return = torch.tensor(
            target_return, device=device, dtype=torch.float32)

        episode_return, episode_length = 0, 0
        for t in range(max_ep_len):

            # add padding
            actions = torch.cat([actions, torch.zeros(
                (1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return=target_return,
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, truncated, stats = env.step(action)

            cur_state = torch.from_numpy(state).to(
                device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            episode_return += reward
            episode_length += 1

            if done:
                test_stats.append(stats)
                test_rewards.append(episode_return)
                break

    keys_to_keep = [
        'total_reward',
        'total_profits',
        'total_energy_charged',
        'total_energy_discharged',
        'average_user_satisfaction',
        'min_user_satisfaction',
        'power_tracker_violation',
        'total_transformer_overload',

    ]

    stats = {}
    for key in test_stats[0].keys():
        if "opt" in key:
            key_name = "opt/" + key.split("opt_")[1]
            if key.split("opt_")[1] not in keys_to_keep:
                continue
        else:
            if key not in keys_to_keep:
                continue
            key_name = "test/" + key
        stats[key_name] = np.mean([test_stats[i][key]
                                   for i in range(len(test_stats))])

    # stats['mean_test_return'] = np.mean(test_rewards)

    return stats  # , episode_length


def evaluate_episode_rtg(
    test_env,
    exp_prefix,
    state_dim,
    act_dim,
    model,
    model_type,
    max_ep_len=1000,
    scale=1000.,
    state_mean=0.,
    state_std=1.,
    device='cuda',
    target_return=None,
    mode='normal',
    n_test_episodes=10,
    **kwargs
):
    model.eval()
    model.to(device=device)

    if type(state_mean) is np.ndarray:
        state_mean = torch.from_numpy(state_mean).to(device=device)
        state_std = torch.from_numpy(state_std).to(device=device)

    # set state_mean and state_std to 0 and 1 if not provided

    state_mean = torch.zeros(state_dim, device=device)
    state_std = torch.ones(state_dim, device=device)

    test_rewards = []
    test_stats = []

    # env = test_env

    global_target_return = 0

    for test_cycle in tqdm.tqdm(range(len(test_env))):
        env = test_env[test_cycle]
        state, _ = env.reset()

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(
            1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        actions_mask = torch.zeros(
            (1, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)

        ep_return = global_target_return
        target_return = torch.tensor(
            ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(
            0, device=device, dtype=torch.long).reshape(1, 1)

        episode_return, episode_length = 0, 0
        for t in range(max_ep_len):

            # add padding
            actions = torch.cat([actions, torch.zeros(
                (1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            if model_type == 'dt' or model_type == 'gnn_dt' or \
                model_type == 'gnn_in_out_dt' or model_type == 'gnn_act_emb':
                action = model.get_action(
                    (states.to(dtype=torch.float32) - state_mean) / state_std,
                    actions.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                    actions_mask.to(dtype=torch.float32),
                )
            elif model_type == 'dmamba':
                action = model.get_action(
                    (states.to(dtype=torch.float32) - state_mean) / state_std,
                    actions.to(dtype=torch.float32),
                    target_return.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                )
            else:
                raise ValueError(f"model_type {model_type} not recognized")

            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, truncated, stats = env.step(action)

            action_mask = torch.from_numpy(stats['action_mask']).to(
                device=device).reshape(1, act_dim)
            actions_mask = torch.cat([actions_mask, action_mask], dim=0)

            cur_state = torch.from_numpy(state).to(
                device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            if mode != 'delayed':
                pred_return = target_return[0, -1] - (reward/scale)
            else:
                pred_return = target_return[0, -1]
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                 torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            if done:
                test_stats.append(stats)
                test_rewards.append(episode_return)
                break

    keys_to_keep = [
        'total_reward',
        'total_profits',
        'total_energy_charged',
        'total_energy_discharged',
        'average_user_satisfaction',
        'min_user_satisfaction',
        'power_tracker_violation',
        'total_transformer_overload',

    ]

    stats = {}
    for key in test_stats[0].keys():
        if "opt" in key:
            key_name = "opt/" + key.split("opt_")[1]
            if key.split("opt_")[1] not in keys_to_keep:
                continue
        else:
            if key not in keys_to_keep:
                continue
            key_name = "test/" + key
        stats[key_name] = np.mean([test_stats[i][key]
                                   for i in range(len(test_stats))])

    # stats['mean_test_return'] = np.mean(test_rewards)

    return stats  # , episode_length


def evaluate_episode_rtg_from_replays(
        env,
        model,
        max_ep_len=1000,
        scale=1.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
):

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model.eval()
    model.to(device=device)

    # state_mean = torch.from_numpy(state_mean).to(device=device)
    # state_std = torch.from_numpy(state_std).to(device=device)

    state_mean = torch.zeros(state_dim, device=device)
    state_std = torch.ones(state_dim, device=device)

    test_rewards = []
    test_stats = []

    global_target_return = 0

    state, _ = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(
        1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    actions_mask = torch.zeros(
        (1, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = global_target_return
    target_return = torch.tensor(
        ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(
        0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros(
            (1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        
        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            actions_mask.to(dtype=torch.float32),
        )

        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, truncated, stats = env.step(action)

        action_mask = torch.from_numpy(stats['action_mask']).to(
            device=device).reshape(1, act_dim)
        actions_mask = torch.cat([actions_mask, action_mask], dim=0)

        cur_state = torch.from_numpy(state).to(
            device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0, -1] - (reward/scale)
        else:
            pred_return = target_return[0, -1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            test_stats.append(stats)
            test_rewards.append(episode_return)

    return stats, episode_return
