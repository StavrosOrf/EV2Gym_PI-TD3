"""
This script is used to evaluate the performance of the ev2gym environment.
"""
from ev2gym.models.ev2gym_env import EV2Gym

from ev2gym.baselines.heuristics import RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible

from agent.state import V2G_grid_state, V2G_grid_state_ModelBasedRL
from agent.reward import V2G_grid_reward, V2G_grid_simple_reward
from agent.loss import VoltageViolationLoss

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pandas as pd
import torch


def eval():
    """
    Runs an evaluation of the ev2gym environment.
    """

    save_plots = False

    replay_path = "./replay/replay_sim_2025_02_14_047599.pkl"
    replay_path = None

    # config_file = "ev2gym/example_config_files/PublicPST.yaml"
    # config_file = "ev2gym/example_config_files/BusinessPST.yaml"
    config_file = "./config_files/v2g_grid.yaml"

    env = EV2Gym(config_file=config_file,
                 load_from_replay_path=replay_path,
                 verbose=False,
                 save_replay=False,
                 save_plots=save_plots,
                 state_function=V2G_grid_state_ModelBasedRL,
                 reward_function=V2G_grid_simple_reward,
                 )

    print(env.action_space)
    print(env.observation_space)
    new_replay_path = f"replay/replay_{env.sim_name}.pkl"

    agent = ChargeAsFastAsPossible()
    # agent = ChargeAsFastAsPossibleToDesiredCapacity()

    max_cs_power = env.charging_stations[0].get_max_power()
    min_cs_power = env.charging_stations[0].get_min_power()

    # print(f'Max CS power: {max_cs_power}')
    # print(f'Min CS power: {min_cs_power}')

    ev_battery_capacity = env.EVs_profiles[0].battery_capacity
    ev_min_battery_capacity = env.EVs_profiles[0].min_battery_capacity
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    loss_fn = VoltageViolationLoss(K=env.grid.net._K_,
                                   L=env.grid.net._L_,
                                   s_base=env.grid.net.s_base,
                                   num_buses=env.grid.net.nb,
                                   max_cs_power=max_cs_power,
                                   min_cs_power=min_cs_power,
                                   ev_battery_capacity=ev_battery_capacity,
                                   ev_min_battery_capacity=ev_min_battery_capacity,
                                   device=device,
                                   verbose=True,
                                   )

    succesful_runs = 0
    failed_runs = 0

    results_df = None

    for i in range(100):
        state, _ = env.reset()
        for t in range(env.simulation_length):
            actions = agent.get_action(env)*1

            new_state, reward, done, truncated, stats = env.step(
                actions)  # takes action

            loss, v = loss_fn.forward_v2(action=torch.tensor(actions, device=device).reshape(1, -1),
                                         state=torch.tensor(state, device=device).reshape(1, -1))

            v_m = env.node_voltage[1:, t]
            print(f'V real: {v_m}')
            print(f'V pred: {v}')
            print(f'v_loss {np.abs(v - v_m).mean()}')
            
            loss_v = np.minimum(np.zeros_like(v_m), 0.05 - np.abs(1-v_m))

            print(f'Loss V: {loss_v}')
            reward_loss = np.abs(reward - loss.cpu().detach().numpy())
            print(f'Reward Loss: {reward_loss} | Reward: {reward} | Loss: {loss} | Loss V sum: {1000*loss_v.sum()}')

            if reward_loss != 0 or reward != 0 or loss != 0:                
                input("Press Enter to continue...\n------------------------------------------------------------------------")

            state = new_state

            if done and truncated:
                print(f"Voltage limits exceeded step: {t}")
                failed_runs += 1
                break

            if done:
                keys_to_print = ['total_ev_served',
                                 'total_energy_charged',
                                 'average_user_satisfaction',
                                 'voltage_up_violation_counter',
                                 'voltage_down_violation_counter',
                                 'saved_grid_energy',
                                 'voltage_violation',
                                 'total_reward'
                                 ]
                print({key: stats[key] for key in keys_to_print})

                new_stats = {key: stats[key] for key in keys_to_print}

                if i == 0:
                    results_df = pd.DataFrame(new_stats, index=[0])
                else:
                    results_df = pd.concat([results_df,
                                            pd.DataFrame(new_stats, index=[0])])

                succesful_runs += 1
                break

        if i % 100 == 0:
            print(
                f' Succesful runs: {succesful_runs} Failed runs: {failed_runs}')

    print(results_df.describe())
    return
    # Solve optimally
    # Power tracker optimizer
    agent = PowerTrackingErrorrMin(replay_path=new_replay_path)
    # # Profit maximization optimizer
    # agent = V2GProfitMaxOracleGB(replay_path=new_replay_path)
    # # Simulate in the gym environment and get the rewards

    env = EV2Gym(config_file=config_file,
                 load_from_replay_path=new_replay_path,
                 verbose=False,
                 save_plots=True,
                 )
    state, _ = env.reset()
    rewards_opt = []

    for t in range(env.simulation_length):
        actions = agent.get_action(env)
        # if verbose:
        #     print(f' OptimalActions: {actions}')

        new_state, reward, done, truncated, stats = env.step(
            actions, visualize=False)  # takes action
        rewards_opt.append(reward)

        # if verbose:
        #     print(f'Reward: {reward} \t Done: {done}')

        if done:
            print(stats)
            break


if __name__ == "__main__":
    # while True:
    eval()
    