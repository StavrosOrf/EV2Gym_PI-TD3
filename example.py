"""
This script is used to evaluate the performance of the ev2gym environment.
"""
from ev2gym.models.ev2gym_env import EV2Gym

from ev2gym.baselines.heuristics import RoundRobin, RandomAgent, ChargeAsFastAsPossible

from agent.state import V2G_grid_state, V2G_grid_state_ModelBasedRL
from agent.reward import V2G_grid_reward, V2G_grid_simple_reward
from agent.loss import VoltageViolationLoss, V2G_Grid_StateTransition

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import pandas as pd
import torch
import time


def eval():
    """
    Runs an evaluation of the ev2gym environment.
    """

    replay_path = "./replay/replay_sim_2025_02_22_360719.pkl"

    replay_path = None

    config_file = "./config_files/v2g_grid_150.yaml"
    config_file = "./config_files/v2g_grid_3.yaml"
    seed = 0

    env = EV2Gym(config_file=config_file,
                 load_from_replay_path=replay_path,
                 verbose=False,
                 #  seed=seed,
                 save_replay=False,
                 save_plots=False,
                 state_function=V2G_grid_state_ModelBasedRL,
                 reward_function=V2G_grid_simple_reward,
                 )

    print(env.action_space)
    print(env.observation_space)
    new_replay_path = f"replay/replay_{env.sim_name}.pkl"

    # agent = ChargeAsFastAsPossible()
    agent = RandomAgent()
    # agent = ChargeAsFastAsPossibleToDesiredCapacity()

    max_cs_power = env.charging_stations[0].get_max_power()
    min_cs_power = env.charging_stations[0].get_min_power()

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
                                   verbose=False,
                                   )

    state_transition = V2G_Grid_StateTransition(verbose=False,
                                                device=device,
                                                num_buses=env.grid.net.nb
                                                )

    succesful_runs = 0
    failed_runs = 0

    results_df = None
    total_timer = 0

    for i in range(100):
        state, _ = env.reset()
        for t in range(env.simulation_length):
            actions = agent.get_action(env)

            new_state, reward, done, truncated, stats = env.step(
                actions)  # takes action
            # print(
            #     "============================================================================")
            predicted_state = state_transition(state=torch.tensor(state, device=device).reshape(1, -1),
                                               new_state=torch.tensor(
                                                   new_state, device=device).reshape(1, -1),
                                               action=torch.tensor(
                                                   actions, device=device).reshape(1, -1),
                                               )

            predicted_state = predicted_state.cpu().detach().numpy().reshape(-1)
            # print(f'Prev State: {state}')
            # print(f'Predicted State: {predicted_state}')
            # print(f'New State: {new_state}')
            # print(f'diff: {np.abs(predicted_state - new_state).mean()}')
            if np.abs(predicted_state - new_state).mean() > 0.001:
                # make noise beep
                print(f'diff: {np.abs(predicted_state - new_state).mean()}')
                input('Error in state transition')
                

            # print("============================================================================")
            timer = time.time()
            loss, v = loss_fn.calc_v(action=torch.tensor(actions, device=device).reshape(1, -1),
                                         state=torch.tensor(state, device=device).reshape(1, -1))
            total_timer += time.time() - timer

            v_m = env.node_voltage[1:, t]
            # print(f'\n \n')
            print(f'V real: {v_m}')
            print(f'V pred: {v}')
            print(f'v_loss {np.abs(v - v_m).mean()}')
            if np.abs(v - v_m).mean() > 0.001:
                print(f'Error in voltage calculation')
                
            loss_v = np.minimum(np.zeros_like(v_m), 0.05 - np.abs(1-v_m))

            print(f'Loss V: {loss_v}')
            reward_loss = np.abs(reward - loss.cpu().detach().numpy())
            print(f'Reward Loss: {reward_loss} | Reward: {reward} | Loss: {loss} | Loss V sum: {1000*loss_v.sum()}')

            if reward_loss > 0.001:
                print(f'Error in reward calculation')                

            state = new_state

            if done and truncated:
                failed_runs += 1
                break

            if done:
                keys_to_print = ['total_ev_served',
                                 'total_energy_charged',
                                 'total_profits',
                                 'average_user_satisfaction',
                                 #  'saved_grid_energy',
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
