# This script reads the replay files and evaluates the performance.

import gymnasium as gym
# from state_action_eda import AnalysisReplayBuffer
from ev2gym.visuals.evaluator_plot import plot_comparable_EV_SoC_single, plot_prices
from ev2gym.visuals.evaluator_plot import plot_total_power_V2G, plot_actual_power_vs_setpoint
from ev2gym.visuals.evaluator_plot import plot_total_power, plot_comparable_EV_SoC, plot_grid_metrics
from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin

from algorithms.SAC.sac import SAC
from algorithms.SAC.pi_SAC import PI_SAC
from algorithms.ppo import PPO
from algorithms.TD3 import TD3
from algorithms.pi_TD3 import PI_TD3
from algorithms.pi_DDPG import PI_DDPG
from algorithms.pi_ppo import PhysicsInformedPPO
from algorithms.shac import SHAC
from algorithms.shac_onpolicy import SHAC_OnPolicy
from algorithms.reinforce import Reinforce
from algorithms.sapo import SAPO
from algorithms.sapo_onpolicy import SAPO_OnPolicy

# from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO
# from stable_baselines3 import PPO, A2C, DDPG, SAC
# from stable_baselines3 import TD3 as TD3_SB3
from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle, V2GProfitMaxLoadsOracle
from ev2gym.baselines.mpc.eMPC_v2 import eMPC_V2G_v2
from ev2gym.baselines.mpc.eMPC import eMPC_V2G, eMPC_G2V
from ev2gym.baselines.mpc.ocmf_mpc import OCMF_V2G, OCMF_G2V
from ev2gym.baselines.heuristics import RandomAgent, DoNothing, ChargeAsFastAsPossible

from agent.state import V2G_grid_state_ModelBasedRL
from agent.reward import Grid_V2G_profitmaxV2

from ev2gym.models.ev2gym_env import EV2Gym
import yaml
import os
import pickle
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import datetime
import time
import random
import gzip

import warnings

# Suppress all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# set seeds
seed = 9
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


def evaluator():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ############# Simulation Parameters #################
    n_test_cycles = 10
    SAVE_REPLAY_BUFFER = False
    SAVE_EV_PROFILES = False

    config_file = "./config_files/v2g_grid_150_300.yaml"
    # config_file = "./config_files/v2g_grid_150.yaml"
    # config_file = "./config_files/v2g_grid_50.yaml"

    state_function_Normal = V2G_grid_state_ModelBasedRL
    reward_function = Grid_V2G_profitmaxV2

    # Algorithms to compare:
    # Use algorithm name or the saved RL model path as string
    algorithms = [        
        ChargeAsFastAsPossible,
        DoNothing,
        RandomAgent,
        V2GProfitMaxOracleGB,
        'pi_td3_run_10_K=30_scenario=grid_v2g_profitmax_80448-188243',
        'td3LookaheadCriticReward=3_Critic=True__K=40_td_lambda_horizon=20_seed=9-563606',
        'shac_run_0_K=20_scenario=grid_v2g_profitmax_97200-432696',
        # 'mb_traj_run_70_K=70_scenario=grid_v2g_profitmax_47956-289507',
        # 'TD3_run_50_K=1_scenario=grid_v2g_profitmax_36803-857049',
    ]

    # create a AnalysisReplayBuffer object for each algorithm

    env = EV2Gym(config_file=config_file,
                 generate_rnd_game=True,
                 state_function=state_function_Normal,
                 reward_function=reward_function,
                 )

    if SAVE_REPLAY_BUFFER:
        replay_buffers = {}
        for algorithm in algorithms:

            replay_buffers[algorithm] = AnalysisReplayBuffer(state_dim=env.observation_space.shape[0],
                                                             action_dim=env.action_space.shape[0],
                                                             max_size=int(1e4))

    #####################################################

    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)

    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    timescale = config["timescale"]
    simulation_length = config["simulation_length"]
    scenario = config_file.split("/")[-1].split(".")[0]
    
    eval_replay_path = f'./replay/{config["number_of_charging_stations"]}cs_{scenario}/'

    print(f'Looking for replay files in {eval_replay_path}')
    try:
        eval_replay_files = [f for f in os.listdir(
            eval_replay_path) if os.path.isfile(os.path.join(eval_replay_path, f))]

        print(
            f'Found {len(eval_replay_files)} replay files in {eval_replay_path}')
        if n_test_cycles > len(eval_replay_files):
            # n_test_cycles = len(eval_replay_files)
            replays_exist = False
        else:

            replay_to_print = 1
            replay_to_print = min(replay_to_print, len(eval_replay_files)-1)
            replays_exist = True

    except:
        n_test_cycles = n_test_cycles
        replays_exist = False

    print(f'Number of test cycles: {n_test_cycles}')

    if SAVE_EV_PROFILES:
        ev_profiles = []

    def generate_replay(evaluation_name):
        env = EV2Gym(config_file=config_file,
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

        if SAVE_EV_PROFILES:
            ev_profiles.append(env.EVs_profiles)
        return replay_path

    evaluation_name = f'eval_{number_of_charging_stations}cs_{n_transformers}tr_{scenario}_{len(algorithms)}_algos' +\
        f'_{n_test_cycles}_exp_' +\
        f'{datetime.datetime.now().strftime("%Y_%m_%d_%f")}'

    # make a directory for the evaluation
    save_path = f'./results/{evaluation_name}/'
    os.makedirs(save_path, exist_ok=True)
    os.system(f'cp {config_file} {save_path}')

    if not replays_exist:
        eval_replay_files = [generate_replay(
            eval_replay_path) for _ in range(n_test_cycles)]

    # save the list of EV profiles to a pickle file
    if SAVE_EV_PROFILES:
        with open(save_path + 'ev_profiles.pkl', 'wb') as f:
            print(f'Saving EV profiles to {save_path}ev_profiles.pkl')
            pickle.dump(ev_profiles, f)

        exit()

    plot_results_dict = {}
    counter = 0

    for i_a, algorithm in enumerate(algorithms):
        print(' +------- Evaluating', algorithm, " -------+")
        for k in range(n_test_cycles):
            print(f' Test cycle {k+1}/{n_test_cycles} -- {algorithm}')
            counter += 1
            h = -1

            if replays_exist:
                replay_path = eval_replay_path + eval_replay_files[k]
            else:
                replay_path = eval_replay_files[k]

            if type(algorithm) == str:
                if "GNN" in algorithm:
                    # state_function = state_function_GNN
                    pass
                else:
                    state_function = state_function_Normal
            else:
                state_function = state_function_Normal

            env = EV2Gym(config_file=config_file,
                         load_from_replay_path=replay_path,
                         state_function=state_function,
                         reward_function=reward_function,
                         )

            # initialize the timer
            timer = time.time()
            state, _ = env.reset()
            # try:
            if type(algorithm) == str:
                if algorithm.split('_')[0] in ['OCMF', 'eMPC']:
                    h = int(algorithm.split('_')[2])
                    algorithm = algorithm.split(
                        '_')[0] + '_' + algorithm.split('_')[1]
                    print(
                        f'Algorithm: {algorithm} with control horizon {h}')
                    if algorithm == 'OCMF_V2G':
                        model = OCMF_V2G(
                            env=env, control_horizon=h)
                        algorithm = OCMF_V2G
                    elif algorithm == 'OCMF_G2V':
                        model = OCMF_G2V(
                            env=env, control_horizon=h)
                        algorithm = OCMF_G2V
                    elif algorithm == 'eMPC_V2G':
                        model = eMPC_V2G(
                            env=env, control_horizon=h)
                        algorithm = eMPC_V2G
                    elif algorithm == 'eMPC_G2V':
                        model = eMPC_G2V(
                            env=env, control_horizon=h)
                        algorithm = eMPC_G2V

                    algorithm_name = algorithm.__name__

                elif any(algo in algorithm for algo in ['a2c', 'ddpg', 'tqc', 'trpo', 'ars', 'rppo']):

                    gym.envs.register(id='evs-v0', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                                      kwargs={'config_file': config_file,
                                              'generate_rnd_game': True,
                                              'state_function': state_function_Normal,
                                              'reward_function': reward_function,
                                              'load_from_replay_path': replay_path,
                                              })
                    env = gym.make('evs-v0')

                    load_path = f'./saved_models/{algorithm}/best_model.zip'

                    # initialize the timer
                    timer = time.time()
                    algorithm_name = algorithm.split('_')[0]

                    if 'rppo' in algorithm:
                        sb3_algo = RecurrentPPO
                    elif 'ppo' in algorithm:
                        sb3_algo = PPO
                    elif 'a2c' in algorithm:
                        sb3_algo = A2C
                    elif 'ddpg' in algorithm:
                        sb3_algo = DDPG
                    elif 'tqc' in algorithm:
                        sb3_algo = TQC
                    elif 'trpo' in algorithm:
                        sb3_algo = TRPO
                    elif 'td3' in algorithm:
                        print("Loading TD3-SB3 model")
                        sb3_algo = TD3_SB3

                    else:
                        exit()

                    model = sb3_algo.load(load_path,
                                          env,
                                          device=device
                                          )
                    # set replay buffer to None

                    if 'tqc' in algorithm or 'ddpg' in algorithm:
                        model.replay_buffer = model.replay_buffer.__class__(1,
                                                                            model.observation_space,
                                                                            model.action_space,
                                                                            device=model.device,
                                                                            optimize_memory_usage=model.replay_buffer.optimize_memory_usage)

                    env = model.get_env()
                    state = env.reset()

                elif "SAC" in algorithm:
                    # remove _SL from the algorithm name

                    algorithm_path = algorithm
                    load_model_path = f'./eval_models/{algorithm_path}/'
                    # Load kwargs.yaml as a dictionary
                    with open(f'{load_model_path}kwargs.yaml') as file:
                        kwargs = yaml.load(
                            file, Loader=yaml.FullLoader)

                    state_dim = env.observation_space.shape[0]
                    model = SAC(num_inputs=state_dim,
                                action_space=env.action_space,
                                args=kwargs)

                    algorithm_name = "SAC"
                    model.load(ckpt_path=f'{load_model_path}model.best',
                               evaluate=True)

                    if k == 0:
                        actor_model = model.policy
                        model_parameters = filter(
                            lambda p: p.requires_grad, actor_model.parameters())
                        params = sum([np.prod(p.size())
                                      for p in model_parameters])
                        print(
                            f'Actor model has {params} trainable parameters')

                elif "pi_td3" in algorithm:
                    algorithm_path = algorithm
                    load_model_path = f'./eval_models/{algorithm_path}/'
                    with open(f'{load_model_path}kwargs.yaml') as file:
                        kwargs = yaml.load(
                            file, Loader=yaml.UnsafeLoader)

                    # else:
                    print("Loading TD3 model")
                    model = PI_TD3(**kwargs)
                    algorithm_name = "PI_TD3"
                    model.load(
                        filename=f'{load_model_path}model.best')

                    if k == 0:
                        actor_model = model.actor
                        model_parameters = filter(
                            lambda p: p.requires_grad, actor_model.parameters())
                        params = sum([np.prod(p.size())
                                      for p in model_parameters])
                        print(
                            f'Actor model has {params} trainable parameters')

                elif "td3" in algorithm:
                    algorithm_path = algorithm
                    load_model_path = f'./eval_models/{algorithm_path}/'
                    with open(f'{load_model_path}kwargs.yaml') as file:
                        kwargs = yaml.load(
                            file, Loader=yaml.UnsafeLoader)

                    print("Loading TD3 model")
                    model = TD3(**kwargs)
                    algorithm_name = "TD3"
                    model.load(
                        filename=f'{load_model_path}model.best')

                    if k == 0:
                        actor_model = model.actor
                        model_parameters = filter(
                            lambda p: p.requires_grad, actor_model.parameters())
                        params = sum([np.prod(p.size())
                                      for p in model_parameters])
                        print(
                            f'Actor model has {params} trainable parameters')

                elif "shac" in algorithm:
                    algorithm_path = algorithm
                    load_model_path = f'./eval_models/{algorithm_path}/'
                    with open(f'{load_model_path}kwargs.yaml') as file:
                        kwargs = yaml.load(
                            file, Loader=yaml.UnsafeLoader)

                    print("Loading SHAC model")
                    model = SHAC(**kwargs)
                    algorithm_name = "SHAC"
                    model.load(
                        filename=f'{load_model_path}model.best')

                else:
                    raise ValueError(
                        f'Unknown algorithm {algorithm}')
            else:
                model = algorithm(env=env,
                                  replay_path=replay_path,
                                  verbose=False)
                algorithm_name = algorithm.__name__

            rewards = []

            for i in range(simulation_length):

                if type(algorithm) == str:
                    if any(algo in algorithm for algo in ['a2c', 'ddpg', 'tqc', 'trpo', 'ars', 'rppo']):
                        action, _ = model.predict(
                            state, deterministic=True)
                        state, reward, done, stats = env.step(action)

                        if i == simulation_length - 2:
                            saved_env = deepcopy(
                                env.get_attr('env')[0])

                        stats = stats[0]
                    # elif "SAC" in algorithm or "TD3" in algorithm or \
                    #         "mb_traj" in algorithm or "LSTM-ModelBasedRL" in algorithm:
                    else:
                        action = model.select_action(state,
                                                     return_mapped_action=True)

                        state, reward, done, _, stats = env.step(
                            action)

                    # else:
                    #     raise ValueError(
                    #         f'Unknown algorithm {algorithm}')

                else:

                    action = model.get_action(env=env)
                    new_state, reward, done, _, stats = env.step(
                        action)

                rewards.append(reward)

            if done:

                results_i = pd.DataFrame({'run': k,
                                          'Algorithm': algorithm_name,
                                          'algorithm_version': algorithm,
                                          'control_horizon': h,
                                          'discharge_price_factor': config['discharge_price_factor'],
                                          'total_ev_served': stats['total_ev_served'],
                                          'total_profits': stats['total_profits'],
                                          'total_energy_charged': stats['total_energy_charged'],
                                          'total_energy_discharged': stats['total_energy_discharged'],
                                          'average_user_satisfaction': stats['average_user_satisfaction'],
                                          'power_tracker_violation': stats['power_tracker_violation'],
                                          'tracking_error': stats['tracking_error'],
                                          'energy_tracking_error': stats['energy_tracking_error'],
                                          'energy_user_satisfaction': stats['energy_user_satisfaction'],
                                          'min_energy_user_satisfaction': stats['min_energy_user_satisfaction'],
                                          'std_energy_user_satisfaction': stats['std_energy_user_satisfaction'],
                                          'total_transformer_overload': stats['total_transformer_overload'],
                                          'battery_degradation': stats['battery_degradation'],
                                          'battery_degradation_calendar': stats['battery_degradation_calendar'],
                                          'battery_degradation_cycling': stats['battery_degradation_cycling'],
                                          'voltage_violation': stats['voltage_violation'],
                                          'voltage_violation_counter': stats['voltage_violation_counter'],
                                          'voltage_violation_counter_per_step': stats['voltage_violation_counter_per_step'],
                                          'total_reward': sum(rewards),
                                          'time': time.time() - timer,
                                          }, index=[counter])

                # change name of key to algorithm_name
                if SAVE_REPLAY_BUFFER:
                    if k == n_test_cycles - 1:
                        replay_buffers[algorithm_name] = replay_buffers.pop(
                            algorithm)

                if counter == 1:
                    results = results_i
                else:
                    results = pd.concat([results, results_i])

                # if algorithm in [PPO, A2C, DDPG, SAC, TD3, TQC, TRPO, ARS, RecurrentPPO]:
                #     env = saved_env

                if k == 0:
                    plot_results_dict[str(
                        algorithm)] = deepcopy(env)

    # save the replay buffers to a pickle file
    if SAVE_REPLAY_BUFFER:
        with open(save_path + 'replay_buffers.pkl', 'wb') as f:
            pickle.dump(replay_buffers, f)

    # save the plot_results_dict to a pickle file
    # with open(save_path + 'plot_results_dict.pkl', 'wb') as f:
    #     pickle.dump(plot_results_dict, f)

        # replace some algorithm_version to other names:
    # change from PowerTrackingErrorrMin -> PowerTrackingError

    # print unique algorithm versions

    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.heuristics.ChargeAsFastAsPossible'>", 'ChargeAsFastAsPossible')
    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.heuristics.RoundRobin_GF_off_allowed'>", 'RoundRobin_GF_off_allowed')
    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.heuristics.RoundRobin_GF'>", 'RoundRobin_GF')
    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.heuristics.RoundRobin'>", 'RoundRobin')
    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.heuristics.DoNothing'>", 'DoNothing')
    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.heuristics.RandomAgent'>", 'RandomAgent')
    results['algorithm_version'] = results['algorithm_version'].astype(str).replace(
        "<class 'ev2gym.baselines.gurobi_models.tracking_error.PowerTrackingErrorrMin'>",
        'Oracle'
    )
    print(results['algorithm_version'].unique())

    # save the results to a csv file
    results.to_csv(save_path + 'data.csv')

    # drop_columns = ['algorithm_version']
    drop_columns = ['Algorithm']

    results = results.drop(columns=drop_columns)

    results_grouped = results.groupby(
        'algorithm_version',).agg(['mean', 'std'])

    # sort results by tracking error
    results_grouped = results_grouped.sort_values(
        by=('voltage_violation', 'mean'), ascending=False)

    print(results_grouped[[
        'total_profits',
        'total_ev_served',
        'average_user_satisfaction',
        'total_energy_charged',
        'total_energy_discharged',
        'total_reward',
        'voltage_violation',
        'voltage_violation_counter',
        'voltage_violation_counter_per_step',
    ]])

    with gzip.open(save_path + 'plot_results_dict.pkl.gz', 'wb') as f:
        pickle.dump(plot_results_dict, f)

    algorithm_names = []
    for algorithm in algorithms:
        # if class has attribute .name, use it
        if hasattr(algorithm, 'algo_name'):
            algorithm_names.append(algorithm.algo_name)
        elif type(algorithm) == str:
            if "GNN" in algorithm:
                # algorithm_names.append('RL')
                algorithm_names.append(algorithm.split(
                    '_')[0] + '_' + algorithm.split('_')[1])

            else:
                # algorithm_names.append(algorithm.split('_')[0])
                algorithm_names.append(algorithm)
        else:
            algorithm_names.append(algorithm.__name__)

    # save algorithm names to a txt file
    with open(save_path + 'algorithm_names.txt', 'w') as f:
        for item in algorithm_names:
            f.write("%s\n" % item)

    print(f'Plottting results at {save_path}')

    plot_grid_metrics(results_path=save_path + 'plot_results_dict.pkl.gz',
                      save_path=save_path,
                      algorithm_names=algorithm_names)

    # plot_total_power(results_path=save_path + 'plot_results_dict.pkl',
    #                  save_path=save_path,
    #                  algorithm_names=algorithm_names)

    # plot_comparable_EV_SoC(results_path=save_path + 'plot_results_dict.pkl',
    #                        save_path=save_path,
    #                        algorithm_names=algorithm_names)

    # plot_actual_power_vs_setpoint(results_path=save_path + 'plot_results_dict.pkl',
    #                               save_path=save_path,
    #                               algorithm_names=algorithm_names)

    # plot_total_power_V2G(results_path=save_path + 'plot_results_dict.pkl',
    #                      save_path=save_path,
    #                      algorithm_names=algorithm_names)

    # plot_comparable_EV_SoC_single(results_path=save_path + 'plot_results_dict.pkl',
    #                               save_path=save_path,
    #                               algorithm_names=algorithm_names)

    # plot_prices(results_path=save_path + 'plot_results_dict.pkl',
    #             save_path=save_path,
    #             algorithm_names=algorithm_names)


if __name__ == "__main__":
    evaluator()
