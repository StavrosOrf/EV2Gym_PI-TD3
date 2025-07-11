import pickle
import gzip
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import datetime
import os
import sys

# Add the project root to Python path to import ev2gym
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ev2gym.models.ev2gym_env import EV2Gym


def plot_grid_metrics(results_path,
                      algorithm_names=None,
                      save_path=None):

    plt.close('all')
    # Plot the total power of the CPO
    plt.figure(figsize=(7, 6))

    # Load the env pickle files
    with gzip.open(results_path, 'rb') as f:
        replay = pickle.load(f)

    # Set default save path if not provided
    if save_path is None:
        save_path = os.path.dirname(results_path) if results_path else './results_analysis/pes'
        os.makedirs(save_path, exist_ok=True)


    # Create a color cycle (one color per algorithm)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', 'D', '^', '*', 'v', '<', '>', 'P', 'X']
    linestyles = [':', '--', '-.', '-', ':', '--', '-.', '-', ':', '--']

    # If algorithm_names not provided, use the keys from replay.
    if algorithm_names is None:
        algorithm_names = list(replay.keys())
    else:
        algorithm_names = list(algorithm_names)
        
    print(f'Plotting grid metrics for algorithms: {algorithm_names}')
    
    # Rename long algorithm names to be up to 10 characters
    def shorten_algorithm_name(name):
        """Shorten algorithm names to max 10 characters"""
        name_mapping = {
            'Charge As Fast As Possible': 'CAFAP',
            'DO NOTHING': 'DoNothing',
            'Random Actions': 'Random',
            'Optimal (Offline)': 'Optimal',
            'pi_td3_run_10_K=30_scenario=grid_v2g_profitmax_80448-188243': 'PI-TD3',
            'td3LookaheadCriticReward=3_Critic=True__K=40_td_lambda_horizon=20_seed=9-563606': 'TD3-LA',
            'shac_run_0_K=20_scenario=grid_v2g_profitmax_97200-432696': 'SHAC'
        }
        
        # Use mapping if available, otherwise truncate to 10 chars
        if name in name_mapping:
            return name_mapping[name]
        elif len(name) <= 10:
            return name
        else:
            return name[:10]
    
    # Apply shortening to algorithm names
    algorithm_names = [shorten_algorithm_name(name) for name in algorithm_names]
    print(f'Shortened algorithm names: {algorithm_names}')
    
    #plot only algorithms 0,1,2,5,6
    selected_algorithms_index = [0, 1, 3, 4]
    # Plot only node 19
    node = 20
    
    algorithm_names_temp = [algorithm_names[i] for i in selected_algorithms_index]
    algorithm_names = algorithm_names_temp        

    # Filter the replay dictionary to keep only the selected algorithms
    replay = {k: replay[k] for i, k in enumerate(replay.keys()) if i in selected_algorithms_index}

    # Assume all env objects share the same simulation parameters.
    first_key = next(iter(replay))
    env_first = replay[first_key]
    number_of_nodes = env_first.grid.node_num
    sim_starting_date = env_first.sim_starting_date
    sim_date = env_first.sim_date
    timescale = env_first.timescale
    simulation_length = env_first.simulation_length

    # Create date ranges
    date_range = pd.date_range(
        start=sim_starting_date,
        end=sim_starting_date + datetime.timedelta(minutes=timescale * (simulation_length - 1)),
        freq=f'{timescale}min'
    )
    date_range_print = pd.date_range(start=sim_starting_date, end=sim_date, periods=10)

    # Determine subplot grid dimensions
    dim_x = int(np.ceil(np.sqrt(number_of_nodes)))
    dim_y = int(np.ceil(number_of_nodes / dim_x))

    # Create the figure for power plot - single subplot for node 19
    plt.figure(figsize=(7, 4))
    plt.rcParams['font.family'] = ['serif']


    
    # For each algorithm, plot its data on this node's subplot with a unique color.
    for index, key in enumerate(replay.keys()):
        env = replay[key]
        # Choose label and color for this algorithm
        label = algorithm_names[index]  # if names were provided, otherwise keys are used.
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        linestyle = linestyles[index % len(linestyles)]
        
        # Plot the total active power (node_active_power + node_ev_power) as a step plot
        plt.step(
            date_range,
            env.node_active_power[node, :] + env.node_ev_power[node, :],
            label=label,
            where='post',
            linewidth=1,
            color=color,
            marker=marker,
            markevery=20,
            markersize=3,
            linestyle=linestyle,
            alpha=0.8
        )
        
    #add a line at 0
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    
    plt.ylabel('Power (kW)', fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.xlim([sim_starting_date, sim_date])
    plt.xticks(date_range_print)
    plt.gca().set_xticklabels([f'{d.hour:02d}:{d.minute:02d}' for d in date_range_print], fontsize=10)
    plt.grid(True, which='minor', axis='both', alpha=0.3)
    plt.grid(True, which='major', axis='both', alpha=0.5)
    plt.legend(fontsize=10)

    plt.tight_layout()
    fig_name = f'{save_path}/grid_power_node.png'
    plt.savefig(fig_name, format='png', dpi=200, bbox_inches='tight')
    print(f'Saved power plot: {fig_name}')

    plt.close('all')
    plt.figure(figsize=(7, 4))
    plt.rcParams['font.family'] = ['serif']

    # Note: replay and algorithm_names are already filtered from the power plotting section above
    # No need to reload and re-filter the data

    # Use the same filtered data from above
    first_key = next(iter(replay))
    env_first = replay[first_key]
    sim_starting_date = env_first.sim_starting_date
    sim_date = env_first.sim_date
    timescale = env_first.timescale
    simulation_length = env_first.simulation_length

    # Create the full date range used for plotting
    date_range = pd.date_range(
        start=sim_starting_date,
        end=sim_starting_date + datetime.timedelta(minutes=timescale * (simulation_length - 1)),
        freq=f'{timescale}min'
    )
    date_range_print = pd.date_range(start=sim_starting_date, end=sim_date, periods=10)

    # Get the default color cycle so that each algorithm gets a unique color.
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', 'D', '^', '*', 'v', '<', '>', 'P', 'X']
    linestyles = [':', '--', '-.', '-', ':', '--', '-.', '-', ':', '--']
    
    # For each algorithm, plot the voltage for this node in a different color.
    for index, key in enumerate(replay.keys()):
        env = replay[key]
        label = algorithm_names[index]
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        linestyle = linestyles[index % len(linestyles)]
        
        plt.step(
            date_range,
            env.node_voltage[node, :],
            label=label,
            where='post',
            linewidth=1,
            color=color,
            marker=marker,
            markevery=20,
            markersize=3,
            linestyle=linestyle,
            alpha=0.8
        )
    
    # Plot voltage limits (same for all algorithms)
    plt.plot(date_range, [0.95] * len(date_range), 'r--', linewidth=2)
    # plt.plot(date_range, [1.05] * len(date_range), 'r--', linewidth=2)

    # plt.title(f'Node {node} - Voltage Profile', fontsize=14, fontweight='bold')
    plt.ylabel('V(pu)', fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.xlim([sim_starting_date, sim_date])
    plt.xticks(date_range_print)
    plt.gca().set_xticklabels([f'{d.hour:02d}:{d.minute:02d}' for d in date_range_print], fontsize=10)
    plt.tick_params(axis='y', labelsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, which='minor', axis='both', alpha=0.3)
    plt.grid(True, which='major', axis='both', alpha=0.5)

    plt.tight_layout()
    fig_name = f'{save_path}/grid_voltage_node.png'
    plt.savefig(fig_name, format='png', dpi=200, bbox_inches='tight')
    print(f'Saved voltage plot: {fig_name}')


def plot_comparable_EV_SoC_single(results_path,
                                  save_path=None,
                                  algorithm_names=None):
    '''
    This function is used to plot the SoC of the EVs in the same plot
    '''

    with gzip.open(results_path, 'rb') as f:
        replay = pickle.load(f)

    # Set default save path if not provided
    if save_path is None:
        save_path = os.path.dirname(results_path) if results_path else './results_analysis/pes'
        os.makedirs(save_path, exist_ok=True)

    # If algorithm_names not provided, use the keys from replay.
    if algorithm_names is None:
        algorithm_names = list(replay.keys())
    else:
        algorithm_names = list(algorithm_names)

    # Rename long algorithm names to be up to 10 characters (same as grid_metrics)
    def shorten_algorithm_name(name):
        """Shorten algorithm names to max 10 characters"""
        name_mapping = {
            'Charge As Fast As Possible': 'CAFAP',
            'DO NOTHING': 'DoNothing',
            'Random Actions': 'Random',
            'Optimal (Offline)': 'Optimal',
            'pi_td3_run_10_K=30_scenario=grid_v2g_profitmax_80448-188243': 'PI-TD3',
            'td3LookaheadCriticReward=3_Critic=True__K=40_td_lambda_horizon=20_seed=9-563606': 'TD3-LA',
            'shac_run_0_K=20_scenario=grid_v2g_profitmax_97200-432696': 'SHAC'
        }
        
        if name in name_mapping:
            return name_mapping[name]
        elif len(name) <= 10:
            return name
        else:
            return name[:10]
    
    # Apply shortening to algorithm names
    algorithm_names = [shorten_algorithm_name(name) for name in algorithm_names]

    # Filter algorithms (same as grid_metrics: [0, 1, 3, 4])
    selected_algorithms_index = [0, 1, 3, 4]
    algorithm_names_temp = [algorithm_names[i] for i in selected_algorithms_index]
    algorithm_names = algorithm_names_temp        

    # Filter the replay dictionary to keep only the selected algorithms
    replay = {k: replay[k] for i, k in enumerate(replay.keys()) if i in selected_algorithms_index}

    plt.close('all')
    
    # Create EV SoC plot with same styling as voltage plot
    plt.figure(figsize=(7, 4))
    plt.rcParams['font.family'] = ['serif']
    
    # Use same color scheme as voltage plot
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['o', 's', 'D', '^', '*', 'v', '<', '>', 'P', 'X']
    linestyles = [':', '--', '-.', '-', ':', '--', '-.', '-', ':', '--']

    for index, key in enumerate(replay.keys()):
        env = replay[key]
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        linestyle = linestyles[index % len(linestyles)]

        date_range = pd.date_range(start=env.sim_starting_date,
                                   end=env.sim_starting_date +
                                   (env.simulation_length - 1) *
                                   datetime.timedelta(minutes=env.timescale),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date,
                                         end=env.sim_date,
                                         periods=10)

        counter = 1
        charger_to_plot = 7
        for cs in env.charging_stations:
            if counter != charger_to_plot:
                counter += 1
                continue

            df = pd.DataFrame([], index=date_range)

            # Check if port_energy_level exists (depends on lightweight_plots setting)
            if hasattr(env, 'port_energy_level'):
                for port in range(cs.n_ports):
                    df[port] = env.port_energy_level[port, cs.id, :]
            else:
                print(f"Warning: port_energy_level not available (likely lightweight_plots=True). Skipping EV SoC plot.")
                return

            # Add another row with one datetime step to make the plot look better
            df.loc[df.index[-1] +
                   datetime.timedelta(minutes=env.timescale)] = df.iloc[-1]

            for port in range(cs.n_ports):
                for i, (t_arr, t_dep) in enumerate(env.port_arrival[f'{cs.id}.{port}']):
                    t_dep = t_dep + 1
                    if t_dep > len(df):
                        t_dep = len(df)
                    y = df[port].values.T[t_arr:t_dep]
                    # fill y with 0 before and after to match the length of df
                    y = np.concatenate(
                        [np.zeros(t_arr), y, np.zeros(len(df) - t_dep)])

                    plt.step(df.index,
                            y,
                            where='post',
                            color=color,
                            marker=marker,
                            markevery=25,
                            markersize=6,
                            linestyle=linestyle,
                            alpha=0.8,
                            linewidth=1.5,
                            markerfacecolor='white',
                            markeredgewidth=1.5,
                            label=algorithm_names[index] if port == 0 and i == 0 else "")

            counter += 1

    # Style the plot similar to voltage plot
    plt.ylabel('SoC', fontsize=14)
    plt.xlabel('Time', fontsize=14)
    plt.ylim([0.1, 1.09])
    plt.xlim([env.sim_starting_date, env.sim_date])
    plt.xticks(date_range_print)
    plt.gca().set_xticklabels([f'{d.hour:02d}:{d.minute:02d}' for d in date_range_print], fontsize=10)
    plt.tick_params(axis='y', labelsize=10)
    
    # Add legend
    plt.legend(fontsize=10)
    
    # Add grid
    plt.grid(True, which='minor', axis='both', alpha=0.3)
    plt.grid(True, which='major', axis='both', alpha=0.5)

    plt.tight_layout()

    fig_name = f'{save_path}/EV_Energy_Level_single.png'
    plt.savefig(fig_name, format='png', dpi=200, bbox_inches='tight')
    print(f'Saved EV SoC plot: {fig_name}')

if __name__ == "__main__":
    # Example usage
    results_path = './results/eval_150cs_-1tr_v2g_grid_150_300_7_algos_1_exp_2025_07_11_465029/plot_results_dict.pkl.gz'    
    # read the algorithm names from a file or define them directly from algorithm_names.txt
    name_file = './results/eval_150cs_-1tr_v2g_grid_150_300_7_algos_1_exp_2025_07_11_465029/algorithm_names.txt'

    # Set save path for output plots
    save_path = './results_analysis/pes/'
    
    # Read algorithm names if file exists
    algorithm_names = None
    if os.path.exists(name_file):
        with open(name_file, 'r') as f:
            algorithm_names = [line.strip() for line in f.readlines()]

    # Plot grid metrics (power and voltage)
    # plot_grid_metrics(results_path, algorithm_names, save_path)
    
    # Plot EV SoC with the same styling and algorithms
    plot_comparable_EV_SoC_single(results_path, save_path, algorithm_names)
