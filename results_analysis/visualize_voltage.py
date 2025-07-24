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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, ConnectionPatch

# Add the project root to Python path to import ev2gym
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ev2gym.models.ev2gym_env import EV2Gym


def shorten_algorithm_name(name):
    """Shorten algorithm names to max 10 characters"""
    name_mapping = {
        'Charge As Fast As Possible': 'CAFAP',
        'DO NOTHING': 'No Charging',
        'Random Actions': 'Random',
        'MPC': 'MPC (Oracle)',
        "td3_run_30_K=1_scenario=grid_v2g_profitmax_26092-665267": 'TD3',
        "sac_run_20_K=1_scenario=grid_v2g_profitmax_69380-857910": 'SAC',
        "pi_td3_run_30_K=40_scenario=grid_v2g_profitmax_37423-665267": 'PI-TD3',
        "ppo_run_0_11257_Grid_V2G_profitmaxV2_V2G_grid_state_ModelBasedRL": 'PPO',
        "<class 'ev2gym.baselines.heuristics.ChargeAsFastAsPossible'>": 'CAFAP',
        "<class 'ev2gym.baselines.gurobi_models.v2g_grid_old.V2GProfitMax_Grid_OracleGB'>": 'MPC (Oracle)',
        "<class 'ev2gym.baselines.heuristics.DoNothing'>": 'No Charging'
    }

    # Use mapping if available, otherwise truncate to 10 chars
    if name in name_mapping:
        return name_mapping[name]
    elif len(name) <= 10:
        return name
    else:
        return name[:10]


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
        save_path = os.path.dirname(
            results_path) if results_path else './results_analysis/pes'
        os.makedirs(save_path, exist_ok=True)

    # Create a color cycle (one color per algorithm) using seaborn tab10
    import seaborn as sns
    
    # Define algorithm-specific colors from tab10 palette
    algorithm_colors = {
        'CAFAP': sns.color_palette("tab10")[3],      # Red
        'No Charging': sns.color_palette("tab10")[7], # Gray  
        'TD3': sns.color_palette("tab10")[0],        # Blue
        'PI-TD3': sns.color_palette("tab10")[2],     # Green
        'MPC (Oracle)': sns.color_palette("tab10")[1] # Orange
    }
    
    markers = ['o', 's', 'D', '^', '*', 'v', '<', '>', 'P', 'X']
    linestyles = ['--', ':', '-.', '--', '-.', '--', '-.', '-', ':', '--']

    # If algorithm_names not provided, use the keys from replay.
    if algorithm_names is None:
        algorithm_names = list(replay.keys())
    else:
        algorithm_names = list(algorithm_names)

    print(f'Plotting grid metrics for algorithms: {algorithm_names}')

    # Rename long algorithm names to be up to 10 characters

    # Apply shortening to algorithm names
    algorithm_names = [shorten_algorithm_name(
        name) for name in algorithm_names]
    print(f'Shortened algorithm names: {algorithm_names}')

    # plot only algorithms 0,1,2,5,6
    # selected_algorithms_index = [0, 1, 2, 3, 4, 5, 6]
    selected_algorithms_index = [ 5,6, 0, 3,1]
    # Plot only node 19
    node = 22

    algorithm_names_temp = [algorithm_names[i]
                            for i in selected_algorithms_index]
    algorithm_names = algorithm_names_temp

    # Filter the replay dictionary to keep only the selected algorithms
    # replay = {k: replay[k] for i, k in enumerate(replay.keys()) if i in selected_algorithms_index}

    replay_temp = {}
    for index in selected_algorithms_index:
        key = list(replay.keys())[index]
        replay_temp[key] = replay[key]
    replay = replay_temp

    # Assume all env objects share the same simulation parameters.
    first_key = next(iter(replay))
    env_first = replay[first_key]
    number_of_nodes = env_first.grid.node_num
    sim_starting_date = env_first.sim_starting_date
    sim_date = env_first.sim_date
    timescale = env_first.timescale
    simulation_length = env_first.simulation_length

    # Create date ranges - start from step 20, plot 220 steps total
    start_step = 15
    max_steps = 220
    end_step = start_step + max_steps
    
    # Ensure we don't exceed simulation length
    if end_step > simulation_length:
        end_step = simulation_length
        max_steps = end_step - start_step
    
    date_range = pd.date_range(
        start=sim_starting_date + datetime.timedelta(minutes=timescale * start_step),
        end=sim_starting_date + datetime.timedelta(minutes=timescale * (end_step - 1)),
        freq=f'{timescale}min'
    )
    date_range_print = pd.date_range(
        start=sim_starting_date + datetime.timedelta(minutes=timescale * start_step), 
        end=sim_starting_date + datetime.timedelta(minutes=timescale * (end_step - 1)), 
        periods=10)

    # Determine subplot grid dimensions
    dim_x = int(np.ceil(np.sqrt(number_of_nodes)))
    dim_y = int(np.ceil(number_of_nodes / dim_x))

    # Create the figure for power plot - single subplot for node 19
    plt.figure(figsize=(7, 2.5))
    plt.rcParams['font.family'] = ['serif']

    # For each algorithm, plot its data on this node's subplot with a unique color.
    for index, key in enumerate(replay.keys()):
        print(f'Plotting algorithm: {key} ({algorithm_names[index]})')
        env = replay[key]
        # Choose label and color for this algorithm
        label = algorithm_names[index]
        color = algorithm_colors.get(label, sns.color_palette("tab10")[4])  # Default to purple if not found
        marker = markers[index % len(markers)]
        linestyle = linestyles[index % len(linestyles)]

        # Plot the total active power (node_active_power + node_ev_power) as a step plot - steps 20-239
        plt.step(
            date_range,
            env.node_active_power[node, start_step:end_step] + env.node_ev_power[node, start_step:end_step],
            label=label,
            where='post',
            linewidth=1.5,
            color=color,
            marker=marker,
            markevery=12,
            markersize=5,
            linestyle=linestyle,
            alpha=0.8,
            markerfacecolor='white',
            markeredgewidth=1.5,
            zorder=10
        )

    # add a line at 0
    plt.axhline(0, color='black', linewidth=1, linestyle='--')

    plt.ylabel('Active Power [kW]', fontsize=14)
    # plt.xlabel('Time [h:m]', fontsize=14)
    plt.xlim([sim_starting_date + datetime.timedelta(minutes=timescale * start_step), 
              sim_starting_date + datetime.timedelta(minutes=timescale * (end_step - 1))])
    plt.xticks(date_range_print)
    plt.gca().set_xticklabels(
        [f'{d.hour:02d}:{d.minute:02d}' for d in date_range_print], fontsize=10)
    
    plt.grid(False, which='minor', axis='both', alpha=0.1,
            linewidth=0.5)
    plt.grid(True, which='major', axis='both', alpha=0.5)
    
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#666666')
    plt.minorticks_on()
    
    # plt.legend(fontsize=10)
    #remove legend
    # plt.gca().get_legend().remove()

    plt.tight_layout()
    fig_name = f'{save_path}/grid_power_node.png'
    plt.savefig(fig_name, format='png', dpi=200, bbox_inches='tight')
    # Save as PDF
    fig_name_pdf = f'{save_path}/grid_power_node.pdf'
    plt.savefig(fig_name_pdf, format='pdf', dpi=200, bbox_inches='tight')
    print(f'Saved power plot: {fig_name}')

    plt.close('all')

    # Note: replay and algorithm_names are already filtered from the power plotting section above
    # No need to reload and re-filter the data

    # Use the same filtered data from above
    first_key = next(iter(replay))
    env_first = replay[first_key]
    sim_starting_date = env_first.sim_starting_date
    sim_date = env_first.sim_date
    timescale = env_first.timescale
    simulation_length = env_first.simulation_length

    # Create the full date range used for plotting - start from step 20, plot 220 steps total
    start_step = 15
    max_steps = 220
    end_step = start_step + max_steps
    
    # Ensure we don't exceed simulation length
    if end_step > simulation_length:
        end_step = simulation_length
        max_steps = end_step - start_step
    
    date_range = pd.date_range(
        start=sim_starting_date + datetime.timedelta(minutes=timescale * start_step),
        end=sim_starting_date + datetime.timedelta(minutes=timescale * (end_step - 1)),
        freq=f'{timescale}min'
    )
    date_range_print = pd.date_range(
        start=sim_starting_date + datetime.timedelta(minutes=timescale * start_step), 
        end=sim_starting_date + datetime.timedelta(minutes=timescale * (end_step - 1)), 
        periods=10)

    # Get the default color cycle so that each algorithm gets a unique color.
    # Use the same algorithm_colors mapping defined earlier
    
    # For each algorithm, plot the voltage for this node in a different color.

    # Plot main voltage profiles
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.rcParams['font.family'] = ['serif']

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idx, key in enumerate(replay.keys()):
        env = replay[key]
        label = algorithm_names[idx]
        color = algorithm_colors.get(label, sns.color_palette("tab10")[4])  # Default to purple if not found
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]

        ax.step(
            date_range,
            env.node_voltage[node, start_step:end_step],
            label=label,
            where='post',
            linewidth=1.5,
            color=color,
            marker=marker,
            markevery=12,
            markersize=5,
            linestyle=linestyle,
            alpha=0.8,
            zorder=10,
            markerfacecolor='white',
            markeredgewidth=1.5
        )

    # Voltage limit line
    ax.plot(date_range, [0.95] * len(date_range),
            linestyle='--', color='purple', linewidth=2,
            label='Voltage Limit (0.95 p.u.)', zorder=0)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#666666')    

    # Formatting main axes
    ax.set_ylim(0.94, 1.005)
    ax.set_ylabel(r'|V| [p.u.]', fontsize=14)                  
    # ax.set_xlabel('Time [h:m]', fontsize=14)
    ax.set_xlim([sim_starting_date + datetime.timedelta(minutes=timescale * start_step), 
                 sim_starting_date + datetime.timedelta(minutes=timescale * (end_step - 1))])
    ax.set_xticks(date_range_print)
    ax.set_xticklabels(
        [f'{d.hour:02d}:{d.minute:02d}' for d in date_range_print], fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.minorticks_on()
    
    # ax.legend(fontsize=10)
    leg = ax.legend(
        fontsize=12,
        loc='upper center',           # put it centered above the plot
        bbox_to_anchor=(0.5, 1.28),    # tweak the vertical position
        ncol=3,              # spread entries into columns
        frameon=True,                  # draw a frame around the legend
    )

    # make that frame fancy
    # frame = leg.get_frame()
    # frame.set_facecolor('white')       # background color
    # frame.set_edgecolor('black')       # border color
    # frame.set_linewidth(1.5)           # border thickness
    # frame.set_alpha(0.9)               # slight transparency
    # frame.set_boxstyle('round,pad=0.3')# rounded corners with padding

    
    ax.grid(False, which='minor', axis='both', alpha=0.1,
            linewidth=0.5)
    ax.grid(True, which='major', axis='both', alpha=0.5)

    # --- Add zoomed inset for x-axis steps 50 to 70 (relative to original numbering) ---
    # Convert indices to work with our new range (steps 20-239)
    inset_start_original = 50  # Original step number
    inset_end_original = 70    # Original step number
    
    # Convert to indices in our date_range (which starts from step 20)
    inset_start_idx = max(0, inset_start_original - start_step)
    inset_end_idx = min(len(date_range) - 1, inset_end_original - start_step)
    
    # Ensure we have a valid range
    if inset_start_idx < inset_end_idx:
        x1 = date_range[inset_start_idx]
        x2 = date_range[inset_end_idx]

    # axins = zoomed_inset_axes(ax, zoom=2.5,
    #                           bbox_to_anchor=(10, -1.15), borderpad=2)
    axins = zoomed_inset_axes(
        ax, 2.25,
        bbox_to_anchor=(0.305, 0.99),  # center top (x=0.5), y near top of figure
        bbox_transform=ax.transAxes,
        loc='upper center',
        borderpad=0.5,
    )

    # Only create inset if we have a valid range
    if inset_start_idx < inset_end_idx:
        for idx, key in enumerate(replay.keys()):
            env = replay[key]
            label = algorithm_names[idx]
            color = algorithm_colors.get(label, sns.color_palette("tab10")[4])  # Default to purple if not found
            axins.step(
                date_range[inset_start_idx:inset_end_idx+1],
                env.node_voltage[node, start_step + inset_start_idx:start_step + inset_end_idx+1],
                where='post',
                linewidth=1,
                color=color,
                marker=markers[idx % len(markers)],
                markevery=5,
                markersize=2,
                linestyle=linestyles[idx % len(linestyles)],
                alpha=0.8,
                zorder=10,
                markerfacecolor='white',
                markeredgewidth=1.5
            )

        axins.plot(date_range[inset_start_idx:inset_end_idx+1], [0.95]*(inset_end_idx-inset_start_idx+1), '--', color='grey', linewidth=1.5,zorder=0,)

        # Set inset limits
        axins.set_xlim(x1, x2)
        axins.set_ylim(0.945, 0.952)
        axins.set_xticklabels([])
        # axins.set_yticklabels([f'{y:.4f}' for y in axins.get_yticks()],
                            #   fontsize=8)
        # show 3 y-ticks, the lower limit and the upper limit, and 0.95
        axins.set_yticks([0.945, 0.95, 0.955])
        axins.set_yticklabels([f'{y:.3f}' for y in axins.get_yticks()],
                             fontsize=8)
        axins.grid(True, which='both', alpha=0.3)

        # Draw rectangle on main plot to show zoom area
        mark_inset(ax, axins, loc1=2, loc2=4,
                   fc="none",
                   ec="black",)
        
        rect = axins.patch
        rect.set_facecolor("white")
        rect.set_edgecolor("C3")  # same as connector if you like
        rect.set_linewidth(1.5)
        
        rect.set_path_effects([
            pe.SimplePatchShadow(offset=(4, -4),   # x,y pix offset
                                shadow_rgbFace=(0,0,0), 
                                alpha=0.3),
            pe.Normal()
        ])
        rect.set_edgecolor("green")
    else:
        # Remove the inset if the range is not valid
        axins.remove()
    


    plt.tight_layout()
    fig_name = f'{save_path}/grid_voltage_node.png'
    plt.savefig(fig_name, format='png', dpi=200, bbox_inches='tight')
    # Save as PDF
    fig_name_pdf = f'{save_path}/grid_voltage_node.pdf'
    plt.savefig(fig_name_pdf, format='pdf', dpi=200, bbox_inches='tight')
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
        save_path = os.path.dirname(
            results_path) if results_path else './results_analysis/pes'
        os.makedirs(save_path, exist_ok=True)

    # If algorithm_names not provided, use the keys from replay.
    if algorithm_names is None:
        algorithm_names = list(replay.keys())
    else:
        algorithm_names = list(algorithm_names)

    # Apply shortening to algorithm names
    algorithm_names = [shorten_algorithm_name(
        name) for name in algorithm_names]

    # Filter algorithms (same as grid_metrics: [0, 1, 3, 4])
    # selected_algorithms_index = [0, 1,2, 3, 4, 5, 6]
    selected_algorithms_index = [ 5,6, 0, 3,1]
    algorithm_names_temp = [algorithm_names[i]
                            for i in selected_algorithms_index]
    algorithm_names = algorithm_names_temp

    # Filter the replay dictionary to keep only the selected algorithms
    # replay = {k: replay[k] for i, k in enumerate(replay.keys()) if i in selected_algorithms_index}

    replay_temp = {}
    for index in selected_algorithms_index:
        key = list(replay.keys())[index]
        replay_temp[key] = replay[key]
    replay = replay_temp

    plt.close('all')

    # Create EV SoC plot with same styling as voltage plot
    plt.figure(figsize=(7, 3))
    plt.rcParams['font.family'] = ['serif']

    # Define algorithm-specific colors from tab10 palette (same as in grid_metrics)
    algorithm_colors = {
        'CAFAP': sns.color_palette("tab10")[3],      # Red
        'No Charging': sns.color_palette("tab10")[7], # Gray  
        'TD3': sns.color_palette("tab10")[0],        # Blue
        'PI-TD3': sns.color_palette("tab10")[2],     # Green
        'MPC (Oracle)': sns.color_palette("tab10")[1] # Orange
    }

    # Use same color scheme as voltage plot with algorithm-specific colors
    # algorithm_colors already defined above
    markers = ['o', 's', 'D', '^', '*', 'v', '<', '>', 'P', 'X']
    linestyles = [':', '--', '-.', '-', ':', '--', '-.', '-', ':', '--']

    for index, key in enumerate(replay.keys()):
        env = replay[key]
        label = algorithm_names[index]
        color = algorithm_colors.get(label, sns.color_palette("tab10")[4])  # Default to purple if not found
        marker = markers[index % len(markers)]
        linestyle = linestyles[index % len(linestyles)]

        # Start from step 20, plot 220 steps total
        start_step = 15
        max_steps = 220
        end_step = start_step + max_steps
        
        # Ensure we don't exceed simulation length
        if end_step > env.simulation_length:
            end_step = env.simulation_length
            max_steps = end_step - start_step
            
        date_range = pd.date_range(start=env.sim_starting_date + datetime.timedelta(minutes=env.timescale * start_step),
                                   end=env.sim_starting_date +
                                   datetime.timedelta(minutes=env.timescale * (end_step - 1)),
                                   freq=f'{env.timescale}min')
        date_range_print = pd.date_range(start=env.sim_starting_date + datetime.timedelta(minutes=env.timescale * start_step),
                                         end=env.sim_starting_date +
                                         datetime.timedelta(minutes=env.timescale * (end_step - 1)),
                                         periods=10)

        counter = 1
        charger_to_plot = 23
        for cs in env.charging_stations:
            if counter != charger_to_plot:
                counter += 1
                continue

            df = pd.DataFrame([], index=date_range)

            # Check if port_energy_level exists (depends on lightweight_plots setting)
            if hasattr(env, 'port_energy_level'):
                for port in range(cs.n_ports):
                    df[port] = env.port_energy_level[port, cs.id, start_step:end_step]*100                    
            else:
                print(
                    f"Warning: port_energy_level not available (likely lightweight_plots=True). Skipping EV SoC plot.")
                return

            # Add another row with one datetime step to make the plot look better
            df.loc[df.index[-1] +
                   datetime.timedelta(minutes=env.timescale)] = df.iloc[-1]

            for port in range(cs.n_ports):
                for i, (t_arr, t_dep) in enumerate(env.port_arrival[f'{cs.id}.{port}']):                    
                    t_dep = t_dep + 1
                    if t_dep > len(df):
                        t_dep = len(df)
                    y = df[port].values.T[t_arr-start_step:t_dep-start_step]
                    # fill y with 0 before and after to match the length of df
                    y = np.concatenate(
                        [np.zeros(t_arr), y, np.zeros(len(df) - t_dep)])

                    plt.step(df.index,
                             y,
                             where='post',
                             color=color,
                             marker=marker,
                             markevery=6,
                             markersize=4,
                             linestyle=linestyle,
                             alpha=0.8,
                             linewidth=2.0,
                             markerfacecolor='white',
                             markeredgewidth=1.5,
                             label=algorithm_names[index] if port == 0 and i == 0 else "")

            counter += 1

    # Style the plot similar to voltage plot
    plt.ylabel("EV's SoC [%]", fontsize=14)
    plt.xlabel('Simulation Time', fontsize=14)
    plt.ylim([10, 105])
    plt.xlim([env.sim_starting_date + datetime.timedelta(minutes=env.timescale * start_step), 
              env.sim_starting_date + datetime.timedelta(minutes=env.timescale * (end_step - 1))])
    plt.xticks(date_range_print)
    plt.gca().set_xticklabels(
        [f'{d.hour:02d}:{d.minute:02d}' for d in date_range_print], fontsize=10)
    plt.tick_params(axis='y', labelsize=10)
    
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#666666')
    plt.minorticks_on()

    # Add legend
    # plt.legend(fontsize=10)

    # Add grid
    plt.grid(False, which='minor', axis='both', alpha=0.1,
            linewidth=0.5)
    plt.grid(True, which='major', axis='both', alpha=0.5)

    plt.tight_layout()

    fig_name = f'{save_path}/EV_Energy_Level_single.png'
    plt.savefig(fig_name, format='png', dpi=200, bbox_inches='tight')
    #save pdf
    fig_name_pdf = f'{save_path}/EV_Energy_Level_single.pdf'
    plt.savefig(fig_name_pdf, format='pdf', dpi=200, bbox_inches='tight')
    print(f'Saved EV SoC plot: {fig_name}')


if __name__ == "__main__":
    # Example usage
    
    #original
    name = "eval_150cs_-1tr_v2g_grid_150_300_7_algos_1_exp_2025_07_14_559975"
    
    name = "eval_150cs_-1tr_v2g_grid_150_300_7_algos_1_exp_2025_07_15_120314"
    results_path = f'./results/{name}/plot_results_dict.pkl.gz'
    # read the algorithm names from a file or define them directly from algorithm_names.txt
    name_file = f'./results/{name}/algorithm_names.txt'

    # Set save path for output plots
    save_path = './results_analysis/pes/'

    # Read algorithm names if file exists
    algorithm_names = None
    if os.path.exists(name_file):
        with open(name_file, 'r') as f:
            algorithm_names = [line.strip() for line in f.readlines()]

    # Plot grid metrics (power and voltage)
    plot_grid_metrics(results_path, algorithm_names, save_path)

    # Plot EV SoC with the same styling and algorithms
    # plot_comparable_EV_SoC_single(results_path, save_path, algorithm_names)
