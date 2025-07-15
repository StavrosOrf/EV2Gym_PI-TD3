import seaborn as sns
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data_paths = [
    './results/eval_150cs_-1tr_v2g_grid_150_300_l=05_7_algos_20_exp_2025_07_15_083495/data.csv',
    './results/eval_150cs_-1tr_v2g_grid_150_300_l=075_7_algos_20_exp_2025_07_15_846801/data.csv',
    './results/eval_150cs_-1tr_v2g_grid_150_300_l=085_7_algos_20_exp_2025_07_15_908817/data.csv',
    './results/eval_150cs_-1tr_v2g_grid_150_300_l=095_7_algos_20_exp_2025_07_15_397420/data.csv',
    './results/eval_150cs_-1tr_v2g_grid_150_300_7_algos_20_exp_2025_07_15_618438/data.csv',
    './results/eval_150cs_-1tr_v2g_grid_150_300_l=105_7_algos_20_exp_2025_07_15_171569/data.csv',
    './results/eval_150cs_-1tr_v2g_grid_150_300_l=115_7_algos_20_exp_2025_07_15_856217/data.csv',
    './results/eval_150cs_-1tr_v2g_grid_150_300_l=125_7_algos_20_exp_2025_07_15_224048/data.csv',
    # './results/eval_150cs_-1tr_v2g_grid_150_300_l=15_7_algos_2_exp_2025_07_15_073287/data.csv',
]

all_data = pd.DataFrame()

for path in data_paths:
    # print(f'Processing file: {path}')

    # find load multiplier in path
    load_multiplier = path.split('l=')
    if len(load_multiplier) > 1:
        load_multiplier = load_multiplier[1].split('_')[0]
        if len(load_multiplier) == 3:
            load_multiplier = float(load_multiplier)/100
        elif len(load_multiplier) == 2:
            load_multiplier = float(load_multiplier)/10
    else:
        load_multiplier = 1

    print(f'Load multiplier: {load_multiplier}')

    data = pd.read_csv(path)

    print(data.shape)

    # group by algotithm and get mean and std
    columns = ['Unnamed: 0', 'run', 'Algorithm', 'total_ev_served', 'total_profits',
               'total_energy_charged', 'total_energy_discharged',
               'average_user_satisfaction', 'power_tracker_violation',
               'tracking_error', 'energy_tracking_error', 'energy_user_satisfaction',
               'total_transformer_overload', 'battery_degradation',
               'battery_degradation_calendar', 'battery_degradation_cycling',
               'total_reward']

    # Add load multiplier to data
    data['load_multiplier'] = load_multiplier

    # Append to all_data
    all_data = pd.concat([all_data, data], ignore_index=True)

# change algo names based on #     'QT', 'Q-DT')
# data_grouped.index = data_grouped.index.str.replace('PI_TD3', 'PI-TD3')
# data_grouped.index = data_grouped.index.str.replace('ppo', 'PPO')
# data_grouped.index = data_grouped.index.str.replace(
#     'DoNothing', 'No Charging')
# data_grouped.index = data_grouped.index.str.replace(
#     'V2GProfitMax_Grid_OracleGB', 'MPC (Oracle)')

# multiply user satisfaction by 100
all_data['average_user_satisfaction'] = all_data['average_user_satisfaction'] * 100
# divide profits by 1000
all_data['total_profits'] = all_data['total_profits'] / 1000
# divide total_reward by 1000
all_data['total_reward'] = all_data['total_reward'] / 10000

all_data['Algorithm'] = all_data['Algorithm'].str.replace('PI_TD3', 'PI-TD3')
all_data['Algorithm'] = all_data['Algorithm'].str.replace('ppo', 'PPO')
all_data['Algorithm'] = all_data['Algorithm'].str.replace(
    'DoNothing', 'No Charging')
all_data['Algorithm'] = all_data['Algorithm'].str.replace(
    'V2GProfitMax_Grid_OracleGB', 'MPC (Oracle)')
all_data['Algorithm'] = all_data['Algorithm'].str.replace(
    'DoNothing', 'No Charging')
all_data['Algorithm'] = all_data['Algorithm'].str.replace(
    'ChargeAsFastAsPossible', 'CAFAP')

# Define radar chart metrics

# drop algos SAC and PPO
all_data = all_data[~all_data['Algorithm'].isin(['SAC', 'PPO'])]

radar_chart_metrics = [
    # "voltage_violation",
    # 'voltage_violation_counter',
    'total_reward',
    'voltage_violation_counter_per_step',
    'average_user_satisfaction',
    'total_profits'
]

print(f'Total combined data shape: {all_data.shape}')
print(f'Load multipliers: {sorted(all_data["load_multiplier"].unique())}')
print(f'Algorithms: {all_data["Algorithm"].unique()}')

# Import necessary libraries for radar charts

# Set consistent styling
plt.rcParams['font.family'] = ['serif']


# Function to create radar chart
def create_radar_chart(data, metric, title, ax):
    # Get unique load multipliers and sort them
    load_multipliers = sorted(data['load_multiplier'].unique())

    # Calculate mean values for each algorithm and load multiplier
    grouped = data.groupby(['Algorithm', 'load_multiplier'])[
        metric].mean().reset_index()

    # Get unique algorithms and order them as requested
    algorithms = grouped['Algorithm'].unique()
    
    # Define the desired order
    algorithm_order = ['CAFAP', 'No Charging', 'TD3', 'PI-TD3', 'MPC (Oracle)']
    
    # Sort algorithms according to the desired order
    ordered_algorithms = [algo for algo in algorithm_order if algo in algorithms]
    # Add any remaining algorithms not in the predefined order
    ordered_algorithms.extend([algo for algo in algorithms if algo not in algorithm_order])

    # Create a pivot table for easier processing
    pivot_data = grouped.pivot(
        index='Algorithm', columns='load_multiplier', values=metric)

    # Use absolute values instead of relative to MPC
    absolute_data = {}
    for algo in ordered_algorithms:
        algo_values = pivot_data.loc[algo]
        absolute_data[algo] = list(algo_values)

    # Number of variables (load multipliers)
    N = len(load_multipliers)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Define colors for algorithms (using ordered list)
    # Use specific tab10 colors for each algorithm
    algorithm_colors = {
        'CAFAP': sns.color_palette("tab10")[3],      # Red
        'No Charging': sns.color_palette("tab10")[7], # Gray  
        'TD3': sns.color_palette("tab10")[0],        # Blue
        'PI-TD3': sns.color_palette("tab10")[2],     # Green
        'MPC (Oracle)': sns.color_palette("tab10")[1] # Orange
    }
    
    # Define markers for each algorithm
    algorithm_markers = {
        'CAFAP': 's',          # Square
        'No Charging': 'o',    # Circle
        'TD3': 'D',           # Diamond
        'PI-TD3': '^',        # Triangle
        'MPC (Oracle)': '*'   # Star
    }
    
    # Define hatching patterns for each algorithm
    algorithm_hatches = {
        'CAFAP': '///',        # Diagonal lines
        'No Charging': '...',  # Dots
        'TD3': 'xxx',         # Crosses
        'PI-TD3': '+++',      # Plus signs
        'MPC (Oracle)': '---' # Horizontal lines
    }
    
    # Define z-order for each algorithm (higher values appear on top)
    algorithm_zorders = {
        'PI-TD3': 100,         # Highest z-order for PI-TD3
        'MPC (Oracle)': 80,    # Second highest for MPC
        'TD3': 50,            # Default z-order for others
        'CAFAP': 50,
        'No Charging': 50
    }
    
    colors = [algorithm_colors.get(algo, sns.color_palette("tab10")[4]) for algo in ordered_algorithms]
    markers = [algorithm_markers.get(algo, 'o') for algo in ordered_algorithms]
    hatches = [algorithm_hatches.get(algo, '') for algo in ordered_algorithms]
    zorders = [algorithm_zorders.get(algo, 5) for algo in ordered_algorithms]

    # Plot each algorithm
    for i, (algo, values) in enumerate(absolute_data.items()):
        plot_values = values + [values[0]]  # Complete the circle
        ax.plot(angles, plot_values, marker=markers[i],
                linewidth=2.5, label=algo, color=colors[i],
                markerfacecolor='white',
                markeredgewidth=1.5,
                #  markersize=2,
                zorder=zorders[i], markersize=6)

        ax.fill(angles, plot_values, alpha=0.15, color=colors[i], 
                hatch=hatches[i], edgecolor=colors[i], linewidth=0.5, zorder=0)#zorders[i]-1)

    # Add labels
    ax.set_xticks(angles[:-1])
    x_tick_text = []
    for lm in load_multipliers:
        if lm == 1:
            x_tick_text.append('Load ×1\n' + r'($\mathbf{Trained}$)' )
        else:
            x_tick_text.append(f'Load\n×{lm}')            
            
    ax.set_xticklabels(x_tick_text)

    # Set y-axis limits and labels
    all_values = [val for vals in absolute_data.values() for val in vals]
    if metric.lower() == "voltage_violation_counter":
        # set yticks from 0 300 with 75 step
        ax.set_yticks(np.arange(0, 301, 75))
        # set y-tick zorder
        ax.tick_params(axis='y', which='both', zorder=100)
        # ax.set_ylabel('Voltage Violations Count', labelpad=30)
    elif 'violation' in metric.lower():
        # ax.set_ylim(0, 85)
        ax.set_yticks(np.arange(0, 76, 15))
        ax.tick_params(axis='y', which='both', zorder=100)
        # ax.set_ylabel('Voltage Violation Sum', labelpad=30)

    elif 'satisfaction' in metric.lower():
        ax.set_yticks(np.arange(40, 101, 20))
        ax.tick_params(axis='y', which='both', zorder=100)
        #set ylims
        # ax.set_ylim(40, 110)
        # ax.set_ylabel('User Satisfaction [%]', labelpad=30)
    elif "profits" in metric.lower():
        # ax.set_ylim(-4, 0.5)  
        ax.set_yticks(np.arange(-4, 1, 1))
        ax.tick_params(axis='y', which='both', zorder=100)

    elif "total_reward" in metric.lower():
        # ax.set_ylim(-20, 2)
        ax.set_yticks(np.arange(-20, 1, 4))
        ax.tick_params(axis='y', which='both', zorder=1000)
        # ax.set_ylabel('Total Reward [-]', labelpad=30)
    else:  # profits
        ax.tick_params(axis='y', which='both', zorder=100)
        # ax.set_ylabel('Total Profits', labelpad=30)

    # Add title
    ax.tick_params(axis='x', which='both', labelsize=13)
    ax.tick_params(axis='y', which='both', labelsize=13)
    ax.set_title(title, size=14, fontweight='bold', pad=20)

    # Add grid with more visibility
    ax.grid(True, alpha=0.7, linewidth=0.9, color='gray')
    
    #hide the spline
    ax.spines['polar'].set_visible(False)

    return ax


# Create all radar charts in a single figure
fig, axs = plt.subplots(figsize=(9, 9), nrows=2, ncols=2,
                        subplot_kw=dict(projection='polar'))
sns.set_style("whitegrid")
plt.rcParams['font.family'] = ['serif']

# Flatten the axes array for easier indexing
axs = axs.flatten()

# Create radar charts for each metric
metrics_titles = {
    'voltage_violation_counter': 'Voltage Violations [p.bus]',
    'voltage_violation': 'Voltage Violation Sum ',
    'voltage_violation_counter_per_step': 'Steps with Voltage Violations [-]',
    'average_user_satisfaction': 'Average User Satisfaction [%]',
    'total_profits': 'Total Profits [x10$^3$ €]',
    'total_reward': 'Total Reward [x10$^4$]'
}

for i, metric in enumerate(radar_chart_metrics):
    if metric in all_data.columns and i < len(axs):
        print(f'Creating radar chart for {metric}')
        create_radar_chart(all_data, metric, metrics_titles[metric], axs[i])
    else:
        if i < len(axs):
            axs[i].axis('off')  # Hide unused subplots
        if metric not in all_data.columns:
            print(f'Warning: {metric} not found in data columns')

# Add a single legend for all subplots
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.04),
           ncol=len(labels), fontsize=14)

plt.tight_layout()
plt.savefig('./results_analysis/pes/radar_charts_combined.png',
            bbox_inches='tight', dpi=300)
plt.savefig('./results_analysis/pes/radar_charts_combined.pdf',
            bbox_inches='tight', dpi=300)
# plt.show()
print("Combined radar charts generation completed!")
