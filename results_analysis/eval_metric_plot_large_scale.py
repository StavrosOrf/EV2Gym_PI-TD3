import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec

# Set style and font
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = ['serif']
sns.set_style("whitegrid")

data = pd.read_csv(
    './results/eval_500cs_-1tr_v2g_grid_500_bus_123_7_algos_50_exp_2025_07_15_974951/data.csv')


print(data.shape)

# group by algotithm and get mean and std
columns = ['Unnamed: 0', 'run', 'Algorithm', 'total_ev_served', 'total_profits',
           'total_energy_charged', 'total_energy_discharged',
           'average_user_satisfaction', 'power_tracker_violation',
           'tracking_error', 'energy_tracking_error', 'energy_user_satisfaction',
           'total_transformer_overload', 'battery_degradation',
           'battery_degradation_calendar', 'battery_degradation_cycling',
           'total_reward']

columns_to_keep = ['Algorithm',
                   'run',                   
                   'total_profits',
                   'voltage_violation',
                   'voltage_violation_counter',
                   'voltage_violation_counter_per_step',
                   'average_user_satisfaction',                                      
                   'total_energy_charged',
                   'total_energy_discharged',
                   'total_reward',
                   'time',
                   ]

metric_columns = [
    'total_energy_charged',
    'total_energy_discharged',
    'average_user_satisfaction',
    'total_profits',
    'voltage_violation_counter_per_step',
    'total_reward',
]

print(f'unique algorithms: {data["Algorithm"].unique()}')

# Filter data to keep only relevant columns
if 'voltage_violation_counter_per_step' not in data.columns and 'voltage_violation_counter' in data.columns:
    # Create per_step metric if it doesn't exist
    if 'time' in data.columns:
        data['voltage_violation_counter_per_step'] = data['voltage_violation_counter'] / data['time']
    else:
        data['voltage_violation_counter_per_step'] = data['voltage_violation_counter']

# Filter data for available columns
available_metrics = [col for col in metric_columns if col in data.columns]
plot_data = data[['Algorithm', 'run'] + available_metrics].copy()

# Clean algorithm names for better visualization
algorithm_mapping = {
    'pi_td3': 'PI-TD3',
    'pi_sac': 'PI-SAC',
    'td3': 'TD3',
    'sac': 'SAC',
    'ppo': 'PPO',
    'shac': 'SHAC',
    'mpc': 'MPC',
    'v2gprofitmax_grid_oraclegb': 'MPC (Oracle)',
    'donothing': 'No Charging',
    'chargeasfastaspossible': 'CAFAP'
}

# Clean algorithm names
plot_data['Algorithm'] = plot_data['Algorithm'].str.lower().replace(algorithm_mapping)

#multiply user satisfaction by 100 to get percentage
# if 'average_user_satisfaction' in plot_data.columns:
plot_data['average_user_satisfaction'] *= 100
#divide total energy charged and discharged by 1000 to get MWh
if 'total_energy_charged' in plot_data.columns:
    plot_data['total_energy_charged'] /= 1000
if 'total_energy_discharged' in plot_data.columns:
    plot_data['total_energy_discharged'] /= 1000

print(f'unique algorithms: {plot_data["Algorithm"].unique()}')

# Create algorithm color mapping using seaborn tab10
tab10_colors = sns.color_palette("tab10")
algorithm_colors = {
    'CAFAP': tab10_colors[3],     # Red
    'TD3': tab10_colors[0],       # Blue  
    'PI-TD3': tab10_colors[2],    # Green
    'MPC (Oracle)': tab10_colors[1],       # Orange
    'PI-SAC': tab10_colors[4],    # Purple
    'SHAC': tab10_colors[5],      # Brown
    'SAC': tab10_colors[6],       # Pink
    'PPO': tab10_colors[7],       # Gray
    'No Charging': tab10_colors[8], # Olive
    'Random': tab10_colors[9]     # Cyan
}

# Create figure with subplots for each metric
n_metrics = len(available_metrics)
fig = plt.figure(figsize=(10, 6))  # Wider for 3 columns, shorter for 2 rows
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
plt.rcParams['font.family'] = ['serif']
# Define metric labels for better visualization
metric_labels = {
    'total_energy_charged': 'Energy Charged [MWh]',
    'total_energy_discharged': 'Energy Discharged [MWh]',
    'average_user_satisfaction': 'User Satisfaction [%]',
    'total_profits': 'Total Profits [€]',
    'voltage_violation_counter_per_step': 'Voltage Violations [-]',
    'total_reward': 'Total Reward [-]'
}

# Define algorithm order for consistent display
unique_algorithms = plot_data['Algorithm'].unique()
# preferred_order = ['PI-TD3', 'TD3', 'SAC', 'PI-SAC',
#                    'PPO', 'SHAC', 'MPC (Oracle)', 'CAFAP', 'No Charging', 'Random']
algorithm_order = ['CAFAP', 'No Charging', 'SAC', 
                   'PPO', 'TD3','PI-TD3', 'MPC (Oracle)']

# print
# algorithm_order = [alg for alg in preferred_order if alg in unique_algorithms]
# Add any remaining algorithms not in preferred order
algorithm_order.extend([alg for alg in unique_algorithms if alg not in algorithm_order])

for i, metric in enumerate(available_metrics):
    row = i // 3  # Changed from 2 to 3 for 3 columns
    col = i % 3   # Changed from 2 to 3 for 3 columns
    ax = fig.add_subplot(gs[row, col])
    
    # Create catplot for this metric
    colors = [algorithm_colors.get(alg, 'gray') for alg in algorithm_order]
    
    # Use different plot types based on metric
    if metric in ['voltage_violation_counter_per_step', 'total_reward', 'total_profits']:
        # Use boxplot for specific metrics
        sns.boxplot(data=plot_data, x='Algorithm', y=metric,
                   order=algorithm_order, palette=colors, ax=ax,
                   showfliers=True, linewidth=1.5, boxprops=dict(alpha=0.7))
    else:
        # Use violin + swarm for other metrics
        # Create violin plot to show distribution shape
        sns.violinplot(data=plot_data, x='Algorithm', y=metric,
                       order=algorithm_order,
                       palette=colors,
                       ax=ax,
                       alpha=0.3,
                       inner=None,
                       linewidth=1)
        
        # Add swarm plot for individual points without overlap
        sns.swarmplot(data=plot_data, x='Algorithm', y=metric,
                      order=algorithm_order, palette=colors, ax=ax,
                      size=4, alpha=0.8)
    
    # Customize the subplot
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=14)
    
    # Remove x-axis labels and ticks for cleaner look
    ax.set_xlabel('')
    ax.set_xticklabels([])
    # ax.tick_params(axis='x', which='both', bottom=False, top=False)
    
    # Format y-axis based on metric type
    if 'energy' in metric.lower():
        ax.ticklabel_format(axis='y', style='sci', scilimits=(1, 1))
        if "discharged" in metric.lower():
            ax.set_ylim(-1, 60)  # Adjust limit for discharged energy
        else:
            ax.set_ylim(-2, 100)
        
    elif "reward" in metric.lower():
        ax.ticklabel_format(axis='y', style='sci', scilimits=(6, 6))
    elif 'profit' in metric.lower() or 'reward' in metric.lower():
        ax.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
    elif 'satisfaction' in metric.lower():
        ax.set_ylim(40, 102)
   
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#666666')
        
    ax.minorticks_on()
        
    # Enhance grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

# Create common legend at the top
handles = []
labels = []
for alg in algorithm_order:
    color = algorithm_colors.get(alg, 'gray')
    handle = plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7)
    handles.append(handle)
    labels.append(alg)

# Add legend at the top of the figure
fig.legend(handles, labels,
           loc='upper center',
           bbox_to_anchor=(0.5, 1.1), 
           ncol=4,  # Increased to 5 columns to fit better with wider layout
           fontsize=13,
           frameon=True,
           fancybox=True,
           shadow=True)

# Remove empty subplot if odd number of metrics
if n_metrics % 3 != 0:  # Changed condition for 3 columns
    for empty_idx in range(n_metrics, 6):  # Fill up to 6 subplots (2x3)
        empty_row = empty_idx // 3
        empty_col = empty_idx % 3
        if empty_row < 2 and empty_col < 3:  # Make sure we're within bounds
            fig.delaxes(fig.add_subplot(gs[empty_row, empty_col]))

# plt.suptitle('Algorithm Performance Comparison - Large Scale Evaluation', 
#              fontsize=16, fontweight='bold', y=0.92)

plt.tight_layout()
plt.subplots_adjust(top=0.90)  # Adjusted for the wider layout with legend
plt.savefig('./results_analysis/pes/algorithm_performance_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('./results_analysis/pes/algorithm_performance_comparison.pdf',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

# Print summary statistics
print("\nSummary Statistics by Algorithm:")
print("=" * 50)
summary_stats = plot_data.groupby('Algorithm')[available_metrics].agg(['mean', 'std', 'count'])
for algorithm in algorithm_order:
    if algorithm in summary_stats.index:
        print(f"\n{algorithm}:")
        for metric in available_metrics:
            mean_val = summary_stats.loc[algorithm, (metric, 'mean')]
            std_val = summary_stats.loc[algorithm, (metric, 'std')]
            count_val = summary_stats.loc[algorithm, (metric, 'count')]
            print(f"  {metric_labels.get(metric, metric)}: {mean_val:.3f} ± {std_val:.3f} (n={count_val})")

print(f"\nVisualization saved as 'algorithm_performance_comparison.png' and '.pdf'")
print(f"Dataset shape: {data.shape}")
print(f"Algorithms analyzed: {', '.join(algorithm_order)}")
print(f"Metrics visualized: {len(available_metrics)}")

