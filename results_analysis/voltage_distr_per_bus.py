import pickle
import gzip
import os
import sys

from visualize_voltage import shorten_algorithm_name

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from ev2gym.models.ev2gym_env import EV2Gym

load_path = "./results/eval_150cs_-1tr_v2g_grid_150_300_7_algos_1_exp_2025_07_15_068952/voltage_minimum.pkl.gz"
# load_path = "./results/eval_150cs_-1tr_v2g_grid_150_300_7_algos_10_exp_2025_07_15_996102/voltage_minimum.pkl.gz"

#open file
with gzip.open(load_path, 'rb') as f:
    voltage_minimum = pickle.load(f)
    
print(f'Loaded voltage minimum from {load_path}')
    
#create new dataframe with the voltage and add bus number and algorithm as column
import pandas as pd
voltage_data = []
for algo, buses in voltage_minimum.items():
    counter = 0
    for bus, voltage in buses.items():
        # print(f'voltage for algorithm {algo} and bus {bus}: {voltage}')
        if counter == 0:
                    counter += 1
                    continue
        
        for v_ in voltage:            
            
            for v in v_:
                
                
                # print(f'Algorithm: {algo}, Bus: {bus}, Voltage: {v}')
                # Ensure all data types are consistent
                voltage_data.append({
                    "algorithm": str(algo), 
                    "bus": int(bus), 
                    "voltage": float(v)
                })
                # input("press enter to continue")
            
voltage_df = pd.DataFrame(voltage_data)

# print unique_algorithms
unique_algorithms = voltage_df['algorithm'].unique()
print(f'Unique algorithms: {unique_algorithms}')

# Apply algorithm name shortening
voltage_df['algorithm'] = voltage_df['algorithm'].apply(shorten_algorithm_name)
unique_algorithms = voltage_df['algorithm'].unique()
print(f'Unique algorithms: {unique_algorithms}')
# input("Press enter to continue...")
#drop SAC
voltage_df = voltage_df[voltage_df['algorithm'] != 'SAC']
voltage_df = voltage_df[voltage_df['algorithm'] != 'No Charging']

# Convert columns to proper data types
voltage_df['algorithm'] = voltage_df['algorithm'].astype('category')
# voltage_df['bus'] = voltage_df['bus'].astype('category') 
voltage_df['voltage'] = pd.to_numeric(voltage_df['voltage'], errors='coerce')

# Remove any rows with NaN values
voltage_df = voltage_df.dropna()

print(voltage_df.head())
print(f"Data types: {voltage_df.dtypes}")
print(f"Shape: {voltage_df.shape}")
#use seaborn to plot the voltage distribution per bus to compare the algorithms
import seaborn as sns
import matplotlib.pyplot as plt

# Set a color palette for algorithms
unique_algos = voltage_df['algorithm'].unique()
colors = sns.color_palette("tab10", len(unique_algos))
algo_color_map = dict(zip(unique_algos, colors))

# Define hatching patterns for each algorithm
algo_patterns = {
    'CAFAP': '///',
    'TD3': '...',
    'PI-TD3': 'xxx',
    'MPC (Oracle)': 'o'
}

# Get unique buses and sort them
unique_buses = sorted(voltage_df['bus'].unique())
n_buses = len(unique_buses)

# Create a horizontal figure with subplots for each bus
fig, axes = plt.subplots(1, n_buses, figsize=(12,3.5), sharey=True)
plt.rcParams['font.family'] = ['serif']

# If only one bus, make axes a list for consistency
if n_buses == 1:
    axes = [axes]

# Plot box plot for each bus
for i, bus in enumerate(unique_buses):
    bus_data = voltage_df[voltage_df['bus'] == bus]
    
    # Create box plot for this bus
    box_plot = sns.boxplot(data=bus_data, 
                          x='algorithm', 
                          y='voltage',
                          order=['CAFAP','TD3','PI-TD3','MPC (Oracle)'],
                          palette=algo_color_map,
                          showfliers=True,
                          ax=axes[i])
    
    # Apply hatching patterns to each box
    for patch, algorithm in zip(box_plot.patches, ['CAFAP','TD3','PI-TD3','MPC (Oracle)']):
        if algorithm in algo_patterns:
            patch.set_hatch(algo_patterns[algorithm])
            patch.set_edgecolor('k')
            patch.set_linewidth(0.2)
    
    #draw a line at 0.95 p.u. voltage
    axes[i].axhline(y=0.95, color='r', linestyle='--', label='0.95 p.u. Voltage Limit')
    
    # Customize each subplot
    if i == 0:
        axes[i].set_title(f'Bus {bus}       ', fontsize=9)
    else:
        axes[i].set_title(f'{bus}', fontsize=9)
    axes[i].set_ylabel('Voltage [p.u.]', fontsize=12)
    # axes[i].set_xlabel('Algorithm', fontsize=12)
    #set empty x-axis label
    axes[i].set_xlabel('')
            
    #remove text from x-axis labels
    
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, fontsize=10, ha='right')

    # Remove text from x-axis labels
    axes[i].set_xticklabels([''] * len(axes[i].get_xticklabels()))

    # Add grid for better readability
    axes[i].grid(True, alpha=0.3)
    
    for spine in axes[i].spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#666666')

    axes[i].minorticks_on()

# Create legend handles for each algorithm with patterns
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=algo_color_map[algo], 
                        hatch=algo_patterns.get(algo, ''), 
                        edgecolor='k',
                        linewidth=0.8,
                        label=algo) 
                  for algo in ['CAFAP', 'TD3', 'PI-TD3', 'MPC (Oracle)'] 
                  if algo in algo_color_map]

# Add legend below the figure
fig.legend(handles=legend_elements, 
          loc='lower center', 
          bbox_to_anchor=(0.5, -0.02),
          ncol=len(legend_elements),
          fontsize=12,
          frameon=False)

# Add main title
# fig.suptitle('Voltage Distribution per Bus by Algorithm', fontsize=16, fontweight='bold')

# Adjust layout to make room for legend
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.15)
plt.savefig('./results_analysis/pes/voltage_boxplot_per_bus.png', bbox_inches='tight', dpi=300)
#save pdf
plt.savefig('./results_analysis/pes/voltage_boxplot_per_bus.pdf', bbox_inches='tight', dpi=300)
# plt.show()
