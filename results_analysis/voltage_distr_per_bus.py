import pickle
import gzip
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from ev2gym.models.ev2gym_env import EV2Gym

# load_path = "./results/eval_150cs_-1tr_v2g_grid_150_300_4_algos_3_exp_2025_07_12_218549/voltage_minimum.pkl.gz"
load_path = "./results/eval_150cs_-1tr_v2g_grid_150_300_4_algos_30_exp_2025_07_12_251942/voltage_minimum.pkl.gz"

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

# Shorten algorithm names
def shorten_algorithm_name(name):
    """Shorten algorithm names to max 10 characters"""
    name_mapping = {
        "<class 'ev2gym.baselines.heuristics.ChargeAsFastAsPossible'>": 'CAFAP',
        "<class 'ev2gym.baselines.heuristics.DoNothing'>": 'Do Nothing',
        "<class 'ev2gym.baselines.heuristics.RandomAgent'>": 'RandomAgent',
        "<class 'ev2gym.baselines.heuristics.Optimal'>": 'Optimal (Offline)',
        "pi_td3_run_10_K=30_scenario=grid_v2g_profitmax_80448-188243": 'PI-TD3',
        "<class 'ev2gym.baselines.heuristics.TD3-LA'>": 'td3Lookahead',
        "<class 'ev2gym.baselines.heuristics.SHAC'>": 'shac'
    }
    
    # Use mapping if available, otherwise truncate to 10 chars
    # print(f"Shortening algorithm name: {name}")
    # input(f"Press Enter to continue...")
    if name in name_mapping:
        return name_mapping[name]
    # elif len(name) <= 10:
    #     return name
    # else:
    #     return name[:10]

# print unique_algorithms
unique_algorithms = voltage_df['algorithm'].unique()
print(f'Unique algorithms: {unique_algorithms}')

# Apply algorithm name shortening
voltage_df['algorithm'] = voltage_df['algorithm'].apply(shorten_algorithm_name)

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
colors = sns.color_palette("husl", len(unique_algos))
algo_color_map = dict(zip(unique_algos, colors))

# Get unique buses and sort them
unique_buses = sorted(voltage_df['bus'].unique())
n_buses = len(unique_buses)

# Create a horizontal figure with subplots for each bus
fig, axes = plt.subplots(1, n_buses, figsize=(4*n_buses, 6), sharey=True)

# If only one bus, make axes a list for consistency
if n_buses == 1:
    axes = [axes]

# Plot histogram for each bus
for i, bus in enumerate(unique_buses):
    bus_data = voltage_df[voltage_df['bus'] == bus]
    
    # Create histogram for this bus
    sns.histplot(data=bus_data, 
                 x='voltage', 
                 hue='algorithm',
                 stat='density',  # Use density for better comparison
                 kde=True,  # Add KDE for smoother visualization
                 alpha=0.7,
                 palette=algo_color_map,
                 ax=axes[i])
    
    # Customize each subplot
    axes[i].set_title(f'Bus {bus}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Voltage (p.u.)', fontsize=12)
    
    # Only show y-label on the first subplot
    if i == 0:
        axes[i].set_ylabel('Density', fontsize=12)
    else:
        axes[i].set_ylabel('')
    
    # Remove legend from individual subplots (we'll add a common one)
    axes[i].legend_.remove() if axes[i].legend_ else None

# Add a common legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title='Algorithm', bbox_to_anchor=(1.02, 0.5), 
           loc='center left', fontsize=10)

# Add main title
fig.suptitle('Voltage Distribution per Bus by Algorithm', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()