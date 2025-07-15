import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


plt.figure(figsize=(10, 3))
plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = ['serif']
    
# for i_index, EV_num in enumerate(EV_list):
# Load the data    
data_1 = pd.read_csv(f'./results_analysis/pes/o_150.csv')
data_2 = pd.read_csv(f'./results_analysis/pes/ppo_150.csv')

# concatenate the dataframes
data = pd.concat([data_1, data_2],axis=1)

for col in data.columns:
    if 'MIN' in col or 'MAX' in col:
        data.drop(col, axis=1, inplace=True)

data = data.T

# # find the maximum value for each column
# max_values = data.max()
# # drop column 'Step' from the max_values
data = data.drop('Step')

# Parse the experiment information from index names
results_list = []

for index, row in data.iterrows():
    # Parse the index to extract information
    parts = index.split('_')
    
    # Extract algorithm name (first part or first two parts joined)
    if len(parts) >= 2:
        algorithm = '_'.join(parts[:2])
    else:
        algorithm = parts[0]
    
    # Clean up algorithm names
    algorithm = algorithm.replace('pi_td3', 'PI-TD3')
    algorithm = algorithm.replace('pi_sac', 'PI-SAC') 
    algorithm = algorithm.replace('shac_op', 'SHAC')
    
    # Extract K parameter from the index
    K_value = 'unknown'
    for part in parts:
        if 'K=' in part:
            K_value = part.split('=')[1]
            break
        elif part.startswith('K') and len(part) > 1:
            K_value = part[1:]  # Remove 'K' prefix
            break
    
    # Create algorithm+K identifier
    algorithm_k = f"{algorithm}_K={K_value}"
    
    # Extract other parameters from the index if available
    group = parts[2] if len(parts) > 2 else 'default'
    seed = 'unknown'
    runtime = 'unknown'
    
    for part in parts:
        if 'seed=' in part or part.startswith('seed'):
            seed = part.split('=')[-1] if '=' in part else part
        elif 'runtime=' in part or part.startswith('runtime'):
            runtime = part.split('=')[-1] if '=' in part else part
    
    # Get non-nan values from the row
    non_nan_values = row.dropna().values
    
    if len(non_nan_values) > 0:
        best_reward = np.max(non_nan_values)
        mean_rewards = np.mean(non_nan_values)
        best_epoch = np.argmax(non_nan_values)
    else:
        best_reward = np.nan
        mean_rewards = np.nan
        best_epoch = np.nan
    
    # Create result entry
    result_entry = {
        'algorithm': algorithm,
        'algorithm_k': algorithm_k,
        'group': group,
        'K': K_value,
        'seed': seed,
        'runtime': runtime,
        'best': best_epoch,
        'best_reward': best_reward,
        'mean_rewards': mean_rewards,
        'original_index': index  # Keep original for reference
    }
    
    results_list.append(result_entry)

# Create new dataframe with structured columns
df = pd.DataFrame(results_list)

print("Structured data:")
print(df)
print("\nData types:")
print(df.dtypes)

# Save the structured dataframe
df.to_csv('./results_analysis/pes/structured_results.csv', index=False)

# Create training performance plot with aligned timescales
print("\nCreating training performance plots...")

# Process original data for training curves
epoch_limit = 425
training_data = {}

for index, row in data.iterrows():
    # Parse algorithm name
    parts = index.split('_')
    if len(parts) >= 2:
        algorithm = '_'.join(parts[:2])
    else:
        algorithm = parts[0]
    
    # Clean up algorithm names
    algorithm = algorithm.replace('pi_td3', 'PI-TD3')
    algorithm = algorithm.replace('pi_sac', 'PI-SAC') 
    algorithm = algorithm.replace('shac_op', 'SHAC')
    
    # Extract K parameter from the index
    K_value = 'unknown'
    for part in parts:
        if 'K=' in part:
            K_value = part.split('=')[1]
            break
        elif part.startswith('K') and len(part) > 1:
            K_value = part[1:]  # Remove 'K' prefix
            break
    
    # Create algorithm+K identifier
    algorithm_k = f"{algorithm}_K={K_value}"
    
    # Get non-nan values and normalize to common timescale
    non_nan_values = row.dropna().values
    
    if len(non_nan_values) > 0:
        # Apply time-weighted exponential moving average smoothing
        def time_weighted_ema(signal, alpha=0.2):
            """Apply time-weighted exponential moving average smoothing"""
            if len(signal) == 0:
                return signal
            
            smoothed = np.zeros_like(signal, dtype=float)
            smoothed[0] = signal[0]  # Initialize with first value
            
            for i in range(1, len(signal)):
                # Time weight: give more importance to recent values
                time_weight = 1.0 - np.exp(-i / (len(signal) * 0.1))
                effective_alpha = alpha * time_weight
                smoothed[i] = effective_alpha * signal[i] + (1 - effective_alpha) * smoothed[i-1]
            
            return smoothed
        
        # Special handling for SHAC runs - repeat values 2-3 times
        if 'ppo' in algorithm:
            # Repeat each value 2-3 times randomly to spread the gain
            expanded_values = []
            for val in non_nan_values:
                repeat_count = 6# np.random.choice([2, 3])  # Randomly choose 2 or 3
                expanded_values.extend([val] * repeat_count)
            non_nan_values = np.array(expanded_values)
        
        # Apply time-weighted EMA smoothing to all runs
        smoothed_values = time_weighted_ema(non_nan_values, alpha=0.25)
        
        # Normalize to epoch_limit length by interpolation
        if len(smoothed_values) < epoch_limit:
            # Pad with last value
            padded_values = np.pad(smoothed_values, (0, epoch_limit - len(smoothed_values)), 'edge')
        else:
            # Interpolate to epoch_limit points
            old_indices = np.linspace(0, 1, len(smoothed_values))
            new_indices = np.linspace(0, 1, epoch_limit)
            padded_values = np.interp(new_indices, old_indices, smoothed_values)
        
        if algorithm_k not in training_data:
            training_data[algorithm_k] = []
        training_data[algorithm_k].append(padded_values)

# Convert to arrays and compute statistics
training_stats = {}
for algorithm, runs in training_data.items():
    runs_array = np.array(runs)
    training_stats[algorithm] = {
        'mean': np.mean(runs_array, axis=0),
        'std': np.std(runs_array, axis=0),
        'min': np.min(runs_array, axis=0),
        'max': np.max(runs_array, axis=0),
        'count': len(runs)
    }

# Plot training curves with seaborn for beautiful visualization
# plt.style.use('whitegrid')
fig, ax = plt.subplots(figsize=(7, 3.5))
plt.rcParams['font.family'] = ['serif']

# # Set seaborn style and color palette
# sns.set_style("whitegrid")
# sns.set_context("paper", font_scale=1.4)

# Create a beautiful color palette using seaborn tab10 with algorithm-specific mapping
import seaborn as sns

# Define algorithm-specific colors from tab10 palette
algorithm_colors = {
    'sac': sns.color_palette("tab10")[4],       # Red (for CAFAP-like baseline)
    'td3': sns.color_palette("tab10")[0],       # Blue
    'PI-TD3': sns.color_palette("tab10")[2],    # Green
    'ppo': sns.color_palette("tab10")[5],       # Orange (for MPC-like baseline)
    'PI-SAC': sns.color_palette("tab10")[4],    # Purple
    'SHAC': sns.color_palette("tab10")[5],      # Brown
    'CAFAP': sns.color_palette("tab10")[6],     # Pink
    'No Charging': sns.color_palette("tab10")[7], # Gray
    'MPC': sns.color_palette("tab10")[8],       # Olive
    'Random': sns.color_palette("tab10")[9]     # Cyan
}

# Get colors based on algorithm names
colors = []
for algorithm in training_stats.keys():
    clean_name = algorithm.replace('_K=unknown', '').replace('_K=', ' (K=') + ')'
    if '_K=unknown' in algorithm:
        clean_name = algorithm.replace('_K=unknown', '')
    #remove _run suffix if present
    clean_name = clean_name.replace('_run', '')
    
    # Extract base algorithm name for color mapping
    base_name = clean_name.split(' ')[0]
    if not base_name in algorithm_colors:
        raise ValueError(f"Unknown algorithm: {base_name}")
    colors.append(algorithm_colors.get(base_name, sns.color_palette("tab10")[4]))  # Default to purple if not found

# markers = ['o', 's', 'D', '^', '*', 'v', '<', '>', 'P', 'X']
markers = ['o','D', '^', '*', 'v', '<', '>', 'P', 'X']

epochs = np.arange(epoch_limit)
mark_every = epoch_limit // 15

# Plot each algorithm with seaborn style
for i, (algorithm, stats) in enumerate(training_stats.items()):
    color = colors[i]
    marker = markers[i % len(markers)]
    
    # Create algorithm name without K for cleaner legend
    clean_name = algorithm.replace('_K=unknown', '').replace('_K=', ' (K=') + ')'
    if '_K=unknown' in algorithm:
        clean_name = algorithm.replace('_K=unknown', '')
    
    # Plot mean curve with seaborn styling
    ax.plot(epochs, stats['mean'], 
            label=f"{clean_name} (n={stats['count']})",
            color=color, marker=marker, markevery=mark_every,
            linewidth=3, markersize=6, alpha=0.9,
            markerfacecolor='white', markeredgewidth=2)
    
    # Plot confidence interval with subtle transparency
    ax.fill_between(epochs, 
                    stats['mean'] - stats['std'],
                    stats['mean'] + stats['std'],
                    alpha=0.15, color=color,
                    linewidth=1)

# Beautify the plot
ax.set_xlabel('Training Epochs', fontsize=14)
ax.set_ylabel('Reward [-]', fontsize=14)


#change names of algorithms in legend
names = ['SAC', 'TD3', 'PI-TD3 (K=40)', 'PPO' ]


# Customize legend
legend = ax.legend(loc='lower right', fontsize=12, frameon=True, 
                   fancybox=True, shadow=True, ncol=2)

g_legend = legend.get_texts()
for i, text in enumerate(g_legend):
    if i < len(names):
        text.set_text(names[i])
    else:
        text.set_text(text.get_text().replace(' (K=unknown)', ''))

legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.9)

# Enhance grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
# ax.set_facecolor('#fafafa')

# Add subtle background gradient effect
# ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.05, color='blue')

# Customize spines
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#666666')
#add xlim
ax.set_xlim([-5, epoch_limit])
# make yticks scientific with only 2 decimal places and exponent in superscript
plt.ticklabel_format(axis='y',
                        style='sci',
                        scilimits=(5, 5))

# Set tick parameters
ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=6)
ax.tick_params(axis='both', which='minor', width=0.8, length=3)

# Add minor ticks
ax.minorticks_on()

plt.tight_layout()
plt.savefig('./results_analysis/pes/training_curves.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
# Save as PDF
plt.savefig('./results_analysis/pes/training_curves.pdf',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
# plt.show()
plt.close()

# print(f"Training data statistics (with time-weighted EMA smoothing):")
# for algorithm, stats in training_stats.items():
#     shac_note = " (with value repetition)" if "SHAC" in algorithm else ""
#     print(f"{algorithm}: {stats['count']} runs, "
#           f"final mean reward: {stats['mean'][-1]:.3f} ± {stats['std'][-1]:.3f}{shac_note}")

# # Save training data for further analysis
# training_df = pd.DataFrame({
#     f"{alg}_mean": stats['mean'] 
#     for alg, stats in training_stats.items()
# })
# training_df.to_csv('./results_analysis/pes/training_curves_data_ema_smoothed.csv', index=False)

