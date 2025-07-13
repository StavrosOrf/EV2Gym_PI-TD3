import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

data = pd.read_csv(
    './results/eval_150cs_-1tr_v2g_grid_150_300_6_algos_25_exp_2025_07_12_842273/data.csv')


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
                #    'total_energy_charged',
                #    'total_energy_discharged',
                #    'total_reward',
                   'time',
                   ]

data = data[columns_to_keep]

print(data.head(20))

columns_to_drop = [
    'run',
]

# Create a half spider plot averaging across different algorithms
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

# Group by algorithm and calculate means
algo_means = data.groupby('Algorithm').agg({
    'voltage_violation_counter_per_step': 'mean',
    'average_user_satisfaction': 'mean', 
    'total_profits': 'mean'
}).reset_index()

# Define the metrics for the spider plot
metrics = ['voltage_violation_counter_per_step', 'average_user_satisfaction', 'total_profits']
metric_labels = ['Voltage Violations', 'User Satisfaction', 'Total Profits']

# Normalize the data to 0-1 scale for better visualization
def normalize_column(series):
    return (series - series.min()) / (series.max() - series.min())

normalized_data = []
for metric in metrics:
    normalized_data.append(normalize_column(algo_means[metric]).values)
normalized_data = np.array(normalized_data).T

# Number of variables
N = len(metrics)

# Compute angle for each axis (only half circle)
angles = [n / float(N) * np.pi for n in range(N)]
angles += angles[:1]  # Complete the circle

# Get unique algorithms and colors
algorithms = algo_means['Algorithm'].unique()
colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))

# Plot each algorithm
for i, algorithm in enumerate(algorithms):
    values = normalized_data[i].tolist()
    values += values[:1]  # Complete the circle
    
    ax.plot(angles, values, 'o-', linewidth=2, label=algorithm, color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])

# Add labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels)

# Set y-axis limits and labels
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])

# Limit to half circle (top half)
ax.set_thetamin(0)
ax.set_thetamax(180)

# Add title and legend
ax.set_title('Algorithm Performance Comparison\n(Half Spider Plot)', 
             size=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# Add grid
ax.grid(True)

plt.savefig('./results_analysis/pes/3d_metrics.png', dpi=300)

