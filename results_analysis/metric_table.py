import pandas as pd


data = pd.read_csv(
    './results_analysis/max_rewards.csv')


print(data.shape)

groups = [
    'v2g',
    'v2g_profitmax',
    'grid',
]

#rename SAC_1 to SAC
data['algorithm'] = data['algorithm'].replace("SAC_1", "SAC")
data['algorithm'] = data['algorithm'].replace("TD3_1", "TD3")

for group in groups:
    data_group = data[data['group'] == group]
    print(f'group: {group}')

# createa a laex table with the  max value of the reward for each algorithm
    alg_order = ['TD3',
                 'SAC',
                 'MB-TD3_1',
                 'MB-TD3_2',
                 'MB-TD3_5',
                 'MB-TD3_10',
                 'MB-TD3_20',
                 'MB-TD3_25',
                 'MB-TD3_40',
                 'MB-TD3_50',
                 'MB-TD3_60',
                 'MB-TD3_70',
                 'MB-TD3_80',
                 ]
    

    
    all_metrics = ['algorithm']
    
    if group == 'grid':
        metrics = ['max_reward', 'profits', 'user_satisfaction', 'voltage_violation']
    elif group == 'v2g':
        metrics = ['max_reward', 'profits']
    else:
        metrics = ['max_reward', 'profits', 'user_satisfaction']

    metrics = all_metrics + metrics + ['time_to_max']
    data_group = data_group[metrics]

    #sort by alg_order
    data_group = data_group.sort_values(by='algorithm', key=lambda x: x.map(
    {alg: i for i, alg in enumerate(alg_order)}))

    data_group['max_reward'] = data_group['max_reward']\
        .apply(lambda x: f"${x:.1f}$")
    
    data_group['profits'] = data_group['profits']\
        .apply(lambda x: f"${x:.1f}$")
    
    if data_group.keys().__contains__('user_satisfaction'):
        data_group['user_satisfaction'] = data_group['user_satisfaction']\
            .apply(lambda x: f"${x:.2f}$")
    
    if data_group.keys().__contains__('voltage_violation'):
        data_group['voltage_violation'] = data_group['voltage_violation']\
            .apply(lambda x: f"${x:.2f}$")
        
    data_group['time_to_max'] = data_group['time_to_max']\
        .apply(lambda x: f"${x:.0f}$")
    
    #rename column name max_reward to Max Reward
    data_group = data_group.rename(columns={'max_reward': 'Max Reward',
                                            'profits': 'Profits [€]',
                                            'user_satisfaction': 'EV User Satisfaction [\%]',
                                            'voltage_violation': 'Voltage Violation [Total p.u.]',
                                            'time_to_max': 'Training Time to Max [min]',
                                            'algorithm': 'Algorithm',
                                            })
    
    #drop index
    data_group = data_group.set_index('Algorithm')

    # print(data_group)
    print(data_group.to_latex())


exit()

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
                   'total_energy_charged',
                   'total_energy_discharged',
                   'average_user_satisfaction',
                   'power_tracker_violation',
                   'total_profits',
                   'total_reward',
                   'time',
                   ]

data = data[columns_to_keep]

# find the PowerTrackingErrorrMin.total_reward
PowerTrackingErrorrMin = data[data.Algorithm ==
                              "PowerTrackingErrorrMin"]
# find the mean and std of the optimality gap for each algorithm

print(data.head(20))

columns_to_drop = [
    'run',
]

data = data.drop(columns=columns_to_drop)


data_grouped = data.groupby('Algorithm').agg(['mean', 'std'])

# create new columns with the mean and std of the total_energy_charged combined as a string
data_grouped['total_energy_charged'] = data_grouped['total_energy_charged']\
    .apply(lambda x: f"${x['mean']/1000:.1f}$ ±${x['std']/1000:.1f}$", axis=1)
data_grouped['total_energy_discharged'] = data_grouped['total_energy_discharged']\
    .apply(lambda x: f"${x['mean']/1000:.2f}$ ±${x['std']/1000:.2f}$", axis=1)
data_grouped['average_user_satisfaction'] = data_grouped['average_user_satisfaction']\
    .apply(lambda x: f"${x['mean']*100:.1f}$ ±${x['std']*100:.1f}$", axis=1)
data_grouped['total_profits'] = data_grouped['total_profits']\
    .apply(lambda x: f"${x['mean']:.0f}$ ±${x['std']:.0f}$", axis=1)
data_grouped['power_tracker_violation'] = data_grouped['power_tracker_violation']\
    .apply(lambda x: f"${x['mean']:.1f}$ ±${x['std']:.1f}$", axis=1)
data_grouped['total_reward'] = data_grouped['total_reward']\
    .apply(lambda x: f"${x['mean']/100000:.3f}$ ±${x['std']/100000:.3f}$", axis=1)
data_grouped['time'] = data_grouped['time']\
    .apply(lambda x: f"${x['mean']/300:.3f}$", axis=1)

# rearange rows


# drop the mean and std columns
data_grouped = data_grouped.droplevel(1, axis=1)
# print the results
# drop duplicate columns
data_grouped = data_grouped.loc[:, ~data_grouped.columns.duplicated()]
# rename columns
data_grouped.columns = ['Energy Charged [MWh]',
                        'Energy Discharged [MWh]',
                        'User Satisfaction [%]',
                        'Power Violation [kW]',
                        'Costs [€]',
                        'Reward [-]',
                        'Step time [sec/step]',
                        ]

print(data_grouped)

# rename algorithm names with shorter names
data_grouped.index = data_grouped.index.str.replace(
    'ChargeAsFastAsPossible', 'CAFAP')
data_grouped.index = data_grouped.index.str.replace(
    'QT', 'Q-DT')
data_grouped.index = data_grouped.index.str.replace('RoundRobin_GF', 'BaU')
data_grouped.index = data_grouped.index.str.replace(
    'mo_PST_V2GProfitMaxOracleGB', 'Optimal (Offline)')
data_grouped.index = data_grouped.index.str.replace('GNN_act_emb_DT', 'GNN-DT')


# change order of rows
data_grouped = data_grouped.reindex(['CAFAP',
                                     'BaU',
                                     'DT',
                                     'Q-DT',
                                     'GNN-DT',
                                     'Optimal (Offline)'
                                     ])


# rename PowerTrackingErrorrMin to Optimal
# print(data_grouped)
print(data_grouped.to_latex())


# % \usepackage{tabularray}
# \begin{table}
# \centering
# \captionsetup{labelformat=empty}
# \caption{Analysis of the Overall Reward into its Three Components}
# \label{tab:reward_breakdown}
# \begin{tblr}{
#   cells = {c,t},
#   vline{2} = {2-7}{0.05em},
#   hline{1,8} = {-}{0.08em},
#   hline{2} = {-}{0.05em},
# }
# Algorithm & {Energy Charged\\~[MWh]} & {Energy Discharged\\~[MWh]} & {User Satisfaction\\~[\%]} & {Power Violation\\~[kW]} & Costs~[€] & Reward [10-5] & {Exec. Time\\~[sec/step]}\\
# CAFAP & $1.3$ ±$0.2$ & $0.00$ ±$0.00$ & $100.0$ ±$0.0$ & $1289.2$ ±$261.8$ & $-277$ ±$165$ & $-1.974$ ±$0.283$ & $0.001$\\
# BaU & $1.3$ ±$0.2$ & $0.00$ ±$0.00$ & $99.9$ ±$0.2$ & $10.5$ ±$9.4$ & $-255$ ±$156$ & $-0.679$ ±$0.067$ & $0.001$\\
# DT & $0.9$ ±$0.1$ & $0.03$ ±$0.01$ & $94.4$ ±$1.6$ & $58.7$ ±$28.3$ & $-173$ ±$104$ & $-0.462$ ±$0.093$ & $0.006$\\
# Q-DT & $1.0$ ±$0.1$ & $0.00$ ±$0.00$ & $93.6$ ±$2.1$ & $20.1$ ±$21.4$ & $-187$ ±$113$ & $-0.665$ ±$0.135$ & $0.010$\\
# GNN-DT & $0.9$ ±$0.1$ & $0.19$ ±$0.03$ & $99.3$ ±$0.2$ & $21.7$ ±$22.8$ & $-142$ ±$89$ & $-0.027$ ±$0.023$ & $0.023$\\
# Optimal (Offline) & $1.9$ ±$0.2$ & $1.08$ ±$0.19$ & $99.1$ ±$0.2$ & $2.0$ ±$4.6$ & $-119$ ±$84$ & $-0.020$ ±$0.015$ & -
# \end{tblr}
# \end{table}
