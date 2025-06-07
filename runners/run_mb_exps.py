"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time

learning_rate = 3e-5
scenario = "grid_v2g_profitmax"
# scenario = "v2g_profitmax"
counter = 0

# for policy in ['TD3', 'mb_traj', 'SAC']: # MB mb_traj_DDPG
# for policy in ['mb_traj', 'SAC']: # MB
for policy in ['mb_traj', 'TD3',]:
    for batch_size in [64]:
        for critic in [True, False]:
            for K in [1, 2, 10, 20]:  # 512
                for seed in [9]:
                    
                    if not critic and policy != 'mb_traj':
                        continue

                    if (policy != 'mb_traj' and policy != 'mb_traj_DDPG') and K != 1:
                        continue
                    
                    if policy == 'mb_traj_DDPG' and K not in [2, 20]:
                        continue

                    extra_args = ''

                    if not critic:

                        extra_args = ' --disable_critic'

                    command = 'tmux new-session -d \; send-keys " /home/sorfanouda/anaconda3/envs/dt/bin/python train_research.py' + \
                        ' --device cuda:0' + \
                        ' --scenario ' + scenario + \
                        ' --batch_size ' + str(batch_size) + \
                        ' --lr ' + str(learning_rate) + \
                        ' --policy ' + policy + \
                        ' --seed ' + str(seed) + \
                        ' --K ' + str(K) + \
                        extra_args + \
                        ' --group_name "AblationTests"' + \
                        ' --name LookaheadReward_' +\
                        f'Critic={critic}_' + \
                        f'{policy}' + \
                        '_K=' + str(K) + \
                        '_batch_size=' + str(batch_size) + \
                        '_seed=' + str(seed) + \
                        '" Enter'
                    os.system(command=command)
                    print(command)
                    # wait for 20 seconds before starting the next experiment
                    time.sleep(15)
                    counter += 1
