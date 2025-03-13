"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time

learning_rate = 3e-5

counter = 0
for policy in ['TD3', 'mb_traj', 'SAC']: # MB
# for policy in ['mb_traj']:
    for batch_size in [256]:
        for expl_noise in [0.1]:
            for K in [1, 2, 10,25]:  # 512
                for seed in [9]:

                    if policy != 'mb_traj' and K != 1:
                        continue

                    command = 'tmux new-session -d \; send-keys " /home/sorfanouda/anaconda3/envs/dt/bin/python train_research.py' + \
                        ' --device cuda:0' + \
                        ' --expl_noise ' + str(expl_noise) + \
                        ' --batch_size=' + str(batch_size) + \
                        ' --lr ' + str(learning_rate) + \
                        ' --policy ' + policy + \
                        ' --seed ' + str(seed) + \
                        ' --K ' + str(K) + \
                        ' --group_name=50_grid_profitMax_tests' + \
                        ' --name ' +\
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
