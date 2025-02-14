"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time
import random

counter = 0
n_runs = 1
strt_run = 0

# save to a file the current execution time
# os.system('date > start_time.txt')


# for algorithm in ['ddpg', 'ppo', 'tqc', 'a2c', 'trpo','td3']:
# for algorithm in ['sac','td3']:

for algorithm in ['ppo', 'td3', 'sac','tqc']:
    for run in range(strt_run, strt_run+n_runs):
        run_name = f'No_reactive_{algorithm}_run_{run}_{random.randint(0, 100000)}'
        command = 'tmux new-session -d \; send-keys "/home/sorfanouda/anaconda3/envs/dt/bin/python train_sb3.py' + \
            ' --algorithm ' + algorithm + \
            ' --device cuda:0' + \
            ' --run_name ' + str(run_name) + \
            '" Enter'
        os.system(command=command)
        print(command)
        # wait for 10 seconds before starting the next experiment
        time.sleep(2)
        # counter += 1