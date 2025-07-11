"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time
import random

counter = 0
n_runs = 2
strt_run = 1

for algorithm in ['td3', 'ppo','sac']:
    for run in range(strt_run, strt_run+n_runs):
        run_name = f'{algorithm}_run_{run}_{random.randint(0, 100000)}'
        command = 'tmux new-session -d \; send-keys "/home/sorfanouda/anaconda3/envs/dt/bin/python train_sb3.py' + \
            ' --algorithm ' + algorithm + \
            ' --device cuda:0' + \
            ' --seed ' + str(run) + \
            ' --run_name ' + str(run_name) + \
            '" Enter'
        os.system(command=command)
        print(command)
        # wait for 10 seconds before starting the next experiment
        time.sleep(2)
        # counter += 1