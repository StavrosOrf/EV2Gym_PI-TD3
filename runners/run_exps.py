"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time

# config = "v2g_grid_150_300.yaml"
config = "PST_V2G_ProfixMax_150_300.yaml"

learning_rate = 3e-5
# scenario = "v2g_profitmax"
# scenario = "grid_v2g_profitmax"
scenario = 'pst_v2g_profitmax'

counter = 0
batch_size = 64  # 256 # 512


for policy in ['pi_td3']:
    for lookahead_critic_reward in [3]:
        for critic in [True, False]:
            for K in [5, 20]:  # 512
                for seed in [9]:

                    extra_args = ''

                    if not critic:

                        extra_args = ' --disable_critic'

                    # command = 'tmux new-session -d \; send-keys " /home/sorfanoudakis/.conda/envs/dt/bin/python train_research.py' + \
                    command = 'tmux new-session -d \; send-keys " /home/sorfanouda/anaconda3/envs/dt/bin/python train.py' + \
                        ' --device cuda:0' + \
                        ' --scenario ' + scenario + \
                        ' --batch_size ' + str(batch_size) + \
                        ' --config ' + config + \
                        ' --lr ' + str(learning_rate) + \
                        ' --policy ' + policy + \
                        ' --seed ' + str(seed) + \
                        ' --K ' + str(K) + \
                        extra_args + \
                        ' --lookahead_critic_reward ' + str(lookahead_critic_reward) + \
                        ' --group_name "AblationTests_300"' + \
                        ' --name ' +\
                        f'LookaheadCriticReward={lookahead_critic_reward}_' + \
                        f'Critic={critic}_' + \
                        f'{policy}' + \
                        '_K=' + str(K) + \
                        '_batch_size=' + str(batch_size) + \
                        '_seed=' + str(seed) + \
                        '" Enter'
                    os.system(command=command)
                    print(command)
                    # wait for 20 seconds before starting the next experiment
                    time.sleep(5)
                    counter += 1
