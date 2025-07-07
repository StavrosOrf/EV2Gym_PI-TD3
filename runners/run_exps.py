"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time

config = "v2g_grid_150_300.yaml"
# config = "PST_V2G_ProfixMax_150_300.yaml"

learning_rate = 3e-5
# scenario = "v2g_profitmax"
scenario = "grid_v2g_profitmax"
# scenario = 'pst_v2g_profitmax'

counter = 0
batch_size = 64  # 256 # 512
N_agents = 24


# for policy in ['pi_td3', 'sapo_op', 'shac_op', 'pi_sac', 'shac','sapo','td3', 'sac']:
for policy in ['pi_ppo']:
    for lookahead_critic_reward in [3]:
        for enable_entropy in [True, False]:
            for critic_update_method in ['td_lambda', 'soft_td_lambda']:
                if not enable_entropy and critic_update_method == 'soft_td_lambda':
                    continue
                for actor_update_steps in [4]:  # , 4, 8]:
                    for critic in [True]:
                        for K in [20]:  # 512
                            for seed in [9]:

                                if policy == 'pi_td3':
                                    lookahead_critic_reward = 3
                                elif policy == 'pi_sac':
                                    lookahead_critic_reward = 4

                                extra_args = ''

                                if not critic:
                                    extra_args = ' --disable_critic'

                                if enable_entropy:
                                    extra_args += ' --enable_entropy'

                                # command = 'tmux new-session -d \; send-keys " /home/sorfanoudakis/.conda/envs/dt/bin/python train_research.py' + \
                                command = 'tmux new-session -d \; send-keys " /home/sorfanouda/anaconda3/envs/dt/bin/python train.py' + \
                                    ' --device cuda:0' + \
                                    ' --scenario ' + scenario + \
                                    ' --batch_size ' + str(batch_size) + \
                                    ' --config ' + config + \
                                    ' --lr ' + str(learning_rate) + \
                                    ' --policy ' + policy + \
                                    ' --seed ' + str(seed) + \
                                    ' --critic_update_method ' + critic_update_method + \
                                    ' --actor_update_steps ' + str(actor_update_steps) + \
                                    ' --N_agents ' + str(N_agents) + \
                                    ' --K ' + str(K) + \
                                    ' --disable_development_mode' + \
                                    extra_args + \
                                    ' --lookahead_critic_reward ' + str(lookahead_critic_reward) + \
                                    ' --group_name "NewModels_AblationTests_300"' + \
                                    ' --name ' +\
                                    f'Entropy={enable_entropy}_' + \
                                    f'CriticUpdateMethod={critic_update_method}_' + \
                                    f'ActorUpdateSteps={actor_update_steps}_' + \
                                    f'{policy}_' + \
                                    f'LookaheadCriticReward={lookahead_critic_reward}_' + \
                                    f'Critic={critic}_' + \
                                    'K=' + str(K) + \
                                    '_seed=' + str(seed) + \
                                    '" Enter'
                                os.system(command=command)
                                print(command)
                                # wait for 20 seconds before starting the next experiment
                                time.sleep(5)
                                counter += 1
