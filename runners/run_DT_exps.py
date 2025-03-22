"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time
import random

# run train_DT.py in a tmux pane for each K and dataset

# batch_size = 64
num_steps_per_iter = 500
max_iters = 350
num_eval_episodes = 100
seed = 42

counter = 0
for model_type in ["dt"]:
    for lr in [1e-4]:
        for physics_loss_weight in [0.0000001,0.1, 100]:        
        # for physics_loss_weight in [0, 0.1, 100, 0.001]:        
            for K in [2, 10]:
                for batch_size in [128]:
                    for dataset in ["random_10000"]:
                        for embed_dim in [128]:  # 128, 512
                            #   ' --device cuda:0' + str(counter % 2) + \
                            for n_layer, n_head in [(3, 4)]:  # (3, 1),(3,4)

                                run_name = f'{model_type}_ph_coeff={physics_loss_weight}_run_{seed}_K={K}_batch={batch_size}_dataset={dataset}_embed_dim={embed_dim}_n_layer={n_layer}_n_head={n_head}'
                                run_name += str(random.randint(0, 100000))
                                
                                command = 'tmux new-session -d \; send-keys " /home/sorfanouda/anaconda3/envs/dt/bin/python train_DT_research.py' + \
                                    ' --dataset ' + dataset + \
                                    ' --K ' + str(K) + \
                                    ' --device cuda:0' + \
                                    ' --seed ' + str(seed) + \
                                    ' --model_type ' + model_type + \
                                    ' --embed_dim ' + str(embed_dim) + \
                                    ' --n_layer ' + str(n_layer) + \
                                    ' --n_head ' + str(n_head) + \
                                    ' --max_iters=' + str(max_iters) + \
                                    ' --batch_size=' + str(batch_size) + \
                                    ' --num_steps_per_iter=' + str(num_steps_per_iter) + \
                                    ' --num_eval_episodes=' + str(num_eval_episodes) + \
                                    ' --learning_rate=' + str(lr) + \
                                    ' --log_to_wandb False' + \
                                    ' --group_name ' + 'phys' + \
                                    ' --physics_loss_weight ' + str(physics_loss_weight) + \
                                    ' --name UpdatedLossSign' +  str(run_name) + \
                                    '" Enter'
                                os.system(command=command)
                                print(command)
                                # wait for 20 seconds before starting the next experiment
                                time.sleep(5)
                                counter += 1