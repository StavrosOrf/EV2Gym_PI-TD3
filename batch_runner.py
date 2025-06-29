'''
This script is used to run the batch of training sciprts for every algorithms evaluated
#old command# srun --mpi=pmix --job-name=interactive-gpu --partition=gpu --gres=gpu:1 --qos=normal --time=01:00:00 --mem-per-cpu=4096 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=5300 --cpus-per-task=3 --ntasks=1 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu-a100-small --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=7000 --cpus-per-task=1 --ntasks=1 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu-a100 --gpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=7000 --cpus-per-task=1 --ntasks=1 --pty /bin/bash -il
srun --mpi=pmix --job-name=interactive --partition=compute --cpus-per-task=1 --qos=normal --time=01:00:00 --mem-per-cpu=4G--ntasks=1 --pty /bin/bash -il
'''
import os
import random

seeds = [0]

batch_size = 64
N_agents = 16

gpu = 'gpu' #gpu-a100 # gpu-a100-small # gpu

# if directory does not exist, create it
if not os.path.exists('./slurm_logs'):
    os.makedirs('./slurm_logs')

# td3, sac, pi_sac, pi_td3, shac
# for algo in ['pi_td3', 'sapo_op', 'shac_op', 'pi_sac', 'shac','sapo','td3', 'sac']:
for algo in ['pi_td3', 'sapo_op', 'shac_op', 'pi_sac','td3', 'sac', 'shac','sapo']:
    # for K in [30,80]:
    for K in [1, 5, 10, 20, 40]:
        for scenario in [
                        #  'v2g_profitmax',
                        #  'grid_v2g_profitmax',
                         'pst_v2g_profitmax'
                         ]:
            
            if 'pst' in scenario:
                config = "PST_V2G_ProfixMax_150_300.yaml"
            else:
                config = "v2g_grid_150_300.yaml"
            
            for lookahead_critic_reward in [0]: # 2 is the default value
                
                if algo == 'pi_td3':
                    lookahead_critic_reward = 3
                elif algo == 'pi_sac':
                    lookahead_critic_reward = 4

                for critic_enabled in [True]:
                    for counter, seed in enumerate(seeds):

                        if K != 1 and algo in ['sac','td3']:
                            continue           
                        elif K == 1:            
                            continue

                        if K <= 10:
                            time = 15
                        # elif K <= 20:
                        #     time = 24
                        # elif K <= 30:
                        #     time = 24
                        else:
                            time = 23

                        if K <= 10:
                            if algo in ['pi_td3', 'pi_sac']:
                                cpu_cores = 3
                            else:
                                cpu_cores = 2
                            
                        elif K <= 20:
                            cpu_cores = 3
                        else:
                            cpu_cores = 4
                                                        

                        if time > 46:
                            time = 46

                        memory = 5300

                        run_name = f'{algo}_run_{seed}_K={K}_scenario={scenario}_'
                        run_name += str(random.randint(0, 100000))
                        
                        if not critic_enabled:

                            extra_args = ' --disable_critic'
                        else:
                            extra_args = ''
                            
                        # gpu-a100, gpu
                        command = '''#!/bin/sh
#!/bin/bash
#SBATCH --job-name="pi_rl"
''' + \
                    f'#SBATCH --partition={gpu}\n' + \
                    f'#SBATCH --time={time}:00:00' + \
                    '''
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
''' + \
                    f'#SBATCH --cpus-per-task={cpu_cores}' + \
                    '''
''' + \
                    f'#SBATCH --mem-per-cpu={memory}' + \
                    '''
#SBATCH --account=research-eemcs-ese

''' + \
                    f'#SBATCH --output=./slurm_logs/{run_name}.out' + \
                    '''
''' + \
                    f'#SBATCH --error=./slurm_logs/{run_name}.err' + \
                    '''

module load 2024r1 openmpi miniconda3 py-pip

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate dt3
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

''' + 'srun python train.py' + \
                    ' --scenario ' + scenario + \
                    ' --K ' + str(K) + \
                    ' --device cuda:0' + \
                    ' --policy ' + algo + \
                    ' --group_name ' + '"NewModels_AblationTests_300"' + \
                    ' --seed ' + str(seed) + \
                    ' --disable_development_mode' + \
                    ' --lookahead_critic_reward ' + str(lookahead_critic_reward) + \
                    ' --N_agents ' + str(N_agents) + \
                    ' --batch_size ' + str(batch_size) + \
                    ' --project_name ' + '"EVs4Grid"' + \
                    ' --config ' + config + \
                    ' --name ' + str(run_name) + \
                    extra_args + \
                    '' + \
                    '''
            
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate
'''

                        with open(f'run_tmp.sh', 'w') as f:
                            f.write(command)

                        with open(f'./slurm_logs/{run_name}.sh', 'w') as f:
                            f.write(command)

                        os.system('sbatch run_tmp.sh')

