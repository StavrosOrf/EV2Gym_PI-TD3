import wandb
import pandas as pd
import numpy as np
import tqdm as tqdm
# Login to W&B if not already logged in
# wandb.login()

# Initialize API
api = wandb.Api(timeout=120)

# Replace 'your_project_name' and 'your_entity_name' with your actual project and entity
project_name = "EVs4Grid_Exps"
entity_name = "stavrosorf"
group_name = "ProofExps_grid_v2g_profitmax_150cs_-1tr"


# Fetch runs from the specified project
runs = api.runs(f"{entity_name}/{project_name}")
print(f"Total runs fetched: {len(runs)}")

runs = [run for run in runs if run.group == group_name]
print(f"Total runs fetched: {len(runs)}")

# Display the filtered runs with group names

run_results = []
# use tqdm to display a progress bar
for i, run in tqdm.tqdm(enumerate(runs), total=len(runs)):
    # if i < 100:
    #     continue

    group_name = run.group
    history = run.history()

    # print(f"History keys: {history.keys()}")
    # print(f'runtime: {np.array(history["_runtime"])[-1]/3600}')
    
    if np.array(history["_runtime"])[-1]/3600 < 1:
        continue
    
    # print(f"_runtime: {history['_runtime']}")
    # print(f"_timestamp: {history['_timestamp']}")
    # print(f"_step: {history['_step']}")
    # exit()
    # History keys: Index(['best', 'time/total', 'time/training', 'opt/total_transformer_overload',
    #    'test/total_energy_charged', '_step', 'time/evaluation',
    #    'test/total_profits', 'opt/average_user_satisfaction', '_timestamp',
    #    'training/action_error', 'opt/total_reward',
    #    'opt/power_tracker_violation', 'test/total_transformer_overload',
    #    'test/power_tracker_violation', '_runtime',
    #    'test/average_user_satisfaction', 'training/train_loss_std',
    #    'training/train_loss_mean', 'test/min_user_satisfaction',
    #    'opt/min_user_satisfaction', 'opt/total_profits', 'test/total_reward',
    #    'opt/total_energy_charged', 'opt/total_energy_discharged',
    #    'test/total_energy_discharged'],
    #   dtype='object')
#  clarify the algorithm used in the run, and the dataset used
    config = run.config
    # print(f'config: {config}')
    # print(f'run: {run}')
    name = run.name
    print(f'run name: {name}')

    if "mb_traj" in name:        
        algorithm = "MB-TD3"
    else:
        algorithm = name.split("_")[0]
    print(f'algorithm: {algorithm}')
    
    if algorithm == "MB-TD3":
        K = name.split("_")[4].split("K=")[1]
        seed = name.split("_")[3]
    else:
        K = name.split("_")[3].split("K=")[1]
        seed = name.split("_")[2]

    # print(f'K: {K}')
    # print(f'seed: {seed}')

    if '_runtime' not in history:
        print(f"Run {run.id} has no _runtime key")
        continue
        
    # best_reward = history["eval_a/best_reward"]    
    # best_reward = [None if np.isnan(x) else x for x in best_reward]
    # best_reward = [x for x in best_reward if x != None]
    
    # print(f'{run.name} mean_rewards: {len(best_reward)}')
    # print(f'{run.name} best_rewards: { max(best_reward)}')
    
    # Convert history to a DataFrame for easier manipulation
    history = run.scan_history()
    history = pd.DataFrame(history)
    print(f"History: {history}")
    
    mean_rewards = history["eval_a/mean_reward"]
    print(f'{run.name} mean_rewards: {len(mean_rewards)}')
    print(f'{run.name} mean_rewards: {mean_rewards}')
    mean_rewards = [None if np.isnan(x) else x for x in mean_rewards]
    print(f'{run.name} mean_rewards: {len(mean_rewards)}')
    print(f'{run.name} mean_rewards: {mean_rewards}')
    mean_rewards = [x for x in mean_rewards if x != None]
    
    print(f'{run.name} mean_rewards: {len(mean_rewards)}')
    print(f'{run.name} mean_rewards: { max(mean_rewards)}')

    exit()
    
    results = {
        "algorithm": algorithm,
        "K": K,
        "seed": seed,
        "runtime": np.array(history["_runtime"])[-1]/3600,
        "best": np.array(history["eval_a/best_reward"])[-1],
        "best_reward": np.array(history["eval_a/best_reward"]),
        "eval_reward": np.array(history["eval/total_reward"]),
        "eval_profits": np.array(history["eval/total_profits"]),
        "eval_voltage_violation": np.array(history["eval/voltage_violation"]),
        "eval_user_satisfaction": np.array(history["eval/average_user_satisfaction"]),
    }
    # print(f'results_rewards: {results["eval_reward"][:40]}')
    # print(f'results_profits: {results["eval_profits"][:40]}')
    # print(f'results_power_tracker_violation: {results["eval_power_tracker_violation"][:40]}')
    # print(f'results_user_satisfaction: {results["eval_user_satisfaction"][:40]}')
    # input("Press")

    run_results.append(results)
    # exit()

    # if i > 102:
    #     break

# Convert the results to a pandas DataFrame
df = pd.DataFrame(run_results)
print(df.head())
print(df.shape)

print(df.describe())

print(df["algorithm"].value_counts())
print(df["K"].value_counts())
print(df["seed"].value_counts())

# Save the results to a CSV file
df.to_csv("./results_analysis/results.csv",
          index=False)
print("Results saved to results.csv")
