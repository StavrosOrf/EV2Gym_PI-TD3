import wandb
import pandas as pd
import numpy as np
import tqdm as tqdm
# Login to W&B if not already logged in
# wandb.login()

# Initialize API
import os
os.environ['WANDB_HTTP_TIMEOUT'] = '300'

api = wandb.Api()

# Replace 'your_project_name' and 'your_entity_name' with your actual project and entity
project_name = "EVs4Grid_PES"
entity_name = "stavrosorf"
group_name = "grid_v2g_profitmax_PI_RL_grid_v2g_profitmax_150cs_-1tr"


# Fetch runs from the specified project
runs = api.runs(f"{entity_name}/{project_name}")
print(f"Total runs fetched: {len(runs)}")

runs = [run for run in runs if run.group == group_name]
print(f"Total runs fetched: {len(runs)}")

# Display the filtered runs with group names

run_results = []
# use tqdm to display a progress bar
for i, run in tqdm.tqdm(enumerate(runs), total=len(runs)):

    group_name = run.group

    if "grid" in group_name:
        group_name = 'grid'
    elif "v2g_profitmax" in group_name:
        group_name = 'v2g_profitmax'
    elif "v2g" in group_name:
        group_name = 'v2g'

    history = run.history()

    if np.array(history["_runtime"])[-1]/3600 < 1:
        continue

    config = run.config
    name = run.name
    print(f"Processing run {i+1}/{len(runs)}: {name} - Group: {group_name}")

    if "pi_td3" in name:
        algorithm = "PI-TD3"
    elif "pi_sac" in name:
        algorithm = "PI-SAC"
    else:
        algorithm = name.split("_")[0]

    if algorithm in ["PI-TD3", "PI-SAC"]:
        K = name.split("_")[4].split("K=")[1]
        seed = name.split("_")[3]
        continue
    else:
        K = name.split("_")[3].split("K=")[1]
        seed = name.split("_")[2]

    if '_runtime' not in history:
        print(f"Run {run.id} has no _runtime key")
        continue

    history = run.scan_history(keys=[
        "_runtime",
        "eval_a/mean_reward",
        "eval_a/best_reward",
        "eval/total_profits",
        "eval/voltage_violation",
        "eval/average_user_satisfaction",
    ])
    
    history_df = pd.DataFrame(history)
    

    mean_rewards = history_df['eval_a/mean_reward'].dropna().tolist()
    best_rewards = history_df['eval_a/best_reward'].dropna().tolist()
    eval_profits = history_df['eval/total_profits'].dropna().tolist()
    eval_voltage_violation = history_df['eval/voltage_violation'].dropna().tolist()
    eval_user_satisfaction = history_df['eval/average_user_satisfaction'].dropna(
    ).tolist()

    results = {
        "algorithm": algorithm,
        "group": group_name,
        "K": K,
        "seed": seed,
        "runtime": history_df["_runtime"].iloc[-1] / 3600,
        "best": np.array(best_rewards)[-1],
        "best_reward": np.array(best_rewards),
        "mean_rewards": np.array(mean_rewards),
        "eval_profits": np.array(eval_profits),
        "eval_voltage_violation": np.array(eval_voltage_violation),
        "eval_user_satisfaction": np.array(eval_user_satisfaction),
    }
    run_results.append(results)
    
    # break


# Convert the results to a pandas DataFrame
df = pd.DataFrame(run_results)
print(df.head())
print(df.shape)

print(df.describe())

print(df["algorithm"].value_counts())
print(df["K"].value_counts())
print(df["seed"].value_counts())

# Save the results to a CSV file
df.to_csv("./results_analysis/results_full.csv",
          index=False)
print("Results saved to results.csv")
