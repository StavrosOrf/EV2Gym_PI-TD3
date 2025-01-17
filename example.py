from rl_adn.environments.env import PowerNetEnv
import pandas as pd
import random

env_config = {
    "voltage_limits": [0.95, 1.05],
    "algorithm": "Laurent",
    "battery_list": [],
    "year": 2020,
    "month": 1,
    "day": 1,
    "train": True,
    "state_pattern": "default",
    "network_info": {'vm_pu': 1.0,
                     's_base': 1000,
                     'bus_info_file': './rl_adn/data_sources/network_data/node_34/Nodes_34.csv',
                     'branch_info_file': './rl_adn/data_sources/network_data/node_34/Lines_34.csv'},    
    "time_series_data_path": "./rl_adn/data_sources/time_series_data/34_node_time_series.csv"
}

env=PowerNetEnv(env_config)

env_args = {
    'env_name': 'PowerNetEnv',
    'state_dim': env.state_space.shape[0],
    'action_dim': env.action_space.shape[0],
    'if_discrete': False
}

print(env_args)

succesful_runs = 0
failed_runs = 0

for i in range(10000):
    state = env.reset()
    done = False
    counter = 0
    while not done:
        # print(f'=== Step {counter} ===')
        counter += 1
        #action is 34 values between -100 and 100, it is the power injection in each node
        action = [0]*33       
        # action[32] = 150
        action =random.sample(range(-1000, 1000), 33)
        next_state, saved_energy, done = env.step(action)
        # print("State: ", state)
        # print("Action: ", action)
        # print("Saved Energy: ", saved_energy)
        # print("Next State: ", next_state)
        
        if done:
            
            succesful_runs += 1
            break
        
        if any(next_state[1] < 0.95) or any(next_state[1] > 1.05):
            # print("Voltage limits exceeded")
            done = True
            failed_runs += 1
          
        state = next_state
    
    if i % 100 == 0:
        print(f' Succesful runs: {succesful_runs} Failed runs: {failed_runs}')