from rl_adn.environments.env import PowerNetEnv
import pandas as pd
import random

env_config = {
    "voltage_limits": [0.95, 1.05],
    "algorithm": "Laurent",
    "timescale": 15,
    "episode_length": 5,
    "train": True,
    "network_info": {'vm_pu': 1.0,
                     's_base': 1000,
                     'bus_info_file': './rl_adn/data_sources/network_data/node_34/Nodes_34.csv',
                     'branch_info_file': './rl_adn/data_sources/network_data/node_34/Lines_34.csv'},
    "time_series_data_path": "./rl_adn/data_sources/time_series_data/34_node_time_series.csv"
}

env = PowerNetEnv(env_config)

env_args = {
    'env_name': 'PowerNetEnv',
}

print(env_args)

succesful_runs = 0
failed_runs = 0

for i in range(10000):
    hour = 0
    month = 7
    day = 17
    year = 2020
    hour = 0
    minute = 0
    

    date = (2020, month, day, hour, minute)
    state = env.reset(date)
    done = False
    counter = 0
    for i in range(env_config['episode_length']):

        # print(f'=== Step {counter} ===')
        counter += 1
        # action is 34 values between -100 and 100, it is the power injection in each node
        action = [0]*33
        # action[32] = 150
        # action =random.sample(range(-1000, 1000), 33)
        next_state, saved_energy = env.step(action)
        # print("State: ", state)
        # print("Action: ", action)
        # print("Saved Energy: ", saved_energy)
        # print("Next State: ", next_state)

        if i == env_config['episode_length'] - 2:
            exit()
            succesful_runs += 1
            break

        # if any(next_state[1] < 0.95) or any(next_state[1] > 1.05):
        #     print("Voltage limits exceeded step: ", counter)
        #     done = True
        #     failed_runs += 1

        state = next_state

    if i % 100 == 0:
        print(f' Succesful runs: {succesful_runs} Failed runs: {failed_runs}')
