from rl_adn.environments.env import PowerNetEnv
import pandas as pd


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

for i in range(1):
    state = env.reset()
    done = False
    counter = 0
    while not done:
        print(f'=== Step {counter} ===')
        counter += 1
        #action is 34 values between -100 and 100, it is the power injection in each node
        action = [-10]*33       
        next_state, saved_energy, vm_pu_after_control_bat, done = env.step(action)
        print("State: ", state)
        print("Action: ", action)
        print("Next State: ", next_state)
        state = next_state
        input("Press Enter to continue...")