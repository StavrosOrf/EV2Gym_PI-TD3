from rl_adn.data_augment.data_augment import ActivePowerDataManager, TimeSeriesDataAugmentor

import pickle
import pandas as pd
import time

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
hour = 0
month = 1
day = 1
year = 2020
hour = 0
minute = 0
date = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute, second=0, tz='UTC')

data_manager = ActivePowerDataManager(
    env_config['time_series_data_path'])

timer_start = time.time()
augmentor = TimeSeriesDataAugmentor(data_manager,
                                      augmentation_model_name='GMM')
print(f'time to fit model: {time.time() - timer_start}')

pickle.dump(augmentor, open('augmentor.pkl', 'wb'))

# load the model
timer_start = time.time()
augmentor = pickle.load(open('augmentor.pkl', 'rb'))
print(f'time to load model: {time.time() - timer_start}')


timer_start = time.time()
augmentor.augment_data(num_nodes=34, 
                       num_days = 1,
                       start_date=date)

print(f'time to augment data: {time.time() - timer_start}')