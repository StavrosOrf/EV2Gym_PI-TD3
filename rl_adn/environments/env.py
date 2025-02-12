import copy as cp
import random

import gym
import numpy as np
import pandapower as pp
import pandas as pd
from gym import spaces

from rl_adn.data_manager.data_manager import GeneralPowerDataManager
from rl_adn.utility.grid import GridTensor
from rl_adn.utility.utils import create_pandapower_net

class PowerNetEnv(gym.Env):
    """
        Custom Environment for Power Network Management.

        The environment simulates a power network, and the agent's task is to
        manage this network by controlling the batteries attached to various nodes.

        Attributes:
            voltage_limits (tuple): Limits for the voltage.
            algorithm (str): Algorithm choice. Can be 'Laurent' or 'PandaPower'.
            battery_list (list): List of nodes where batteries are attached.
            year (int): Current year in simulation.
            month (int): Current month in simulation.
            day (int): Current day in simulation.
            train (bool): Whether the environment is in training mode.
            state_pattern (str): Pattern for the state representation.
            network_info (dict): Information about the network.
            node_num (int): Number of nodes in the network.
            action_space (gym.spaces.Box): Action space of the environment.
            data_manager (GeneralPowerDataManager): Manager for the time-series data.
            episode_length (int): Length of an episode.
            state_length (int): Length of the state representation.
            state_min (np.ndarray): Minimum values for each state element.
            state_max (np.ndarray): Maximum values for each state element.
            state_space (gym.spaces.Box): State space of the environment.
            current_time (int): Current timestep in the episode.
            after_control (np.ndarray): Voltages after control is applied.

        Args:
            env_config_path (str): Path to the environment configuration file.

        """

    def __init__(self, env_config) -> None:
        """
         Initialize the PowerNetEnv environment.
         :param env_config_path: Path to the environment configuration file. Defaults to 'env_config.py'.
         :type env_config_path: str
         """
        config = env_config

        self.voltage_low_boundary = config['voltage_limits'][0]
        self.voltage_high_boundary = config['voltage_limits'][1]
        self.algorithm = config['algorithm']
        self.battery_list = config['battery_list']
        self.year = config['year']
        self.month = config['month']
        self.day = config['day']
        self.train = config['train']
        self.state_pattern = config['state_pattern']
        self.network_info = config['network_info']
        # network_info for building the network
        if self.network_info == 'None':
            print(
                'create basic 34 node IEEE network, when initial data is not identified')
            self.network_info = {'vm_pu': 1.0, 's_base': 1000,
                                 'bus_info_file': '../data_sources/network_data/node_34/Nodes_34.csv',
                                 'branch_info_file': '../data_sources/network_data/node_34/Lines_34.csv'}
            self.s_base = 1000
            self.node_num = 34
        else:
            self.s_base = self.network_info['s_base']
            network_bus_info = pd.read_csv(self.network_info['bus_info_file'])
            self.node_num = len((network_bus_info.NODES))
        # Conditional initialization of the distribution network based on the chosen algorithm
        if self.algorithm == "Laurent":
            # Logic for initializing with GridTensor
            self.net = GridTensor(self.network_info['bus_info_file'],
                                  self.network_info['branch_info_file'])
            self.net.Q_file = np.zeros(33)
            self.dense_Ybus = self.net._make_y_bus().toarray()

        elif self.algorithm == "PandaPower":
            # Logic for initializing with PandaPower
            self.net = create_pandapower_net(self.network_info)
        else:
            raise ValueError(
                "Invalid algorithm choice. Please choose 'Laurent' or 'PandaPower'.")

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.node_num-1, 1), dtype=np.float32)
        self.data_manager = GeneralPowerDataManager(
            config['time_series_data_path'])
        self.episode_length: int = 24 * 60 / self.data_manager.time_interval

        if self.state_pattern == 'default':
            self.state_length = (self.node_num)*2  + 2

        else:
            raise ValueError(
                "Invalid value for 'state_pattern'. Expected 'default' or define by yourself.")

        self.state_space = spaces.Box(
            low=-2, high=2, shape=(self.state_length,), dtype=np.float32)

    def reset(self) -> np.ndarray:
        """
        Reset the environment to its initial state and return the initial state.

        :return: The normalized initial state of the environment.
        :rtype: np.ndarray
        """
        self._reset_date()
        self._reset_time()
        return self._build_state()

    def _reset_date(self) -> None:
        """
        Resets the date for the next episode.
        """
        if self.train:
            self.year, self.month, self.day = random.choice(
                self.data_manager.train_dates)
        else:
            self.year, self.month, self.day = random.choice(
                self.data_manager.test_dates)

    def _reset_time(self) -> None:
        """
        Resets the time for the next episode.
        """
        self.current_time = 0

    def _build_state(self) -> np.ndarray:
        """
        Builds the current state of the environment based on the current time and data from PowerDataManager.

        Returns:
            normalized_state (np.ndarray): The current state of the environment, normalized between 0 and 1.
                The state includes the following variables:
                - Netload power
                - SOC (State of Charge) of the last battery in the battery list
                - Price of the energy
                - Time state of the day
                - Voltage from estimation
        """        
        obs = self._get_obs()
        active_power = np.array(list(obs['node_data']['active_power'].values()))
        # renewable_active_power = np.array(list(obs['node_data']['renewable_active_power'].values()))
        vm = np.array(list(obs['node_data']['voltage'].values()))
                        
        return (active_power, vm, self.current_time)

    def _get_obs(self):
        """
        Executes the power flow based on the chosen algorithm and returns the observations.

        Returns:
            dict: The observation dictionary containing various state elements.
        """
        if self.state_pattern == 'default':
            one_slot_data = self.data_manager.select_timeslot_data(
                self.year, self.month, self.day, self.current_time)

            if self.algorithm == "Laurent":
                # This is where bugs comes from, if we don't use copy, this slice is actually creating a view of originally data.
                active_power = cp.copy(one_slot_data[0:34])
                renewable_active_power = one_slot_data[34:68]
                self.active_power = (
                    active_power - renewable_active_power)[1:34]
                reactive_power = np.zeros(33)
                price = one_slot_data[-1]
                self.solution = self.net.run_pf(active_power=self.active_power)

                obs = {'node_data': {'voltage': {}, 'active_power': {}, 'reactive_power': {},
                                     'renewable_active_power': {}},
                       'battery_data': {'soc': {}}, 'price': {}, 'aux': {}}

                # NODES[1-34], node_index[0-33]
                for node_index in range(len(self.net.bus_info.NODES)):
                    if node_index == 0:
                        obs['node_data']['voltage'][f'node_{node_index}'] = 1.0
                        obs['node_data']['active_power'][f'node_{node_index}'] = 0.0
                        obs['node_data']['renewable_active_power'][f'node_{node_index}'] = 0.0
                    else:
                        obs['node_data']['voltage'][f'node_{node_index}'] = abs(
                            self.solution['v'].T[node_index - 1]).squeeze()
                        obs['node_data']['active_power'][f'node_{node_index}'] = active_power[node_index - 1]
                        obs['node_data']['renewable_active_power'][f'node_{node_index}'] = renewable_active_power[
                            node_index - 1]
                obs['price'] = price
            else:
                active_power = one_slot_data[0:34]
                active_power[0] = 0
                renewable_active_power = one_slot_data[34:68]
                renewable_active_power[0] = 0
                price = one_slot_data[-1]
                for bus_index in self.net.load.bus.index:
                    self.net.load.p_mw[bus_index] = (active_power[bus_index] - renewable_active_power[
                        bus_index]) / self.s_base
                    self.net.load.q_mvar[bus_index] = 0
                pp.runpp(self.net, algorithm='nr')
                v_real = self.net.res_bus["vm_pu"].values * \
                    np.cos(np.deg2rad(self.net.res_bus["va_degree"].values))
                v_img = self.net.res_bus["vm_pu"].values * \
                    np.sin(np.deg2rad(self.net.res_bus["va_degree"].values))
                v_result = v_real + 1j * v_img

                obs = {'node_data': {'voltage': {}, 'active_power': {}, 'reactive_power': {},
                                     'renewable_active_power': {}},
                       'battery_data': {'soc': {}}, 'price': {}, 'aux': {}}

                for node_index in self.net.load.bus.index:
                    bus_idx = self.net.load.at[node_index, 'bus']
                    obs['node_data']['voltage'][f'node_{node_index}'] = self.net.res_bus.vm_pu.at[bus_idx]
                    obs['node_data']['active_power'][f'node_{node_index}'] = active_power[node_index]
                    obs['node_data']['reactive_power'][f'node_{node_index}'] = self.net.res_load.q_mvar[node_index]
                    obs['node_data']['renewable_active_power'][f'node_{node_index}'] = renewable_active_power[
                        node_index]

                obs['price'] = price
        else:
            raise ValueError(
                'please redesign the get obs function to fit the pattern you want')
        return obs

    def _runpf(self, action):
        '''apply action to battery charge/discharge, update the battery condition, excute power flow, update the network condition'''
        if self.state_pattern == 'default':
            if self.algorithm == "Laurent":
                v = self.solution["v"]
                v_totall = np.insert(v, 0, 1)
                current_each_node = np.matmul(self.dense_Ybus, v_totall)
                power_imported_from_ex_grid_before = current_each_node[0].real

                self.active_power += action

                self.solution = self.net.run_pf(active_power=self.active_power)

                v = self.solution["v"]
                v_totall = np.insert(v, 0, 1)
                vm_pu_after_control = cp.deepcopy(abs(v_totall))
                self.after_control = vm_pu_after_control
                current_each_node = np.matmul(self.dense_Ybus, v_totall)
                power_imported_from_ex_grid_after = current_each_node[0].real
                saved_energy = power_imported_from_ex_grid_before - \
                    power_imported_from_ex_grid_after
            else:
                power_imported_from_ex_grid_before = cp.deepcopy(
                    self.net.res_ext_grid['p_mw'])

                for i, node_index in enumerate(self.battery_list):
                    getattr(self, f"battery_{node_index}").step(action[i])
                    self.net.load.p_mw[node_index] += getattr(
                        self, f"battery_{node_index}").energy_change / 1000
                pp.runpp(self.net, algorithm='nr')
                vm_pu_after_control = cp.deepcopy(
                    self.net.res_bus.vm_pu).to_numpy(dtype=float)
                vm_pu_after_control_bat = vm_pu_after_control[self.battery_list]

                self.after_control = vm_pu_after_control
                power_imported_from_ex_grid_after = self.net.res_ext_grid['p_mw']
                saved_energy = power_imported_from_ex_grid_before - \
                    power_imported_from_ex_grid_after
        else:
            raise ValueError(
                'Expected default or define yourself based on the goal')
        return saved_energy

    def step(self, action: np.ndarray) -> tuple:
        """
        Advance the environment by one timestep based on the provided action.

        :param action: Action to execute.
        :type action: np.ndarray
        :return: Tuple containing the next normalized observation, the reward, a boolean indicating if the episode has ended, and additional info.
        :rtype: tuple
        """
        # Apply battery actions and get updated observations
        saved_energy = self._runpf(action)

        done = (self.current_time == self.episode_length - 1)
        self.current_time += 1
        if done:
            self.current_time = 0
            new_state = self.reset()
        else:
            new_state = self._build_state()
        return new_state, saved_energy, done
