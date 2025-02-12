'''
The following code is based on the implementation of https://github.com/ShengrenHou/RL-ADN from

Cite as:
Shengren Hou, Shuyi Gao, Weijie Xia, Edgar Mauricio Salazar Duque, Peter Palensky, Pedro P. Vergara,
RL-ADN: A high-performance Deep Reinforcement Learning environment for optimal Energy Storage Systems dispatch in active distribution networks, Energy and AI,
Volume 19, 2025, 100457, ISSN 2666-5468, https://doi.org/10.1016/j.egyai.2024.100457
'''

import copy as cp
import numpy as np
import pandapower as pp
import pandas as pd
import pickle

from ev2gym.models.grid_utility.grid_tensor import GridTensor
from ev2gym.models.grid_utility.grid_utils import create_pandapower_net
from ev2gym.models.data_augment import DataGenerator
import pkg_resources

from matplotlib import pyplot as plt


class PowerGrid():
    """
        Custom Environment for Power Network Management.

        The environment simulates a power network, and the agent's task is to
        manage this network by controlling the batteries attached to various nodes.

        """

    def __init__(self, env_config, date) -> None:

        config = env_config

        self.algorithm = config['pf_solver']
        self.network_info = config['network_info']
        self.s_base = self.network_info['s_base']

        network_bus_info = pd.read_csv(self.network_info['bus_info_file'])
        self.node_num = len((network_bus_info.NODES))

        # Conditional initialization of the distribution network based on the chosen algorithm
        if self.algorithm == "Laurent":
            # Logic for initializing with GridTensor
            self.net = GridTensor(self.network_info['bus_info_file'],
                                  self.network_info['branch_info_file'])
            self.net.Q_file = np.zeros(self.node_num-1)
            self.dense_Ybus = self.net._make_y_bus().toarray()

        elif self.algorithm == "PandaPower":
            # Logic for initializing with PandaPower
            self.net = create_pandapower_net(self.network_info)
        else:
            raise ValueError(
                "Invalid algorithm choice. Please choose 'Laurent' or 'PandaPower'.")

        assert config['timescale'] == 15, "Only 15 minutes timescale is supported with the simulate_grid=True !!!"

        data_generator = pkg_resources.resource_filename(
            'ev2gym', 'data/augmentor.pkl')
        # self.data_generator = pickle.load(open(data_generator, 'rb'))
        with open(data_generator, "rb") as f:
            self.data_generator = CustomUnpickler(f).load()

        # self.episode_length: int = 24 * 60 / self.data_manager.time_interval
        self.episode_length = config['simulation_length']

        self.reset(date)

    def reset(self, date) -> np.ndarray:
        """
        Reset the environment to its initial state and return the initial state.
        """

        hour = date.hour
        minute = date.minute
        time_slot = hour * 4 + minute // 15
        print(f'date and step: {date}, {time_slot}')
        self.current_step = 0
        self.load_data = self.data_generator.sample_data(n_buses=self.node_num,
                                                         n_steps=self.episode_length + 24,
                                                         start_day=date.weekday(),
                                                         start_step=time_slot,
                                                         )
        
        assert not np.isnan(self.load_data).any(), "There are nan values in the load_data"
            
        return self._build_state(), 0

    def _build_state(self) -> np.ndarray:
        """
        Builds the current state of the environment based on the current time and data from PowerDataManager.
        """
        obs = self._get_obs()
        active_power = np.array(
            list(obs['node_data']['active_power'].values()))
        # renewable_active_power = np.array(list(obs['node_data']['renewable_active_power'].values()))
        vm = np.array(list(obs['node_data']['voltage'].values()))

        return (active_power, vm, self.current_step)

    def _get_obs(self):


        if self.algorithm == "Laurent":
            # This is where bugs comes from, if we don't use copy, this slice is actually creating a view of originally data.
            active_power = cp.copy(self.load_data[self.current_step, :])

            self.active_power = (active_power)[1:self.node_num]
            reactive_power = np.zeros(self.node_num)

            self.solution = self.net.run_pf(active_power=self.active_power)

            obs = {'node_data': {'voltage': {},
                                 'active_power': {},
                                 'reactive_power': reactive_power
                                 }}

            # NODES[1-34], node_index[0-33]
            for node_index in range(len(self.net.bus_info.NODES)):
                if node_index == 0:
                    obs['node_data']['voltage'][f'node_{node_index}'] = 1.0
                    obs['node_data']['active_power'][f'node_{node_index}'] = 0.0
                else:
                    obs['node_data']['voltage'][f'node_{node_index}'] = abs(
                        self.solution['v'].T[node_index - 1]).squeeze()
                    obs['node_data']['active_power'][f'node_{node_index}'] = active_power[node_index - 1]

        else:
            active_power = cp.copy(self.load_data[self.current_step, :])
            active_power[0] = 0

            for bus_index in self.net.load.bus.index:
                self.net.load.p_mw[bus_index] = (
                    active_power[bus_index]) / self.s_base
                self.net.load.q_mvar[bus_index] = 0
            pp.runpp(self.net, algorithm='nr')
            v_real = self.net.res_bus["vm_pu"].values * \
                np.cos(np.deg2rad(self.net.res_bus["va_degree"].values))
            v_img = self.net.res_bus["vm_pu"].values * \
                np.sin(np.deg2rad(self.net.res_bus["va_degree"].values))
            v_result = v_real + 1j * v_img

            obs = {'node_data': {'voltage': {}, 'active_power': {}, 'reactive_power': {},
                                 'renewable_active_power': {}},
                   'battery_data': {'soc': {}}, 'aux': {}}

            for node_index in self.net.load.bus.index:
                bus_idx = self.net.load.at[node_index, 'bus']
                obs['node_data']['voltage'][f'node_{node_index}'] = self.net.res_bus.vm_pu.at[bus_idx]
                obs['node_data']['active_power'][f'node_{node_index}'] = active_power[node_index]
                obs['node_data']['reactive_power'][f'node_{node_index}'] = self.net.res_load.q_mvar[node_index]
                obs['node_data']['renewable_active_power'][f'node_{node_index}'] = renewable_active_power[
                    node_index]

        return obs

    def _runpf(self, action):
        '''apply action to battery charge/discharge, update the battery condition, excute power flow, update the network condition'''
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

            # for i, node_index in enumerate(self.battery_list):
            #     getattr(self, f"battery_{node_index}").step(action[i])
            #     self.net.load.p_mw[node_index] += getattr(
            #         self, f"battery_{node_index}").energy_change / 1000

            pp.runpp(self.net, algorithm='nr')
            vm_pu_after_control = cp.deepcopy(
                self.net.res_bus.vm_pu).to_numpy(dtype=float)
            # vm_pu_after_control_bat = vm_pu_after_control[self.battery_list]

            self.after_control = vm_pu_after_control
            power_imported_from_ex_grid_after = self.net.res_ext_grid['p_mw']
            saved_energy = power_imported_from_ex_grid_before - \
                power_imported_from_ex_grid_after

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

        # done = (self.current_step == self.episode_length - 1)
        self.current_step += 1

        new_state = self._build_state()

        return new_state, saved_energy


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "ev2gym.models.data_augment"
        return super().find_class(module, name)
