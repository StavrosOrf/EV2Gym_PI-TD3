'''
The following code is based on the implementation of https://github.com/ShengrenHou/RL-ADN

Cite as:
Shengren Hou, Shuyi Gao, Weijie Xia, Edgar Mauricio Salazar Duque, Peter Palensky, Pedro P. Vergara,
RL-ADN: A high-performance Deep Reinforcement Learning environment for optimal Energy Storage Systems dispatch in active distribution networks, Energy and AI,
Volume 19, 2025, 100457, ISSN 2666-5468, https://doi.org/10.1016/j.egyai.2024.100457
'''

import copy as cp
import numpy as np
from numpy.linalg import eigvals
import pandapower as pp
import pandas as pd
import pickle

from ev2gym.models.grid_utility.grid_tensor import GridTensor
from ev2gym.models.grid_utility.grid_utils import create_pandapower_net
from ev2gym.models.data_augment import DataGenerator
import pkg_resources


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

        with open(data_generator, "rb") as f:
            self.data_generator = CustomUnpickler(f).load()

        # self.episode_length: int = 24 * 60 / self.data_manager.time_interval
        self.episode_length = config['simulation_length']

        self.reset(date, None)

    def reset(self, date, load_data) -> np.ndarray:
        """
        Reset the environment to its initial state and return the initial state.
        """

        hour = date.hour
        minute = date.minute
        time_slot = hour * 4 + minute // 15
        self.current_step = 0

        if load_data is not None:
            self.load_data = load_data
        else:
            self.load_data = self.data_generator.sample_data(n_buses=self.node_num,
                                                             n_steps=self.episode_length + 24,
                                                             start_day=date.weekday(),
                                                             start_step=time_slot,
                                                             )

        return *self._build_state(), 0

    def _build_state(self) -> np.ndarray:
        """
        Builds the current state of the environment based on the current time and data from PowerDataManager.
        """

        obs = {'node_data': {'voltage': {},
                             'active_power': {},
                             #  'reactive_power': {}
                             }}

        if self.algorithm == "Laurent":

            active_power = cp.copy(self.load_data[self.current_step, :])
            self.active_power = (active_power)[1:self.node_num].reshape(1, -1)
            # reactive_power = np.zeros(self.node_num)
            self.reactive_power = self.active_power * 0

            self.solution = self.net.run_pf(active_power=self.active_power,
                                            # reactive_power=self.reactive_power
                                            )

            # NODES[1-NODES], node_index[0-(NODES-1)]
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
                # self.net.load.p_mw[bus_index] = (
                #     active_power[bus_index]) / self.s_base
                # self.net.load.q_mvar[bus_index] = 0
                self.net.load.loc[bus_index,
                                  "p_mw"] = active_power[bus_index] / self.s_base
                self.net.load.loc[bus_index, "q_mvar"] = 0

            pp.runpp(self.net, algorithm='nr')
            v_real = self.net.res_bus["vm_pu"].values * \
                np.cos(np.deg2rad(self.net.res_bus["va_degree"].values))
            v_img = self.net.res_bus["vm_pu"].values * \
                np.sin(np.deg2rad(self.net.res_bus["va_degree"].values))
            v_result = v_real + 1j * v_img

            for node_index in self.net.load.bus.index:
                bus_idx = self.net.load.at[node_index, 'bus']
                obs['node_data']['voltage'][f'node_{node_index}'] = self.net.res_bus.vm_pu.at[bus_idx]
                obs['node_data']['active_power'][f'node_{node_index}'] = active_power[node_index]
                # obs['node_data']['reactive_power'][f'node_{node_index}'] = self.net.res_load.q_mvar[node_index]

        # return obs
        active_power = np.array(
            list(obs['node_data']['active_power'].values()))
        # renewable_active_power = np.array(list(obs['node_data']['renewable_active_power'].values()))
        vm = np.array(list(obs['node_data']['voltage'].values()))

        return active_power, vm

    def _runpf(self, action):

        if self.algorithm == "Laurent":
            v = self.solution["v"]
            v_totall = np.insert(v, 0, 1)
            current_each_node = np.matmul(self.dense_Ybus, v_totall)
            power_imported_from_ex_grid_before = current_each_node[0].real

            self.active_power += action
            # my_v = self.calculate_PF(self.dense_Ybus, v_totall, self.active_power)

            print(f'self.net.z_base: {self.net.z_base}')
            self.Z = np.linalg.inv(self.dense_Ybus[1:, 1:]) / self.net.z_base
            active_power_pu = self.active_power / self.s_base
            # Vector with all reactive power except slack
            reactive_power_pu = self.reactive_power / self.s_base

            self.S = active_power_pu + 1j * reactive_power_pu

            # v_approx = laurent_voltage_adaptive(np.array([1+0j], dtype=complex),
            #                                     self.Z,
            #                                     self.S,
            #                                     self.dense_Ybus[1:, 1:]*self.net.z_base,
            #                                     )
            v_prev = self.solution['v']
            v_prev = np.insert(v_prev, 0, 1).real
            v_approx, angle = continuation_power_flow(Ybus=self.dense_Ybus,
                                                      P_spec=np.insert(
                                                          active_power_pu, 0, 0),
                                                      Q_spec=np.insert(
                                                          reactive_power_pu, 0, 0),
                                                      slack_index=0,
                                                      V_slack=1,
                                                      theta_slack=0,)

            #      (np.array([1+0j], dtype=complex),
            # self.dense_Ybus,
            # np.zeros(self.node_num,dtype=complex),
            # v_prev,
            # )

            v_approx = v_approx.real
            self.solution = self.net.run_pf(active_power=self.active_power,
                                            # reactive_power=self.reactive_power
                                            )

            solution_v = self.solution["v"].real
            # solution_v = np.insert(solution_v, 0, 1)
            print(f'True v: {solution_v}')
            print(f'Approx: {v_approx}')
            print(f'v error: {np.linalg.norm(v_approx - solution_v)}')
            input()

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

            pp.runpp(self.net, algorithm='nr')
            vm_pu_after_control = cp.deepcopy(
                self.net.res_bus.vm_pu).to_numpy(dtype=float)

            self.after_control = vm_pu_after_control
            power_imported_from_ex_grid_after = self.net.res_ext_grid['p_mw']
            saved_energy = power_imported_from_ex_grid_before - \
                power_imported_from_ex_grid_after

        return saved_energy

    def step(self, actions: np.ndarray) -> tuple:
        """
        Advance the environment by one timestep based on the provided action.
        """

        # Update active power of each node based on EVs and run power flow
        saved_energy = self._runpf(actions)
        self.current_step += 1
        active_power, vm = self._build_state()

        return active_power, vm, saved_energy


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "ev2gym.models.data_augment"
        return super().find_class(module, name)


def newton_raphson_power_flow(Ybus, P_spec, Q_spec, slack_index, V_slack, theta_slack,
                              V_init, theta_init, tol=1e-6, max_iter=50, damping=0.7):
    """
    Solve the full AC power flow equations in polar coordinates using Newton–Raphson.

    For non-slack buses (assumed PQ buses), the equations are:
      P_i = V_i * sum_{k=1}^{N} V_k [G_ik cos(theta_i-theta_k) + B_ik sin(theta_i-theta_k)]
      Q_i = V_i * sum_{k=1}^{N} V_k [G_ik sin(theta_i-theta_k) - B_ik cos(theta_i-theta_k)]

    Mismatches:
      ΔP_i = P_spec[i] - P_i
      ΔQ_i = Q_spec[i] - Q_i

    The unknowns are the voltage angles and magnitudes for non-slack buses.

    Parameters
    ----------
    Ybus : ndarray, shape (N, N)
        Full bus admittance matrix (in per unit).
    P_spec : ndarray, shape (N,)
        Specified real power injections (p.u.) for all buses.
    Q_spec : ndarray, shape (N,)
        Specified reactive power injections (p.u.) for all buses.
    slack_index : int
        Index of the slack bus.
    V_slack : float
        Voltage magnitude at the slack bus (p.u.).
    theta_slack : float
        Voltage angle at the slack bus (radians).
    V_init : ndarray, shape (N,)
        Initial guess for bus voltage magnitudes.
    theta_init : ndarray, shape (N,)
        Initial guess for bus voltage angles.
    tol : float, optional
        Convergence tolerance.
    max_iter : int, optional
        Maximum iterations.
    damping : float, optional
        Damping factor for updates.

    Returns
    -------
    V : ndarray, shape (N,)
        Converged voltage magnitudes (p.u.).
    theta : ndarray, shape (N,)
        Converged voltage angles (radians).
    """
    N = Ybus.shape[0]
    G = np.real(Ybus)
    B = np.imag(Ybus)

    V = V_init.copy()
    theta = theta_init.copy()
    # Fix slack bus:
    V[slack_index] = V_slack
    theta[slack_index] = theta_slack

    non_slack = [i for i in range(N) if i != slack_index]

    for iteration in range(max_iter):
        dP = np.zeros(N)
        dQ = np.zeros(N)
        # Compute calculated P and Q at each bus
        for i in range(N):
            sum_P = 0.0
            sum_Q = 0.0
            for k in range(N):
                angle_diff = theta[i] - theta[k]
                sum_P += V[k] * (G[i, k] * np.cos(angle_diff) +
                                 B[i, k] * np.sin(angle_diff))
                sum_Q += V[k] * (G[i, k] * np.sin(angle_diff) -
                                 B[i, k] * np.cos(angle_diff))
            P_calc = V[i] * sum_P
            Q_calc = V[i] * sum_Q
            dP[i] = P_spec[i] - P_calc
            dQ[i] = Q_spec[i] - Q_calc

        # Assemble mismatch vector for non-slack buses:
        F = []
        for i in non_slack:
            F.append(dP[i])
        for i in non_slack:
            F.append(dQ[i])
        F = np.array(F)

        if np.linalg.norm(F, np.inf) < tol:
            # Convergence reached.
            return V, theta

        n_ns = len(non_slack)
        J11 = np.zeros((n_ns, n_ns))  # dP/dtheta
        J12 = np.zeros((n_ns, n_ns))  # dP/dV
        J21 = np.zeros((n_ns, n_ns))  # dQ/dtheta
        J22 = np.zeros((n_ns, n_ns))  # dQ/dV

        # Build Jacobian for each non-slack bus:
        for idx_i, i in enumerate(non_slack):
            for idx_k, k in enumerate(non_slack):
                if i == k:
                    sum_term = 0.0
                    for m in range(N):
                        if m == i:
                            continue
                        angle_diff = theta[i] - theta[m]
                        sum_term += V[m] * (-G[i, m]*np.sin(angle_diff) +
                                            B[i, m]*np.cos(angle_diff))
                    J11[idx_i, idx_k] = -V[i] * sum_term
                else:
                    angle_diff = theta[i] - theta[k]
                    J11[idx_i, idx_k] = V[i]*V[k] * \
                        (G[i, k]*np.sin(angle_diff) - B[i, k]*np.cos(angle_diff))

            # dP/dV for bus i (diagonal)
            sum_term = 0.0
            for m in range(N):
                angle_diff = theta[i] - theta[m]
                sum_term += V[m]*(G[i, m]*np.cos(angle_diff) +
                                  B[i, m]*np.sin(angle_diff))
            J12[idx_i, idx_i] = sum_term

            # Off-diagonal dP/dV:
            for idx_k, k in enumerate(non_slack):
                if i != k:
                    angle_diff = theta[i] - theta[k]
                    J12[idx_i, idx_k] = V[i] * \
                        (G[i, k]*np.cos(angle_diff) + B[i, k]*np.sin(angle_diff))

        for idx_i, i in enumerate(non_slack):
            for idx_k, k in enumerate(non_slack):
                if i == k:
                    sum_term = 0.0
                    for m in range(N):
                        if m == i:
                            continue
                        angle_diff = theta[i] - theta[m]
                        sum_term += V[m]*(G[i, m]*np.cos(angle_diff) +
                                          B[i, m]*np.sin(angle_diff))
                    J21[idx_i, idx_k] = V[i] * sum_term
                else:
                    angle_diff = theta[i] - theta[k]
                    J21[idx_i, idx_k] = -V[i]*V[k] * \
                        (G[i, k]*np.cos(angle_diff) + B[i, k]*np.sin(angle_diff))

            for idx_i, i in enumerate(non_slack):
                sum_term = 0.0
                for m in range(N):
                    angle_diff = theta[i] - theta[m]
                    sum_term += V[m]*(G[i, m]*np.sin(angle_diff) -
                                      B[i, m]*np.cos(angle_diff))
                J22[idx_i, idx_i] = sum_term
                for idx_k, k in enumerate(non_slack):
                    if i != k:
                        angle_diff = theta[i] - theta[k]
                        J22[idx_i, idx_k] = V[i] * \
                            (G[i, k]*np.sin(angle_diff) -
                             B[i, k]*np.cos(angle_diff))

        J_full = np.block([[J11, J12],
                           [J21, J22]])

        try:
            dx = np.linalg.solve(J_full, -F)
        except np.linalg.LinAlgError:
            print("Jacobian singular at iteration", iteration)
            break

        # Update non-slack voltages and angles:
        for idx, i in enumerate(non_slack):
            theta[i] += damping * dx[idx]
        for idx, i in enumerate(non_slack):
            V[i] += damping * dx[n_ns + idx]

        # Fix slack bus:
        theta[slack_index] = theta_slack
        V[slack_index] = V_slack

    print("Newton-Raphson did not converge within max_iter.")
    return V, theta


def continuation_power_flow(Ybus, P_spec, Q_spec, slack_index, V_slack, theta_slack,
                            n_steps=20, tol=1e-6, max_iter=50, damping=0.7):
    """
    Solve the full AC power flow using a continuation method.

    Starting from a no-load condition (alpha = 0), we gradually increase the load to full (alpha = 1)
    in n_steps, using the previous solution as the initial guess for the next step.

    Parameters
    ----------
    Ybus : ndarray, shape (N, N)
        Full bus admittance matrix (in per unit).
    P_spec : ndarray, shape (N,)
        Specified real power injections (p.u.) at all buses (for loads, use negative values).
    Q_spec : ndarray, shape (N,)
        Specified reactive power injections (p.u.) at all buses (for loads, use negative values).
    slack_index : int
        Index of the slack bus.
    V_slack : float
        Slack bus voltage magnitude (p.u.).
    theta_slack : float
        Slack bus voltage angle (radians).
    n_steps : int, optional
        Number of continuation steps (default is 20).
    tol, max_iter, damping : parameters for the NR solver.

    Returns
    -------
    V : ndarray, shape (N,)
        Bus voltage magnitudes (p.u.) at full load (alpha = 1).
    theta : ndarray, shape (N,)
        Bus voltage angles (radians) at full load.
    """
    N = Ybus.shape[0]
    alphas = np.linspace(0, 1, n_steps)

    # Initialize solution at no-load: all buses at slack voltage.
    V = V_slack * np.ones(N)
    theta = theta_slack * np.ones(N)

    for alpha in alphas[1:]:
        P_target = P_spec * alpha
        Q_target = Q_spec * alpha
        # Use previous solution as initial guess:
        V, theta = newton_raphson_power_flow(Ybus, P_target, Q_target, slack_index, V_slack, theta_slack,
                                             V, theta, tol=tol, max_iter=max_iter, damping=damping)
        # Optionally print iteration info:
        # print(f"Alpha = {alpha:.3f}, V = {V}, theta = {theta}")
    return V, theta
