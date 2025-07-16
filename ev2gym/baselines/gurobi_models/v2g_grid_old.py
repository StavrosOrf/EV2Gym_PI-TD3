import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pickle


class V2GProfitMax_Grid_OracleGB():
    '''
    This file contains the EV_City_Math_Model class, which is used to solve the ev_city V2G problem optimally.
    '''
    algo_name = 'MPC'

    def __init__(self,
                 replay_path=None,
                 timelimit=600, #60
                 MIPGap=None,
                 verbose=True,
                 **kwargs):
        
        verbose = True
        replay = pickle.load(open(replay_path, 'rb'))

        self.sim_length = replay.sim_length
        self.n_cs = replay.n_cs
        self.number_of_ports_per_cs = replay.max_n_ports
        self.n_transformers = replay.n_transformers
        self.timescale = replay.timescale
        dt = replay.timescale / 60  # time step
        print(f'\nGurobi MIQP solver for MO V2G Grid.')
        print('Loading data...')

        tra_max_amps = replay.tra_max_amps
        tra_min_amps = replay.tra_min_amps
        cs_transformer = replay.cs_transformer
        port_max_charge_current = replay.port_max_charge_current
        port_min_charge_current = replay.port_min_charge_current
        port_max_discharge_current = replay.port_max_discharge_current
        port_min_discharge_current = replay.port_min_discharge_current
        voltages = replay.voltages / 1000  # phases included in voltage
        # phases = replay.phases

        charge_prices = replay.charge_prices  # Charge prices are in €/kWh
        discharge_prices = replay.discharge_prices  # Discharge prices are in €/kWh

        power_setpoints = replay.power_setpoints

        cs_ch_efficiency = replay.cs_ch_efficiency
        cs_dis_efficiency = replay.cs_dis_efficiency

        ev_max_energy = replay.ev_max_energy
        ev_des_energy = replay.ev_des_energy

        ev_max_ch_power = replay.ev_max_ch_power  # * self.dt
        ev_max_dis_power = replay.ev_max_dis_power  # * self.dt
        ev_max_energy_at_departure = replay.max_energy_at_departure
        ev_des_energy = replay.ev_des_energy
        u = replay.u
        energy_at_arrival = replay.energy_at_arrival
        ev_arrival = replay.ev_arrival
        t_dep = replay.t_dep

        # grid related
        active_power = replay.active_power[1:, :]
        reactive_power = replay.reactive_power[1:, :]
        K = replay.K
        L_const = replay.L
        s_base = replay.s_base
        self.n_b = self.n_transformers

        S_r = active_power.T
        S_i = reactive_power.T / s_base
        K_r = np.real(K)
        K_i = np.imag(K)
        W_r = np.real(L_const).reshape(-1)  # constant offset, shape (n,)
        W_i = np.imag(L_const).reshape(-1)
        
        #test formula for voltages 
        # print(compute_voltage_magnitudes(K,L_const, active_power/s_base, reactive_power/s_base))

        V_min = 0.95
        V_max = 1.05
        penalty_low = 1.0
        penalty_high = 1.0

        if verbose:
            print(f'Number of buses: {self.n_b}')
            print(f'Number of charging stations: {self.n_cs}')
            print(f'Number of transformers: {self.n_transformers}')
            print(f'active_power: {active_power.shape}')
            print(f'reactive_power: {reactive_power.shape}')
            print(f'K: {K.shape if hasattr(K, "shape") else "scalar"}')
            print(f'L: {L_const.shape if hasattr(L_const, "shape") else "scalar"}')
            print(f'W_r: {W_r.shape}')
            print(f'W_i: {W_i.shape}')
            print(f'cs transformes: {cs_transformer}')
            print(f'sbase: {s_base}')
            print('Creating Gurobi model...')
        
        self.m = gp.Model("ev_city")
        if verbose:
            self.m.setParam('OutputFlag', 1)
        else:
            self.m.setParam('OutputFlag', 0)

        if MIPGap is not None:
            self.m.setParam('MIPGap', MIPGap)
        if timelimit is not None:
            self.m.setParam('TimeLimit', timelimit)

        # energy of EVs t timeslot t
        energy = self.m.addVars(self.number_of_ports_per_cs,
                                self.n_cs,
                                self.sim_length,
                                vtype=GRB.CONTINUOUS,
                                name='energy')

        current_ev_dis = self.m.addVars(self.number_of_ports_per_cs,
                                        self.n_cs,
                                        self.sim_length,
                                        vtype=GRB.CONTINUOUS,
                                        name='current_ev_dis')
        current_ev_ch = self.m.addVars(self.number_of_ports_per_cs,
                                       self.n_cs,
                                       self.sim_length,
                                       vtype=GRB.CONTINUOUS,
                                       name='current_ev_ch')

        act_current_ev_dis = self.m.addVars(self.number_of_ports_per_cs,
                                            self.n_cs,
                                            self.sim_length,
                                            vtype=GRB.CONTINUOUS,
                                            name='act_current_ev_dis')
        act_current_ev_ch = self.m.addVars(self.number_of_ports_per_cs,
                                           self.n_cs,
                                           self.sim_length,
                                           vtype=GRB.CONTINUOUS,
                                           name='act_current_ev_ch')

        current_cs_ch = self.m.addVars(self.n_cs,
                                       self.sim_length,
                                       ub=32,
                                       lb=0,
                                       vtype=GRB.CONTINUOUS,
                                       name='current_cs_ch')

        current_cs_dis = self.m.addVars(self.n_cs,
                                        self.sim_length,
                                        # ub=32,
                                        # lb=0,
                                        vtype=GRB.CONTINUOUS,
                                        name='current_cs_dis')

        omega_ch = self.m.addVars(self.number_of_ports_per_cs,
                                  self.n_cs,
                                  self.sim_length,
                                  vtype=GRB.BINARY,
                                  name='omega_ch')
        omega_dis = self.m.addVars(self.number_of_ports_per_cs,
                                   self.n_cs,
                                   self.sim_length,
                                   vtype=GRB.BINARY,
                                   name='omega_dis')

        # current_tr_ch = self.m.addVars(self.n_transformers,
        #                                self.sim_length,
        #                                vtype=GRB.CONTINUOUS,
        #                                name='current_tr_ch')
        # current_tr_dis = self.m.addVars(self.n_transformers,
        #                                 self.sim_length,
        #                                 vtype=GRB.CONTINUOUS,
        #                                 name='current_tr_dis')

        power_cs_ch = self.m.addVars(self.n_cs,
                                     self.sim_length,
                                     ub=22,
                                     lb=0,
                                     
                                     vtype=GRB.CONTINUOUS,
                                     name='power_cs_ch')

        power_cs_dis = self.m.addVars(self.n_cs,
                                      self.sim_length,
                                      ub=22,
                                      lb=0,
                                      vtype=GRB.CONTINUOUS,
                                      name='power_cs_dis')

        power_tr_ch = self.m.addVars(self.n_transformers,
                                     self.sim_length,
                                     lb=0,
                                     vtype=GRB.CONTINUOUS,
                                     name='power_tr_ch')

        power_tr_dis = self.m.addVars(self.n_transformers,
                                      self.sim_length,
                                      lb=0,
                                      vtype=GRB.CONTINUOUS,
                                      name='power_tr_dis')

        total_power_per_bus = self.m.addVars(self.n_transformers,
                                             self.sim_length,
                                             lb=-100,
                                             ub=100,
                                             vtype=GRB.CONTINUOUS,
                                             name='total_power_per_bus')

        user_satisfaction = self.m.addVars(self.number_of_ports_per_cs,
                                           self.n_cs,
                                           self.sim_length,
                                           vtype=GRB.CONTINUOUS,
                                           name='user_satisfaction')

        costs = self.m.addVar(vtype=GRB.CONTINUOUS,
                              name='total_soc')

        self.m.update()

        for t in range(self.sim_length):
            for i in range(self.n_transformers):
                # self.m.addConstr(current_tr_ch[i, t] == gp.quicksum(current_cs_ch[m, t]
                #                                                     for m in range(self.n_cs)
                #                                                     if cs_transformer[m] == i))
                # self.m.addConstr(current_tr_dis[i, t] == gp.quicksum(current_cs_dis[m, t]
                #                                                      for m in range(self.n_cs)
                #                                                      if cs_transformer[m] == i))

                self.m.addConstr(power_tr_ch[i, t] == gp.quicksum(power_cs_ch[m, t]
                                                                  for m in range(self.n_cs)
                                                                  if cs_transformer[m] == i),
                                 name=f'power_tr_ch.{i}.{t}')

                self.m.addConstr(power_tr_dis[i, t] == gp.quicksum(power_cs_dis[m, t]
                                                                   for m in range(self.n_cs)
                                                                   if cs_transformer[m] == i),
                                 name=f'power_tr_dis.{i}.{t}')

        costs = gp.quicksum(act_current_ev_ch[p, i, t] * voltages[i] * cs_ch_efficiency[i, t] * dt * charge_prices[i, t] +
                            act_current_ev_dis[p, i, t] * voltages[i] *
                            cs_dis_efficiency[i, t] *
                            dt * discharge_prices[i, t]
                            for p in range(self.number_of_ports_per_cs)
                            for i in range(self.n_cs)
                            for t in range(self.sim_length))

        self.m.addConstrs(power_cs_ch[i, t] == (current_cs_ch[i, t] * voltages[i])
                          for i in range(self.n_cs)
                          for t in range(self.sim_length))
        self.m.addConstrs(power_cs_dis[i, t] == (current_cs_dis[i, t] * voltages[i])
                          for i in range(self.n_cs)
                          for t in range(self.sim_length))

        self.m.addConstrs((total_power_per_bus[i, t] == (power_tr_ch[i, t]/s_base - power_tr_dis[i, t]/s_base + active_power[i, t]/s_base)
                           for i in range(self.n_transformers)
                           for t in range(self.sim_length)),
                          name='power_per_bus_const')

        # charging station total current output (sum of ports) constraint
        self.m.addConstrs((current_cs_ch[i, t] == act_current_ev_ch.sum('*', i, t)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='cs_ch_current_output')
        self.m.addConstrs((current_cs_dis[i, t] == act_current_ev_dis.sum('*', i, t)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='cs_dis_current_output')

        # charging station current output constraint
        self.m.addConstrs((-current_cs_dis[i, t] + current_cs_ch[i, t] >= port_max_discharge_current[i]
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='cs_current_dis_limit_max')
        self.m.addConstrs((-current_cs_dis[i, t] + current_cs_ch[i, t] <= port_max_charge_current[i]
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='cs_curent_ch_limit_max')

        self.m.addConstrs((act_current_ev_ch[p, i, t] == current_ev_ch[p, i, t] * omega_ch[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           #    if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0
                           ), name='act_ev_current_ch')

        self.m.addConstrs((act_current_ev_dis[p, i, t] == current_ev_dis[p, i, t] * omega_dis[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           #    if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0
                           ), name='act_ev_current_dis')

        # ev current output constraint
        self.m.addConstrs((current_ev_ch[p, i, t] >= port_min_charge_current[i]  # * omega_ch[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           #    if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0
                           ), name='ev_current_ch_limit_min')
        self.m.addConstrs((current_ev_dis[p, i, t] >= -port_min_discharge_current[i]  # * omega_dis[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           #    if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0
                           ), name='ev_current_dis_limit_min')

        # ev max charging current constraint
        self.m.addConstrs((current_ev_ch[p, i, t] <= min(ev_max_ch_power[p, i, t]/voltages[i], port_max_charge_current[i])
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0
                           ),
                          name='ev_current_ch_limit_max')

        # ev max discharging current constraint
        self.m.addConstrs((current_ev_dis[p, i, t] <= min(-ev_max_dis_power[p, i, t]/voltages[i], -port_max_discharge_current[i])
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           if u[p, i, t] == 1 and ev_arrival[p, i, t] == 0
                           ),
                          name='ev_current_dis_limit_max')

        # ev charge power if empty port constraint
        for t in range(self.sim_length):
            for i in range(self.n_cs):
                for p in range(self.number_of_ports_per_cs):
                    if u[p, i, t] == 0 or ev_arrival[p, i, t] == 1:
                        #     self.m.addLConstr((act_current_ev_ch[p, i, t] == 0),
                        #                       name=f'ev_empty_port_ch.{p}.{i}.{t}')
                        #     self.m.addLConstr((act_current_ev_dis[p, i, t] == 0),
                        #                       name=f'ev_empty_port_dis.{p}.{i}.{t}')

                        self.m.addLConstr((omega_ch[p, i, t] == 0),
                                          name=f'omega_empty_port_ch.{p}.{i}.{t}')
                        self.m.addLConstr((omega_dis[p, i, t] == 0),
                                          name=f'omega_empty_port_dis.{p}.{i}.{t}')

                    if u[p, i, t] == 0 and t_dep[p, i, t] == 0:
                        self.m.addLConstr(energy[p, i, t] == 0,
                                          name=f'ev_empty_port_energy.{p}.{i}.{t}')

        # energy of EVs after charge/discharge constraint
        for t in range(1, self.sim_length):
            for i in range(self.n_cs):
                for p in range(self.number_of_ports_per_cs):
                    if ev_arrival[p, i, t] == 1:
                        self.m.addLConstr(
                            energy[p, i, t] == energy_at_arrival[p, i, t],
                            name=f'ev_arrival_energy.{p}.{i}.{t}')

                    if u[p, i, t-1] == 1:
                        self.m.addConstr(energy[p, i, t] == (energy[p, i, t-1] +
                                                             act_current_ev_ch[p, i, t] * voltages[i] * cs_ch_efficiency[i, t] * dt -
                                                             act_current_ev_dis[p, i, t] * voltages[i] * cs_dis_efficiency[i, t] * dt),
                                         name=f'ev_energy.{p}.{i}.{t}')

        # energy level of EVs constraint
        self.m.addConstrs((energy[p, i, t] >= 0
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='ev_energy_level_min')
        self.m.addConstrs((energy[p, i, t] <= ev_max_energy[p, i, t]
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)
                           if t_dep[p, i, t] != 1
                           ), name='ev_energy_level_max')

        # Power output of EVs constraint
        self.m.addConstrs((omega_dis[p, i, t] * omega_ch[p, i, t] == 0
                           for p in range(self.number_of_ports_per_cs)
                           for i in range(self.n_cs)
                           for t in range(self.sim_length)), name='ev_power_mode_2')

        # time of departure of EVs
        for t in range(self.sim_length):
            for i in range(self.n_cs):
                for p in range(self.number_of_ports_per_cs):
                    if t_dep[p, i, t] == 1:
                        # self.m.addLConstr(energy[p, i, t] >= ev_max_energy_at_departure[p, i, t]-5,
                        #                   name=f'ev_departure_energy.{p}.{i}.{t}')

                        self.m.addConstr(user_satisfaction[p, i, t] ==
                                         (ev_des_energy[p, i, t] -
                                          energy[p, i, t])**2,
                                         name=f'ev_user_satisfaction.{p}.{i}.{t}')

        print('Adding simplified grid constraints...')

        timesteps = range(self.sim_length)
        buses = range(self.n_b)

        # Simplified voltage magnitude variables (linear approximation)
        v_magnitude = self.m.addVars(timesteps, buses, 
                                   lb=0.8, ub=1.2, 
                                   name="v_magnitude")

        # Slack variables for voltage violations
        slack_low = self.m.addVars(timesteps, buses, lb=0.0, name="slack_low")
        slack_high = self.m.addVars(timesteps, buses, lb=0.0, name="slack_high")
        
        total_reactive_per_bus = self.m.addVars(
            self.n_b, self.sim_length,
            lb=-GRB.INFINITY, ub=GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name='total_reactive_per_bus'
        )
        
        self.m.addConstrs(
            (total_reactive_per_bus[i, t]
                == reactive_power[i, t] / s_base
            for i in buses for t in timesteps),
            name = 'reactive_per_bus'
        )
        
        for t in timesteps:
            for j in buses:
                # start from the real part of your L_const offset
                expr = gp.LinExpr(W_r[j])
                # add P‐sensitivities
                for i in buses:
                    expr.addTerms(K_r[j, i], total_power_per_bus[i, t])
                    expr.addTerms(K_i[j, i], total_reactive_per_bus[i, t])
                # link to our decision variable
                self.m.addConstr(
                    v_magnitude[t, j] == expr,
                    name = f"voltage_approx_t{t}_bus{j}"
                )
                # enforce limits via slacks
                self.m.addConstr(
                    v_magnitude[t, j] + slack_low[t, j]  >= V_min,
                    name = f"volt_low_t{t}_bus{j}"
                )
                self.m.addConstr(
                    v_magnitude[t, j] - slack_high[t, j] <= V_max,
                    name = f"volt_high_t{t}_bus{j}"
                )


        # Linear voltage slack penalty (changed from quadratic)
        voltage_slack = gp.LinExpr()
        for t in timesteps:
            for j in buses:
                voltage_slack.add(penalty_low * slack_low[t, j])
                voltage_slack.add(penalty_high * slack_high[t, j])

        # Simplified objective: minimize costs and voltage violations
        self.m.setObjective( -costs +
                            10000*voltage_slack + 10000 *user_satisfaction.sum(),
                            GRB.MINIMIZE)

        # Write and solve model
        self.m.write("model.lp")
        print(f'Optimizing simplified model...')
        
        # Remove NonConvex parameter since we're using linear approximation
        # Enable presolve for better performance
        self.m.params.Presolve = 2
        
        self.m.optimize()

        if self.m.status == GRB.INFEASIBLE:
            self.m.computeIIS()
            self.m.write("model.ilp")
        else:
            print("Model is feasible; no IIS to compute.")

        print(f'model status: {self.m.status}')

        # self.m.computeIIS()
        # self.m.write("model.ilp")

        self.act_current_ev_ch = act_current_ev_ch
        self.act_current_ev_dis = act_current_ev_dis
        self.port_max_charge_current = port_max_charge_current
        self.port_max_discharge_current = port_max_discharge_current

        if self.m.status != GRB.Status.OPTIMAL:
            print(f'Optimization ended with status {self.m.status}')
            # exit()

        self.get_actions()

    def get_actions(self):
        '''
        This function returns the actions of the EVs in the simulation normalized to [-1, 1]
        '''

        self.actions = np.zeros([self.number_of_ports_per_cs,
                                 self.n_cs, self.sim_length])

        for t in range(self.sim_length):
            for i in range(self.n_cs):
                for p in range(self.number_of_ports_per_cs):
                    if self.act_current_ev_ch[p, i, t].x > 0:
                        self.actions[p, i, t] = self.act_current_ev_ch[p, i, t].x  \
                            / self.port_max_charge_current[i]
                    elif self.act_current_ev_dis[p, i, t].x > 0:
                        self.actions[p, i, t] = self.act_current_ev_dis[p, i, t].x \
                            / self.port_max_discharge_current[i]

        return self.actions

    def get_action(self, env, **kwargs):
        '''
        This function returns the action for the current step of the environment.
        '''

        step = env.current_step

        return self.actions[:, :, step].T.reshape(-1)



def compute_voltage_magnitudes(K, L, P, Q):
    """
    Compute voltage magnitudes using linear sensitivity:
    |V| ≈ Re{L} + Re{K}·P + Im{K}·Q

    Parameters:
    - K: (n_buses, n_buses) complex sensitivity matrix
    - L: (n_buses,) complex offset vector
    - P: (n_buses,) real active power injections (p.u.)
    - Q: (n_buses,) real reactive power injections (p.u.)

    Returns:
    - V: (n_buses,) estimated voltage magnitudes (p.u.)
    """
    K_r = np.real(K)
    K_i = np.imag(K)
    W_r = np.real(L)
    return W_r + K_r.dot(P) + K_i.dot(Q)
