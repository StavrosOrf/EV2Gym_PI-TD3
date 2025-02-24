import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pickle

class V2GProfitMax_Grid_OracleGB():
    '''
    This file contains the EV_City_Math_Model class, which is used to solve the ev_city V2G problem optimally.
    '''
    algo_name = 'Optimal (Offline)'

    def __init__(self,
                 replay_path=None,
                 timelimit=None,
                 MIPGap=None,
                 verbose=True,
                 **kwargs):

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
        active_power = replay.active_power[1:,:]
        reactive_power = replay.reactive_power[1:,:]
        K = replay.K
        L = replay.L
        s_base = replay.s_base
        self.n_b = self.n_transformers
        
        Vmin = 0.95
        Vmax = 1.05

        print(f'Number of buses: {self.n_b}')
        print(f'Number of charging stations: {self.n_cs}')
        print(f'Number of transformers: {self.n_transformers}')
        print(f'active_power: {active_power.shape}')
        print(f'reactive_power: {reactive_power.shape}')
        print(f'K: {K.shape}')
        print(f'L: {L.shape}')
        print(f'cs transformes: {cs_transformer}')
        # create model
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
                                       vtype=GRB.CONTINUOUS,
                                       name='current_cs_ch')

        current_cs_dis = self.m.addVars(self.n_cs,
                                        self.sim_length,
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

        current_tr_ch = self.m.addVars(self.n_transformers,
                                       self.sim_length,
                                       vtype=GRB.CONTINUOUS,
                                       name='current_tr_ch')
        current_tr_dis = self.m.addVars(self.n_transformers,
                                        self.sim_length,
                                        vtype=GRB.CONTINUOUS,
                                        name='current_tr_dis')

        power_cs_ch = self.m.addVars(self.n_cs,
                                     self.sim_length,
                                     vtype=GRB.CONTINUOUS,
                                     name='power_cs_ch')

        power_cs_dis = self.m.addVars(self.n_cs,
                                      self.sim_length,
                                      vtype=GRB.CONTINUOUS,
                                      name='power_cs_dis')

        power_tr_ch = self.m.addVars(self.n_transformers,
                                     self.sim_length,
                                     vtype=GRB.CONTINUOUS,
                                     name='power_tr_ch')

        power_tr_dis = self.m.addVars(self.n_transformers,
                                      self.sim_length,
                                      vtype=GRB.CONTINUOUS,
                                      name='power_tr_dis')
        
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
                self.m.addConstr(current_tr_ch[i, t] == gp.quicksum(current_cs_ch[m, t]
                                                                    for m in range(self.n_cs)
                                                                    if cs_transformer[m] == i))
                self.m.addConstr(current_tr_dis[i, t] == gp.quicksum(current_cs_dis[m, t]
                                                                     for m in range(self.n_cs)
                                                                     if cs_transformer[m] == i))

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

        # transformer current output constraint (circuit breaker)
        self.m.addConstrs((current_tr_ch[i, t] - current_tr_dis[i, t] <= tra_max_amps[i, t]
                           for i in range(self.n_transformers)
                           for t in range(self.sim_length)), name='tr_current_limit_max')
        self.m.addConstrs((current_tr_ch[i, t] - current_tr_dis[i, t] >= tra_min_amps[i, t]
                           for i in range(self.n_transformers)
                           for t in range(self.sim_length)), name='tr_current_limit_min')

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
                        
                        self.m.addConstr(user_satisfaction[p, i, t] == \
                            (ev_des_energy[p, i, t] - energy[p, i, t])**2,
                            name=f'ev_user_satisfaction.{p}.{i}.{t}')
        
        
        print('Adding grid constraints...')
        
        # For clarity, we define:
        B = range(self.n_b)        # set of buses
        T = range(self.sim_length)   # set of time steps
        CS = range(self.n_cs)        # set of charging stations

        # Precompute constant terms for each bus b and time t based on the Q injections.
        # For each bus b, for time t:
        #   const_R[b,t] = W[b].real + sum_{j in B} K[b,j].imag * Q_param[j,t]
        #   const_I[b,t] = W[b].imag - sum_{j in B} K[b,j].real * Q_param[j,t]
        const_R = {}
        const_I = {}
        for b in B:
            const_R[b] = {}
            const_I[b] = {}
            for t in T:
                const_R[b][t] = 1 + sum(K[b, j].imag * reactive_power[j, t] for j in B)
                const_I[b][t] = 0 - sum(K[b, j].real * reactive_power[j, t] for j in B)

        # Create voltage-related variables for each bus b and time t.
        # t_v[b,t] represents the voltage magnitude.
        # s_low[b,t] and s_high[b,t] are slack variables for under- and over-voltage violations.
        t_v = self.m.addVars(self.n_b, self.sim_length, lb=0.0, name="t_v")
        s_low = self.m.addVars(self.n_b, self.sim_length, lb=0.0, name="s_low")
        s_high = self.m.addVars(self.n_b, self.sim_length, lb=0.0, name="s_high")

        # Build expressions for the real and imaginary parts of voltage at each bus b and time t.
        # First, we need the net active injection at each bus b at time t.
        # For each bus b and time t, sum over all CS connected to b:
        P_inj_expr = {}  # dictionary keyed by (b, t)
        for b in B:
            for t in T:
                expr = gp.LinExpr(0.0)
                # Sum over all charging stations cs that are mapped to bus b.
                for cs in CS:
                    if cs_transformer[cs] == b:
                        # Retrieve the decision variables defined for CS: 
                        #   "power_cs_ch[cs,t]" and "power_cs_dis[cs,t]"
                        Pch = self.m.getVarByName(f"power_cs_ch[{cs},{t}]")
                        Pdis = self.m.getVarByName(f"power_cs_dis[{cs},{t}]")
                        # Net active injection from this CS is (Pch - Pdis)
                        expr.addTerms(1.0, Pch)
                        expr.addTerms(-1.0, Pdis)
                P_inj_expr[(b, t)] = expr

        print('Adding grid constraints 2...')
        # Now, for each bus b and time t, build the linear expressions for the real and imaginary parts:
        R_expr = {}  # Expression for the real component of voltage at bus b, time t.
        I_expr = {}  # Expression for the imaginary component.
        for b in B:
            for t in T:
                expr_R = gp.LinExpr(const_R[b][t])
                expr_I = gp.LinExpr(const_I[b][t])
                # Sum over buses j for the contribution from net active injections.
                for j in B:
                    # Retrieve the net injection at bus j at time t.
                    # (P_inj_expr[(j,t)] was built above.)
                    # expr_R.addTerms(K[b, j].real, P_inj_expr[(j, t)])
                    # expr_I.addTerms(K[b, j].imag, P_inj_expr[(j, t)])
                    expr_R += K[b, j].real * P_inj_expr[(j, t)]
                    expr_I += K[b, j].imag * P_inj_expr[(j, t)]
                    
                R_expr[(b, t)] = expr_R
                I_expr[(b, t)] = expr_I
                
                # Add a quadratic (SOC) constraint to link the computed voltage components to the magnitude.
                # (R_expr)^2 + (I_expr)^2 <= t_v[b,t]^2.
                self.m.addQConstr(expr_R * expr_R + expr_I * expr_I <= t_v[b, t] * t_v[b, t],
                                name=f"volt_soc_{b}_{t}")
                                
                # Enforce voltage limits with slack variables:
                # t_v[b,t] + s_low[b,t] >= Vmin, and t_v[b,t] - s_high[b,t] <= Vmax.
                self.m.addConstr(t_v[b, t] + s_low[b, t] >= Vmin, name=f"volt_min_{b}_{t}")
                self.m.addConstr(t_v[b, t] - s_high[b, t] <= Vmax, name=f"volt_max_{b}_{t}")

        # Incorporate the voltage slack penalty into the overall objective.
        # Assume "costs" (or a similar profit measure) is already defined in your model.
        # We add a term that penalizes the total voltage violation:
        voltage_slack = gp.quicksum(s_low[b, t] + s_high[b, t] for b in B for t in T)
                                

        self.m.setObjective(costs + 100 * - 10 * user_satisfaction.sum() + voltage_slack,
                            GRB.MAXIMIZE)

        # print constraints
        # self.m.write("model.lp")
        print(f'Optimizing...')
        self.m.params.NonConvex = 2

        self.m.optimize()

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


if __name__ == '__main__':
    
    
    # # ---------------------------
    # # Data (example placeholders)
    # # ---------------------------
    # n = 5  # number of buses (adjust as needed)
    # Vmin = 0.95
    # Vmax = 1.05

    # # Q: reactive power injection (parameter) for each bus (length n)
    # # (assumed known)
    # Q = np.array([0.1, 0.05, 0.0, -0.02, 0.08])

    # # K: sensitivity matrix (n x n complex matrix)
    # # For instance, K might be computed from a network sensitivity analysis.
    # K = np.array([[0.05+0.01j, 0.01-0.005j, 0, 0, 0],
    #             [0.01+0.005j, 0.04+0.02j, 0.01+0j, 0, 0],
    #             [0, 0.01-0.005j, 0.03+0.015j, 0.005+0.002j, 0],
    #             [0, 0, 0.005-0.002j, 0.02+0.01j, 0.003+0.001j],
    #             [0, 0, 0, 0.003-0.001j, 0.01+0.005j]])

    # # W: base voltage vector (length n, complex)
    # W = np.array([1.0+0j, 0.99+0.01j, 1.01-0.005j, 1.0+0j, 0.98+0.02j])

    # # ---------------------------
    # # Create the model
    # # ---------------------------
    # model = gp.Model("voltage_violation_minimization")

    # # Decision variables: active power injections P[j] for j=0,...,n-1.
    # P = model.addVars(n, lb=-GRB.INFINITY, name="P")

    # # Slack variables for voltage violations (nonnegative)
    # s_low = model.addVars(n, lb=0.0, name="s_low")   # for under-voltage violation
    # s_high = model.addVars(n, lb=0.0, name="s_high") # for over-voltage violation

    # # Auxiliary variables: voltage magnitude at each bus
    # v_m = model.addVars(n, lb=0.0, name="v_m")

    # # ---------------------------
    # # Precompute constant parts for voltage expressions
    # # ---------------------------
    # # For each bus i, we write:
    # #   Re(v_i) = W_i.real + sum_j (K[i,j].real * P[j]) + const_R[i]
    # #   Im(v_i) = W_i.imag + sum_j (K[i,j].imag * P[j]) + const_I[i]
    # # where:
    # #   const_R[i] = sum_j (K[i,j].imag * Q[j])
    # #   const_I[i] = - sum_j (K[i,j].real * Q[j])
    # const_R = np.zeros(n)
    # const_I = np.zeros(n)
    # for i in range(n):
    #     # Initialize with the contributions from reactive power and the base voltage
    #     const_R[i] = W[i].real + sum(K[i, j].imag * Q[j] for j in range(n))
    #     const_I[i] = W[i].imag - sum(K[i, j].real * Q[j] for j in range(n))

    # # ---------------------------
    # # Build voltage expressions for each bus i
    # # ---------------------------
    # # We'll create linear expressions for the real and imaginary parts of v_i:
    # #   Re(v_i) = const_R[i] + sum_j K[i,j].real * P[j]
    # #   Im(v_i) = const_I[i] + sum_j K[i,j].imag * P[j]
    # R_expr = {}
    # I_expr = {}
    # for i in range(n):
    #     expr_R = gp.LinExpr(const_R[i])
    #     expr_I = gp.LinExpr(const_I[i])
    #     for j in range(n):
    #         expr_R.addTerms(K[i, j].real, P[j])
    #         expr_I.addTerms(K[i, j].imag, P[j])
    #     R_expr[i] = expr_R
    #     I_expr[i] = expr_I

    # # ---------------------------
    # # Add SOC constraints to capture voltage magnitude
    # # ---------------------------
    # # We require: sqrt( R_expr[i]^2 + I_expr[i]^2 ) <= v_m[i]
    # # which is equivalent to: R_expr[i]^2 + I_expr[i]^2 <= v_m[i]^2
    # for i in range(n):
    #     model.addQConstr(R_expr[i]*R_expr[i] + I_expr[i]*I_expr[i] <= v_m[i]*v_m[i],
    #                     name=f"soc_{i}")

    # # ---------------------------
    # # Voltage limit constraints (with slack)
    # # ---------------------------
    # # To allow violations we add slack variables:
    # #   v_m[i] + s_low[i] >= Vmin   (if v_m[i] is too low, s_low[i] > 0)
    # #   v_m[i] - s_high[i] <= Vmax   (if v_m[i] is too high, s_high[i] > 0)
    # for i in range(n):
    #     model.addConstr(v_m[i] + s_low[i] >= Vmin, name=f"volt_min_{i}")
    #     model.addConstr(v_m[i] - s_high[i] <= Vmax, name=f"volt_max_{i}")

    # # ---------------------------
    # # Objective: minimize total voltage violation
    # # ---------------------------
    # model.setObjective(gp.quicksum(s_low[i] + s_high[i] for i in range(n)), GRB.MINIMIZE)

    # # ---------------------------
    # # Optimize the model
    # # ---------------------------
    # model.optimize()
    # model.write("v2g_grid_model.lp")
    # # ---------------------------
    # # Retrieve and display results
    # # ---------------------------
    # if model.status == GRB.OPTIMAL:
    #     P_opt = model.getAttr('x', P)
    #     t_opt = model.getAttr('x', v_m)
    #     slack_low = model.getAttr('x', s_low)
    #     slack_high = model.getAttr('x', s_high)
    #     print("Optimal active power injections (P):")
    #     for i in range(n):
    #         print(f"Bus {i}: {P_opt[i]:.4f}")
    #     print("\nVoltage magnitudes (v_m):")
    #     for i in range(n):
    #         print(f"Bus {i}: {t_opt[i]:.4f} (s_low: {slack_low[i]:.4f}, s_high: {slack_high[i]:.4f})")
    # else:
    #     print("No optimal solution found.")
    
    pass
