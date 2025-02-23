import torch
from torch import nn
torch.set_printoptions(precision=10)
torch.autograd.set_detect_anomaly(True)


class VoltageViolationLoss(nn.Module):

    def __init__(self,
                 K,
                 L,
                 s_base,
                 num_buses,
                 max_cs_power=22.17,
                 min_cs_power=-22.17,
                 ev_battery_capacity=70,
                 ev_min_battery_capacity=15,
                 device='cpu',
                 verbose=True,
                 iterations=100,
                 tolerance=1e-6,):

        super(VoltageViolationLoss, self).__init__()

        self.K = torch.from_numpy(K).to(device)
        self.L = torch.from_numpy(L).to(device)
        # self.s_base = torch.tensor(s_base, device=device)
        self.s_base = s_base
        self.num_buses = num_buses

        # EV parameters
        self.max_cs_power = torch.tensor(max_cs_power, device=device)
        self.min_cs_power = torch.tensor(min_cs_power, device=device)
        self.max_ev_charge_power = torch.tensor(22, device=device)
        self.max_ev_discharge_power = torch.tensor(-22, device=device)
        self.ev_battery_capacity = torch.tensor(
            ev_battery_capacity, device=device)
        self.ev_min_battery_capacity = torch.tensor(
            ev_min_battery_capacity, device=device)

        self.iterations = iterations
        self.tolerance = tolerance
        self.device = device

        self.verbose = verbose
        self.timescale = 15

    def forward(self, action, state):

        if self.verbose:
            print("==================================================")
            print(f'action: {action.shape}')
            print(f'state: {state.shape}')

        number_of_cs = action.shape[1]
        prices = state[:, 4]
        step_size = 3
        ev_state_start = 4 + 2*(self.num_buses-1)
        batch_size = state.shape[0]

        # timesscale is a vactor of size number_of_cs with the varaible timescale
        timescale = torch.ones((batch_size, number_of_cs),
                               device=self.device) * self.timescale / 60

        current_capacity = state[:, ev_state_start:(
            ev_state_start + step_size*number_of_cs):step_size]
        connected_bus = state[:, ev_state_start+2:(
            ev_state_start + 2 + step_size*number_of_cs):step_size]
        ev_connected_binary = current_capacity > 0

        max_ev_charge_power = self.max_ev_charge_power * torch.ones(
            (batch_size, number_of_cs), device=self.device)
        max_ev_discharge_power = self.max_ev_discharge_power * torch.ones(
            (batch_size, number_of_cs), device=self.device) 
        battery_capacity = self.ev_battery_capacity * torch.ones(
            (batch_size, number_of_cs), device=self.device)
        ev_min_battery_capacity = self.ev_min_battery_capacity * torch.ones(
            (batch_size, number_of_cs), device=self.device)

        max_ev_charge_power = torch.min(
            max_ev_charge_power, ev_connected_binary * (battery_capacity - current_capacity)/timescale)
        max_ev_discharge_power = torch.max(
            max_ev_discharge_power, ev_connected_binary * (ev_min_battery_capacity - current_capacity)/timescale)

        if self.verbose:
            print("--------------------------------------------------")
            print(f'actions: {action}')
            print(f'ev_connected_binary: {ev_connected_binary}')
            print(f'max_ev_charge_power: {max_ev_charge_power}')
            print(f'max_ev_discharge_power: {max_ev_discharge_power}')
            print(f'current_capacity: {current_capacity}')
            print(f'connected_bus: {connected_bus}')

        # make a binary matrix when action is > 0
        action_binary = torch.where(action >= 0, 1, 0)

        power_usage = action * self.max_cs_power * action_binary -\
            action * self.min_cs_power * (1 - action_binary)

        if self.verbose:
            print(f'power_usage: {power_usage}')

        power_usage = torch.min(power_usage, max_ev_charge_power)
        power_usage = torch.max(power_usage, max_ev_discharge_power)

        if self.verbose:
            print(f'power_usage: {power_usage}')

        # go from power usage to EV_power_per_bus
        EV_power_per_bus = torch.zeros(
            (batch_size, self.num_buses-1),
            device=self.device,
            dtype=power_usage.dtype)

        EV_power_per_bus = EV_power_per_bus.scatter_add(
            dim=1,
            index=connected_bus.long(),
            src=power_usage
        )

        active_power_per_bus = state[:, 4:4+self.num_buses-1]
        reactive_power_per_bus = state[:, 4 +
                                       self.num_buses-1:4+2*(self.num_buses-1)]

        if self.verbose:
            print("--------------------------------------------------")
            print(f'EV_power_per_bus: {EV_power_per_bus}')
            print(f'active_power_per_bus: {active_power_per_bus}')
            print(f'reactive_power_per_bus: {reactive_power_per_bus}')

        active_power_pu = (active_power_per_bus +
                           EV_power_per_bus) / self.s_base

        reactive_power_pu = reactive_power_per_bus / self.s_base

        S = active_power_pu + 1j * reactive_power_pu

        tol = torch.inf
        iteration = 0
        L = torch.zeros((self.num_buses - 1, batch_size),
                        dtype=torch.complex128, device=self.device)
        Z = torch.zeros((self.num_buses - 1, batch_size),
                        dtype=torch.complex128, device=self.device)
        v_k = torch.zeros((self.num_buses - 1, batch_size),
                          dtype=torch.complex128, device=self.device)
        v0 = torch.tensor([1+0j]*(self.num_buses - 1),
                          dtype=torch.complex128, device=self.device)
        v0 = torch.repeat_interleave(v0.view(-1, 1), batch_size, dim=1)
        
        # L_m = torch.repeat_interleave(self.L, batch_size, dim=1)
        L_m = self.L.repeat(1,batch_size)

        v0 = v0.view(-1, batch_size)
        S = S.view(-1, batch_size)
        
        if self.verbose:
            print(f'L_m: {L_m.shape}')
            print(f'self.K: {self.K.shape}')
            print(f'self.L: {self.L.shape}')
            print(f'Z: {Z.shape}')
            print(f'L: {L.shape}')
            print(f'v0: {v0.shape}')
            print(f'v_k: {v_k.shape}')
            print(f'S: {S.shape}')
            
            # print(f'self.L: {self.L}')
            # print(f'L_m: {L_m}')

        while iteration < self.iterations and tol >= self.tolerance:

            L = torch.conj(S * (1 / (v0)))
            Z = self.K @ L
            print(f'Z: {Z.shape} | {iteration}')
            v_k = Z + L_m
            tol = torch.max(torch.abs(torch.abs(v_k) - torch.abs(v0)))
            v0 = v_k

            iteration += 1

        # Convert v0 to a real tensor (for example using its real part)
        v0_real = torch.abs(v0)
        v0_clamped = v0_real.view(batch_size, -1)

        # Compute the loss as a real number
        # For example, penalty on deviation from 1.0
        loss = torch.min(torch.zeros_like(v0_clamped, device=self.device),
                         0.05 - torch.abs(1 - v0_clamped))

        if self.verbose:
            print(f'voltage shape {v0_clamped.real.shape}')
            print(f'Voltage: {v0_clamped.real}')
            print(f'Loss: {loss}')
            print(f'Loss: {loss.shape}'
                  )

        # return 1000*loss.sum(), v0_clamped.real.cpu().detach().numpy()
        return 1000*loss.sum(axis=1)


class V2G_Grid_StateTransition(nn.Module):
    def __init__(self,
                 num_buses,
                 device,
                 max_cs_power=22.17,
                 min_cs_power=-22.17,
                 ev_battery_capacity=70,
                 ev_min_battery_capacity=15,
                 verbose=True,
                 ):
        super(V2G_Grid_StateTransition, self).__init__()

        self.device = device
        self.verbose = verbose
        self.num_buses = num_buses
        self.timescale = 15

        self.max_cs_power = torch.tensor(max_cs_power, device=device)
        self.min_cs_power = torch.tensor(min_cs_power, device=device)
        self.max_ev_charge_power = torch.tensor(22, device=device)
        self.max_ev_discharge_power = torch.tensor(-22, device=device)
        self.ev_battery_capacity = torch.tensor(
            ev_battery_capacity, device=device)
        self.ev_min_battery_capacity = torch.tensor(
            ev_min_battery_capacity, device=device)

    def forward(self, state, new_state, action):

        if self.verbose:
            print(f'\n-------------------- State Transition Function ----------------')
            print(f'old state: {state.shape}')
            print(f'new state: {new_state.shape}')
            print(f'action: {action.shape}')
            print("--------------------------------------------------")

        number_of_cs = action.shape[1]
        prices = state[:, 4]
        step_size = 3
        ev_state_start = 4 + 2*(self.num_buses-1)
        batch_size = state.shape[0]

        timescale = torch.ones((batch_size, number_of_cs),
                               device=self.device) * self.timescale / 60

        current_capacity = state[:, ev_state_start:(
            ev_state_start + step_size*number_of_cs):step_size]
        ev_time_left = state[:, ev_state_start+1:(
            ev_state_start + 1 + step_size*number_of_cs):step_size]

        max_ev_charge_power = self.max_ev_charge_power * torch.ones(
            (batch_size, number_of_cs), device=self.device)
        max_ev_discharge_power = self.max_ev_discharge_power * torch.ones(
            (batch_size, number_of_cs), device=self.device)
        battery_capacity = self.ev_battery_capacity * torch.ones(
            (batch_size, number_of_cs), device=self.device)
        ev_min_battery_capacity = self.ev_min_battery_capacity * torch.ones(
            (batch_size, number_of_cs), device=self.device)
        ev_connected_binary = current_capacity > 0

        max_ev_charge_power = torch.min(
            max_ev_charge_power, ev_connected_binary * (battery_capacity - current_capacity)/timescale)
        max_ev_discharge_power = torch.max(
            max_ev_discharge_power, ev_connected_binary * (ev_min_battery_capacity - current_capacity)/timescale)

        if self.verbose:
            print(f'prices: {prices}')
            print(f'current_capacity: {current_capacity}')
            print(f'max battery_capacity: {battery_capacity}')
            print(f'min_battery_capacity: {ev_min_battery_capacity}')
            print(f'time_left: {ev_time_left}')
            print(f'action: {action}')

        action_binary = torch.where(action >= 0, 1, 0)

        power_usage = action * self.max_cs_power * action_binary -\
            action * self.min_cs_power * (1 - action_binary)

        power_usage = torch.min(power_usage, max_ev_charge_power)
        power_usage = torch.max(power_usage, max_ev_discharge_power)

        new_ev_binary = torch.where(ev_time_left > 1, 1, 0)
        if self.verbose:
            print(f'power_usage: {power_usage}')
            print(f'new_ev_binary: {new_ev_binary}')
            print(f'timescale: {timescale}')

        new_values = torch.zeros_like(new_state, device=self.device)
        new_values[:, ev_state_start:(ev_state_start + step_size*number_of_cs):step_size] = \
            (current_capacity + power_usage * timescale) * new_ev_binary

        if self.verbose:
            print(f'new_values: {new_values}')

        mask = torch.ones_like(new_state, device=self.device)
        mask[:, ev_state_start:(
            ev_state_start + step_size*number_of_cs):step_size] = 1 - new_ev_binary

        new_state = new_state * mask + new_values

        return new_state
