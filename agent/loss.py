import torch
from torch import nn
# torch.set_printoptions(precision=10)
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
        ev_state_start = 37
        batch_size = state.shape[0]

        # timesscale is a vactor of size number_of_cs with the varaible timescale
        timescale = torch.ones((batch_size, number_of_cs),
                               device=self.device) * self.timescale / 60

        current_capacity = state[:, ev_state_start:(
            ev_state_start + step_size*number_of_cs):step_size]
        connected_bus = state[:, ev_state_start+2:(
            ev_state_start + 2 + step_size*number_of_cs):step_size]

        max_ev_charge_power = self.max_ev_charge_power * torch.ones(
            (batch_size, number_of_cs), device=self.device)
        max_ev_discharge_power = self.max_ev_discharge_power * torch.ones(
            (batch_size, number_of_cs), device=self.device)
        battery_capacity = self.ev_battery_capacity * torch.ones(
            (batch_size, number_of_cs), device=self.device)
        ev_min_battery_capacity = self.ev_min_battery_capacity * torch.ones(
            (batch_size, number_of_cs), device=self.device)

        # print(f'battery_capacity: {battery_capacity.shape}')
        # print(f'current_capacity: {current_capacity.shape}')
        # max_ev_charge_power is bound by max battery capacity and min battery capacity
        max_ev_charge_power = torch.min(
            max_ev_charge_power, (battery_capacity - current_capacity)/timescale)
        max_ev_discharge_power = torch.max(
            max_ev_discharge_power, (ev_min_battery_capacity - current_capacity)/timescale)

        if self.verbose:
            # print(f'prices: {prices}')
            print("--------------------------------------------------")
            print(f'current_capacity: {current_capacity}')
            print(f'connected_bus: {connected_bus}')

        # make a binary matrix when action is > 0
        action_binary = torch.where(action > 0, 1, 0)

        power_usage = action * self.max_cs_power * action_binary -\
            action * self.min_cs_power * (1 - action_binary)

        if self.verbose:
            print("--------------------------------------------------")
            print(f'power_usage: {power_usage}')

        power_usage = torch.min(power_usage, max_ev_charge_power)
        if self.verbose:
            print("--------------------------------------------------")
            print(f'power_usage: {power_usage}')        

        # go from power usage to EV_power_per_bus
        EV_power_per_bus = torch.zeros(
            (batch_size, self.num_buses-1), device=self.device)
        for i in range(number_of_cs):
            EV_power_per_bus[:, connected_bus[:,
                                              i].long()] += power_usage[:, i]

        active_power_per_bus = state[:, 4:4+self.num_buses-1]
        reactive_power_per_bus = torch.zeros_like(
            active_power_per_bus, device=self.device)

        if self.verbose:
            print("--------------------------------------------------")
            print(f'EV_power_per_bus: {EV_power_per_bus}')            
            print(f'active_power_per_bus: {active_power_per_bus}')
            # print("--------------------------------------------------")
            # print(f'EV_power_per_bus: {EV_power_per_bus.shape}')
            # print(f'active_power_per_bus: {active_power_per_bus.shape}')

        active_power_pu = (active_power_per_bus +
                           EV_power_per_bus) / self.s_base
        
        reactive_power_pu = reactive_power_per_bus / self.s_base

        S = active_power_pu + 1j * reactive_power_pu

        tol = torch.inf
        iteration = 0
        L = torch.zeros((self.num_buses - 1, batch_size), dtype=torch.complex128, device=self.device)
        Z = torch.zeros((self.num_buses - 1, batch_size), dtype=torch.complex128, device=self.device)
        v_k = torch.zeros((self.num_buses - 1, batch_size), dtype=torch.complex128, device=self.device)
        v0 = torch.tensor([1+0j]*(self.num_buses - 1), dtype=torch.complex128, device=self.device)
        v0 = torch.repeat_interleave(v0.view(-1, 1), batch_size, dim=1)

        v0 = v0.view(-1, batch_size)
        S = S.view(-1, batch_size)
        # print(f'S v2:{S}')
        # if self.verbose:
        #     print(f'S shape {S.shape}')
        #     print(f'v0 shape {v0.shape}')
        #     print(f'self.K shape {self.K.shape}')
        #     print(f'self.L shape {self.L.shape}')
        #     print(f'L shape {L.shape}')
        #     print(f'Z shape {Z.shape}')
        
        # print(f'K shape: {self.K.shape}')
        # print(f'L shape: {self.L.shape}')
        # print(f'S shape: {S.shape}')
        # print(f'v0 shape: {v0.shape}')
        # print(f'vk shape: {v_k.shape}')

        while iteration < self.iterations and tol >= self.tolerance:
            L = torch.conj(S * (1 / (v0)))
            Z = self.K @ L
            v_k = Z + self.L
            tol = torch.max(torch.abs(torch.abs(v_k) - torch.abs(v0)))
            v0 = v_k

            iteration += 1

        v0 = v0.view(batch_size, -1)
        
        # TODO: consider masking loss components where EVs are not connected
        # loss = torch.min(0, (1.05-0.95)/2 - torch.abs(1-v0.real))
        loss = torch.min(torch.zeros_like(v0.real, device=self.device),
                         0.05 - torch.abs(1-v0.real))
        
        # if self.verbose:
        #     print(f'voltage shape {v0.real.shape}')
        #     print(f'Voltage: {v0.real}')
        # print(f'Loss: {loss}')
        #     print(f'Loss: {loss.shape}')       
        
        # return 1000*loss.sum(), v0.real.cpu().detach().numpy()
        return 1000*loss.mean()
