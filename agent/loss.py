import torch
from torch import nn


class VoltageViolationLoss(nn.Module):

    def __init__(self,
                 K,
                 L,
                 s_base,
                 num_buses,
                 iterations=100,
                 tolerance=1e-6,):

        super(VoltageViolationLoss, self).__init__()

        self.K = torch.tensor(K)
        self.L = torch.tensor(L)
        self.s_base = torch.tensor(s_base)

        self.num_buses = num_buses
        self.iterations = iterations
        self.tolerance = tolerance

    
    def forward(self, EV_power_per_bus, active_power_per_bus, reactive_power_per_bus):
        # TODO: go from actions to EV_power_per_bus
        
        EV_power_per_bus = torch.tensor(EV_power_per_bus)
        active_power_per_bus = torch.tensor(active_power_per_bus)
        reactive_power_per_bus = torch.tensor(reactive_power_per_bus)

        active_power_pu = (active_power_per_bus +
                           EV_power_per_bus) / self.s_base
        
        reactive_power_pu = reactive_power_per_bus / self.s_base

        S = active_power_pu + 1j * reactive_power_pu
        # print(f'S shape {S.shape}')
        tol = torch.inf
        iteration = 0
        L = torch.zeros((self.num_buses - 1,1), dtype=torch.complex128)
        Z = torch.zeros((self.num_buses - 1,1), dtype=torch.complex128)
        v_k = torch.zeros((self.num_buses - 1,1), dtype=torch.complex128)
        v0 = torch.tensor([1+0j], dtype=torch.complex128)

        v0 = v0.view(-1, 1)
        S = S.view(-1, 1)
        
        while iteration < self.iterations and tol >= self.tolerance:
            L = torch.conj(S * (1 / (v0)))
            Z = self.K @ L
            v_k = Z + self.L
            tol = torch.max(torch.abs(torch.abs(v_k) - torch.abs(v0)))
            v0 = v_k
            
            iteration += 1

        v0 =v0.view(-1)
        # print(f'voltage shape {v0.real.shape}')
        # print(f'It: {iteration} tol: {tol}')
        return v0.real.detach().numpy()
