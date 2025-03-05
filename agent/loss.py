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

        v_k = torch.zeros((self.num_buses - 1, batch_size),
                          dtype=torch.complex128, device=self.device)
        v0 = torch.tensor([1+0j]*(self.num_buses - 1),
                          dtype=torch.complex128, device=self.device)
        v0 = torch.repeat_interleave(v0.view(-1, 1), batch_size, dim=1)

        v0 = v0.view(batch_size, -1)
        S = S.view(batch_size, -1)

        W = self.L.view(-1)

        if self.verbose:
            print(f'W: {W.shape}')
            print(f'self.K: {self.K.shape}')
            print(f'self.L: {self.L.shape}')
            # print(f'Z: {Z.shape}')
            # print(f'L: {L.shape}')
            print(f'v0: {v0.shape}')
            print(f'v_k: {v_k.shape}')
            print(f'S: {S.shape}')

            # print(f'self.L: {self.L}')
            # print(f'L_m: {L_m}')

        while iteration < self.iterations and tol >= self.tolerance:

            L = torch.conj(S / v0)
            # print(f'L: {L.shape} | {iteration}')
            Z = torch.matmul(self.K, L.T)
            Z = Z.T
            # print(f'Z: {Z.shape} | {iteration}')
            v_k = Z + W
            tol = torch.max(torch.abs(torch.abs(v_k) - torch.abs(v0)))
            v0 = v_k
            # print(f'v0: {v0.shape} | {iteration}')

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

    def voltage_real_operations(self, state, action):

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
        # assuming S comes with batch_size as first dimension
        batch_size = S.shape[0]
        n = self.num_buses - 1  # number of buses (excluding slack)

        # -- Initialize v0 as 1+0j represented by two real tensors:
        v0_r = torch.ones(
            (batch_size, n), dtype=torch.float64, device=self.device)
        v0_i = torch.zeros(
            (batch_size, n), dtype=torch.float64, device=self.device)

        # -- Split S into real and imaginary parts:
        S_r = S.real.view(batch_size, -1)
        S_i = S.imag.view(batch_size, -1)

        # -- Get W from self.L (the constant offset) and split into real and imaginary parts.
        #    Assume self.L is a complex tensor of shape (n,)
        W_r = self.L.real.view(-1)  # shape: (n,)
        W_i = self.L.imag.view(-1)
        # Repeat W for the batch dimension:
        W_r = W_r.unsqueeze(0).repeat(batch_size, 1)  # shape: (batch_size, n)
        W_i = W_i.unsqueeze(0).repeat(batch_size, 1)

        # Pre-extract K's real and imaginary parts (assume self.K is a complex matrix of shape (n, n))
        K_r = self.K.real  # shape: (n, n)
        K_i = self.K.imag  # shape: (n, n)
        # Expand for batch multiplication:
        K_r_batch = K_r.unsqueeze(0)  # shape: (1, n, n)
        K_i_batch = K_i.unsqueeze(0)  # shape: (1, n, n)

        iteration = 0
        tol = float('inf')

        while iteration < self.iterations and tol >= self.tolerance:
            # --- Compute L = conj(S / v0) without using complex numbers.
            # For each batch, compute denominator: v0_abs_sq = v0_r^2 + v0_i^2
            v0_abs_sq = v0_r**2 + v0_i**2  # shape: (batch_size, n)

            # Compute S/v0:
            #   real part = (S_r*v0_r + S_i*v0_i) / (v0_r^2+v0_i^2)
            #   imag part = (S_i*v0_r - S_r*v0_i) / (v0_r^2+v0_i^2)
            div_real = (S_r * v0_r + S_i * v0_i) / v0_abs_sq
            div_imag = (S_i * v0_r - S_r * v0_i) / v0_abs_sq

            # Take the conjugate: L = conj(S/v0) = div_real - j*div_imag.
            L_r = div_real      # real part
            L_i = -div_imag     # imaginary part

            # --- Compute Z = K @ L^T for each batch.
            # L is (batch_size, n); we first reshape L_r and L_i to (batch_size, n, 1)
            L_r_unsq = L_r.unsqueeze(2)  # shape: (batch_size, n, 1)
            L_i_unsq = L_i.unsqueeze(2)  # shape: (batch_size, n, 1)
            # Using complex multiplication rules:
            #   Z_r = K_r @ L_r - K_i @ L_i
            #   Z_i = K_r @ L_i + K_i @ L_r
            # shape: (batch_size, n, 1)
            Z_r = torch.matmul(K_r_batch, L_r_unsq) - \
                torch.matmul(K_i_batch, L_i_unsq)
            # shape: (batch_size, n, 1)
            Z_i = torch.matmul(K_r_batch, L_i_unsq) + \
                torch.matmul(K_i_batch, L_r_unsq)
            # Remove the last dimension:
            Z_r = Z_r.squeeze(2)  # shape: (batch_size, n)
            Z_i = Z_i.squeeze(2)

            # --- Update voltage: v_k = Z + W.
            v_k_r = Z_r + W_r
            v_k_i = Z_i + W_i

            # --- Compute tolerance: compare the change in voltage magnitudes.
            # Voltage magnitude: |v| = sqrt(v_r^2 + v_i^2)
            mag_v_k = torch.sqrt(v_k_r**2 + v_k_i**2)
            mag_v0 = torch.sqrt(v0_r**2 + v0_i**2)
            tol = torch.max(torch.abs(mag_v_k - mag_v0))

            # Update v0 for the next iteration.
            v0_r = v_k_r
            v0_i = v_k_i

            iteration += 1

        # --- After convergence, compute the final voltage magnitudes.
        v0_magnitude = torch.sqrt(v0_r**2 + v0_i**2)
        v0_clamped = v0_magnitude.view(batch_size, -1)

        return v0_magnitude


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
        alpha = 10.0
        
        max_ev_charge_power = smooth_min(
            max_ev_charge_power,
            ev_connected_binary * (battery_capacity -
                                   current_capacity) / timescale,
            alpha=alpha
        )
        max_ev_discharge_power = smooth_max(
            max_ev_discharge_power,
            ev_connected_binary *
            (ev_min_battery_capacity - current_capacity) / timescale,
            alpha=alpha
        )

        if self.verbose:
            print(f'prices: {prices}')
            print(f'current_capacity: {current_capacity}')
            print(f'max battery_capacity: {battery_capacity}')
            print(f'min_battery_capacity: {ev_min_battery_capacity}')
            print(f'time_left: {ev_time_left}')
            print(f'action: {action}')

        # 2) Smooth approximation of "action_binary = torch.where(action >= 0, 1, 0)"
        #    We use a smooth_step that transitions around 0.
        # in [0,1], ~1 if action >= 0
        action_smooth = smooth_step(action, alpha=alpha)

        # Original piecewise:
        #   power_usage = action * max_cs_power * action_binary
        #                - action * min_cs_power * (1 - action_binary)
        # We'll rewrite it using the smooth step:
        power_usage = (
            action_smooth * (action * self.max_cs_power) +
            (1.0 - action_smooth) * (-action * self.min_cs_power)
        )

        # 3) Smooth clamp instead of torch.min / torch.max
        power_usage = smooth_clamp(
            power_usage,
            lower=max_ev_discharge_power,
            upper=max_ev_charge_power,
            alpha=alpha
        )

        # 4) Smooth step for "new_ev_binary = torch.where(ev_time_left > 1, 1, 0)"
        #    We'll shift by 1 so that if ev_time_left is well above 1, the output is ~1,
        #    and if below 1, output is ~0.
        new_ev_smooth = smooth_step(ev_time_left - 1.0, alpha=alpha)

        if self.verbose:
            print(f'power_usage: {power_usage}')
            print(f'new_ev_smooth: {new_ev_smooth}')
            print(f'timescale: {timescale}')

        # 5) Build new_values with shape matching new_state
        new_values = torch.zeros_like(new_state, device=self.device)

        # For the EV states, we add: (current_capacity + power_usage * timescale) * new_ev_smooth
        #   -> if new_ev_smooth ~1, we keep updated capacity
        #   -> if new_ev_smooth ~0, we keep old capacity
        # This is a bit trickier if you need exact indexing.
        # We'll assume the same shape logic as your original code:
        new_values[:, ev_state_start:(ev_state_start + step_size*number_of_cs):step_size] = (
            current_capacity + power_usage * timescale
        ) * new_ev_smooth

        if self.verbose:
            print(f'new_values: {new_values}')

        # 6) Build the "mask" with a smooth approximation:
        #    originally: mask = torch.where(..., 1, 0) => 1 - new_ev_binary
        #    now: 1.0 - new_ev_smooth
        mask = torch.ones_like(new_state, device=self.device)
        mask[:, ev_state_start:(
            ev_state_start + step_size*number_of_cs):step_size] = 1.0 - new_ev_smooth

        # 7) Update new_state
        new_state = new_state * mask + new_values
        
        return new_state


def smooth_step(x, alpha=10.0):
    """
    Smooth approximation of a step function:
      step(x >= 0) ~ 0.5 * [1 + tanh(alpha * x)]
    - alpha controls how steeply we transition around x=0.
    - Larger alpha => closer to a hard step, but can lead to large gradients near 0.
    """
    return 0.5 * (1.0 + torch.tanh(alpha * x))

def smooth_min(a, b, alpha=10.0, eps=1e-6):
    """
    Smooth approximation of min(a, b):
      min(a,b) ~ 0.5 * (a + b - sqrt((a - b)^2 + eps))
    """
    diff = a - b
    return 0.5 * (a + b - torch.sqrt(diff * diff + eps))

def smooth_max(a, b, alpha=10.0, eps=1e-6):
    """
    Smooth approximation of max(a, b):
      max(a,b) ~ 0.5 * (a + b + sqrt((a - b)^2 + eps))
    """
    diff = a - b
    return 0.5 * (a + b + torch.sqrt(diff * diff + eps))

def smooth_clamp(x, lower, upper, alpha=10.0):
    """
    Clamp x into [lower, upper] with smooth_min/max.
      clamp(x, lower, upper) = min( max(x, lower), upper )
    """
    return smooth_min(
        smooth_max(x, lower, alpha=alpha),
        upper,
        alpha=alpha
    )
