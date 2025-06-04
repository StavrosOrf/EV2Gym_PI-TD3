import torch
import gym
from gym.vector import SyncVectorEnv
torch.autograd.set_detect_anomaly(True)

# Precompute constants
PI = torch.tensor(3.141592653589793, dtype=torch.float64)
TWO_PI = 2.0 * PI

# -------------------------------------
# Batched Transition, Reward, Terminal
# -------------------------------------

def mountaincar_continuous_transition_batch(state: torch.Tensor, action: torch.Tensor, **kwargs) -> torch.Tensor:
    pos = state[:, 0]            # [B]
    vel = state[:, 1]            # [B]
    u = action.view(-1).clamp(-1.0, 1.0)    # [B]

    vel = vel + u * 0.0015 - 0.0025 * torch.cos(3.0 * pos)
    vel = vel.clamp(-0.07, 0.07)

    pos = pos + vel
    pos = pos.clamp(-1.2, 0.6)

    mask = (pos <= -1.2) & (vel < 0)
    vel = torch.where(mask, torch.zeros_like(vel), vel)

    return torch.stack((pos, vel), dim=1)  # [B,2]


def mountaincar_continuous_reward_batch(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    pos = state[:, 0]                     # [B]
    u = action.view(-1).clamp(-1.0, 1.0)   # [B]
    base = -0.1 * (u * u)                 # [B]
    bonus = torch.where(pos >= 0.45, 100.0 * torch.ones_like(pos), torch.zeros_like(pos))
    return base + bonus                   # [B]


def mountaincar_continuous_is_terminal_batch(state: torch.Tensor) -> torch.Tensor:
    pos = state[:, 0]    # [B]
    return pos >= 0.45   # [B] (bool)


def pendulum_state_batch(obs: torch.Tensor) -> torch.Tensor:
    """
    Convert batched Pendulum-v1 observations [B,3] = [cosθ, sinθ, θ̇]
    into batched states [B,2] = [θ, θ̇].
    """
    cos_th = obs[:, 0]
    sin_th = obs[:, 1]
    thdot = obs[:, 2]
    th = torch.atan2(sin_th, cos_th)
    return torch.stack((th, thdot), dim=1)  # [B,2]


def pendulum_transition_batch(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    th = state[:, 0]              # [B]
    thdot = state[:, 1]           # [B]
    u = action.view(-1).clamp(-2.0, 2.0)  # [B]

    g = 10.0
    m = 1.0
    l = 1.0
    dt = 0.05
    max_speed = 8.0

    theta_dd = (-3.0 * g / (2.0 * l)) * torch.sin(th + PI) + (3.0 / (m * (l**2))) * u
    new_thdot = thdot + theta_dd * dt
    new_thdot = new_thdot.clamp(-max_speed, max_speed)
    new_th = th + new_thdot * dt
    new_th = torch.remainder(new_th + PI, TWO_PI) - PI

    return torch.stack((new_th, new_thdot), dim=1)  # [B,2]


def pendulum_reward_batch(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    th = state[:, 0]                  # [B]
    thdot = state[:, 1]               # [B]
    u = action.view(-1).clamp(-2.0, 2.0)  # [B]
    return - (th**2 + 0.1 * (thdot**2) + 0.001 * (u**2))  # [B]


def pendulum_is_terminal_batch(state: torch.Tensor) -> torch.Tensor:
    B = state.shape[0]
    return torch.zeros(B, dtype=torch.bool)  # never terminates early


# -------------------------------------
# Helper to create Vector Env
# -------------------------------------

def make_vector_env(env_id: str, num_envs: int):
    """
    Create a SyncVectorEnv with num_envs parallel copies of env_id.
    """
    def make_fn():
        return gym.make(env_id)
    return SyncVectorEnv([make_fn for _ in range(num_envs)])


# -------------------------------------
# Testing Batched vs. Gym Vector
# -------------------------------------

def test_batched_env(env_id: str, batch_size: int = 8, max_steps: int = 2000):
    """
    Test our batched transition/reward/terminal against Gym's VectorEnv:
      - env_id: "MountainCarContinuous-v0" or "Pendulum-v1"
      - batch_size: number of parallel envs (B)
      - max_steps: maximum steps before breaking
    """
    # 1) Create a VectorEnv
    vec_env = make_vector_env(env_id, batch_size)
    obs, _ = vec_env.reset()         # shape [B, obs_dim]

    # 2) Convert initial observations into batched states
    if env_id == "MountainCarContinuous-v0":
        # obs.shape = [B,2]
        state = torch.from_numpy(obs).to(torch.float64)  # [B,2]
    elif env_id == "Pendulum-v1":
        # obs.shape = [B,3] = [cosθ, sinθ, θ̇]
        obs_t = torch.from_numpy(obs).to(torch.float64)  # [B,3]
        state = pendulum_state_batch(obs_t)              # [B,2]
    else:
        raise ValueError(f"Unsupported env: {env_id}")

    for step in range(max_steps):
        # 3) Sample a batch of actions from the VectorEnv
        action_batch_np = vec_env.action_space.sample()  # shape [B, action_dim] 
        # Flatten to [B]
        action_batch = torch.from_numpy(action_batch_np).view(batch_size).to(torch.float64)

        # 4) Step Gym vector env
        next_obs_np, reward_np, done_np, truncated_np, info = vec_env.step(action_batch_np)
        # reward_np: [B], done_np: [B]

        # 5) Convert Gym outputs to torch tensors
        rewards_true = torch.from_numpy(reward_np).to(torch.float64)  # [B]
        dones_true = torch.from_numpy(done_np).to(torch.bool)         # [B]
        truncated_true = torch.from_numpy(truncated_np).to(torch.bool)  # [B]

        if env_id == "MountainCarContinuous-v0":
            next_state_true = torch.from_numpy(next_obs_np).to(torch.float64)  # [B,2]
            next_state_custom = mountaincar_continuous_transition_batch(state, action_batch)
            reward_custom = mountaincar_continuous_reward_batch(state, action_batch)
            done_custom = mountaincar_continuous_is_terminal_batch(next_state_custom)

        else:  # Pendulum-v1
            next_obs_t = torch.from_numpy(next_obs_np).to(torch.float64)  # [B,3]
            next_state_true = pendulum_state_batch(next_obs_t)            # [B,2]
            next_state_custom = pendulum_transition_batch(state, action_batch)
            reward_custom = pendulum_reward_batch(state, action_batch)
            done_custom = pendulum_is_terminal_batch(next_state_custom)

        #multiply state by truncated_true to make zeros
        next_state_custom = next_state_custom * (~truncated_true).view(-1, 1)
        next_state_true = next_state_true * (~truncated_true).view(-1, 1)
        
        # 6) Compare
        state_match = torch.allclose(next_state_true, next_state_custom, atol=1e-3)
        reward_match = torch.allclose(rewards_true, reward_custom, atol=1e-2)
        done_match = torch.equal(dones_true, done_custom)

        print(f"\n{env_id} | Step {step}")
        print(f"Gym next_state:\n {next_state_true}")
        print(f"Custom next_state:\n {next_state_custom}")
        print(f"Gym reward: {rewards_true}")
        print(f"Custom reward: {reward_custom}")
        print(f"Gym done: {dones_true}")
        print(f"Custom done: {done_custom}")
        print(f"Gym truncated: {truncated_true}")
        # print(f"Custom truncated: {truncated_custom}")
        


        assert state_match,  f"{env_id}: state mismatch at step {step}"
        assert reward_match, f"{env_id}: reward mismatch at step {step}"
        assert done_match,   f"{env_id}: done mismatch at step {step}"

        # 7) Update state for next iteration
        state = next_state_custom.clone()

        # 8) If any envs are done, reset those
        if dones_true.any() or truncated_true.any():
            # Reset only the environments that finished
            # Gym's VectorEnv supports partial resets via `reset_done=True` in recent versions
            # Otherwise, we manually reset and replace states.
            # Here: after reset, VectorEnv returns a full batch; we then replace only the done indices in state.
            reset_obs,_ = vec_env.reset()
            # Convert reset_obs into state_batch
            if env_id == "MountainCarContinuous-v0":
                reset_states = torch.from_numpy(reset_obs).to(torch.float64)
            else:  # Pendulum-v1
                reset_obs_t = torch.from_numpy(reset_obs).to(torch.float64)
                reset_states = pendulum_state_batch(reset_obs_t)
            # Replace state entries where done is True
            mask = dones_true | truncated_true
            state[mask, :] = reset_states[mask, :]

    print(f"\nAll {env_id} steps matched custom batch functions!")


if __name__ == "__main__":
    batch_size = 8

    # print("\n=== Testing MountainCarContinuous-v0 with batch size", batch_size, "===")
    # test_batched_env("MountainCarContinuous-v0", batch_size)

    print("\n=== Testing Pendulum-v1 with batch size", batch_size, "===")
    test_batched_env("Pendulum-v1", batch_size)
