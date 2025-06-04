import numpy as np
import gym
from enum import Enum

class EnvName(Enum):
    CARTPOLE = "CartPole-v1"
    # PENDULUM = "Pendulum-v1"
    MOUNTAINCAR = "MountainCar-v0"
    # ACROBOT = "Acrobot-v1"
    # LUNARLANDER = "LunarLander-v2"

# ---------------------
# Transition, Reward, and Terminal Functions
# ---------------------

def cartpole_transition(state, action):
    x, x_dot, theta, theta_dot = state
    force_mag = 10.0
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    length = 0.5
    tau = 0.02

    total_mass = masspole + masscart
    polemass_length = masspole * length

    force = force_mag if action == 1 else -force_mag
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc

    return np.array([x, x_dot, theta, theta_dot])

def cartpole_reward(state, action):
    return 1.0

def cartpole_is_terminal(state):
    x, _, theta, _ = state
    theta_threshold_radians = 12 * 2 * np.pi / 360
    x_threshold = 2.4
    return (
        x < -x_threshold or x > x_threshold or theta < -theta_threshold_radians or theta > theta_threshold_radians
    )

def pendulum_transition(state, action):
    th, thdot = state
    max_speed = 8
    dt = 0.05
    g = 10.0
    m = 1.0
    l = 1.0
    max_torque = 2.0

    u = np.clip(action, -max_torque, max_torque)

    newthdot = (
        thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l ** 2) * u) * dt
    )
    newthdot = np.clip(newthdot, -max_speed, max_speed)
    newth = th + newthdot * dt

    return np.array([np.cos(newth), np.sin(newth), newthdot])

def pendulum_reward(state, action):
    th, thdot = state
    u = np.clip(action, -2.0, 2.0)
    return -(th**2 + 0.1 * thdot**2 + 0.001 * u**2)

def pendulum_is_terminal(state):
    return False

def mountaincar_transition(state, action):
    position, velocity = state
    force = [-1.0, 0.0, 1.0][action]
    velocity += 0.001 * force - 0.0025 * np.cos(3 * position)
    velocity = np.clip(velocity, -0.07, 0.07)
    position += velocity
    position = np.clip(position, -1.2, 0.6)
    if position == -1.2 and velocity < 0:
        velocity = 0
    return np.array([position, velocity])

def mountaincar_reward(state, action):
    return -1.0

def mountaincar_is_terminal(state):
    position, _ = state
    return position >= 0.5

def wrap_angle(x):
    """
    Wrap x into [-π, π].
    """
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def acrobot_transition(obs, action):
    """
    Input:
      obs     6‐vector = [cos θ₁, sin θ₁, cos θ₂, sin θ₂, θ̇₁, θ̇₂]
      action  integer in {0,1,2}  (→ torque = {−1,0,+1})
    Output:
      next_obs 6‐vector = [cos θ₁′, sin θ₁′, cos θ₂′, sin θ₂′, θ̇₁′, θ̇₂′]
    
    This uses exactly the same forward‐Euler dynamics (τ = 0.2 s, torque = ±1 Nm) as Gym’s AcrobotEnv.
    """
    print(f'state: {obs}')
    # 1) Unpack incoming observation
    c1, s1, c2, s2, θ1_dot, θ2_dot = np.asarray(obs, dtype=float).ravel()
    # Recover the “raw” angles via atan2:
    θ1 = np.arctan2(s1, c1)
    θ2 = np.arctan2(s2, c2)

    # 2) Physical constants (identical to Gym’s AcrobotEnv)
    m1 = 1.0
    m2 = 1.0
    l1 = 1.0
    l2 = 1.0
    lc1 = 0.5
    lc2 = 0.5
    I1 = 1.0
    I2 = 1.0
    g = 9.8
    torque_mag = 1.0
    τ = 0.2

    # 3) Map discrete action ∈ {0,1,2} to continuous torque ∈ {−1,0,+1}
    torque = (int(action) - 1) * torque_mag

    # 4) Compute common terms (exactly as in Gym’s source)
    d1 = (
        m1 * lc1**2
        + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(θ2))
        + I1
        + I2
    )
    d2 = m2 * (lc2**2 + l1 * lc2 * np.cos(θ2)) + I2

    phi2 = m2 * lc2 * g * np.cos(θ1 + θ2 - np.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * (θ2_dot**2) * np.sin(θ2)
        - 2 * m2 * l1 * lc2 * θ1_dot * θ2_dot * np.sin(θ2)
        + (m1 * lc1 + m2 * l1) * g * np.cos(θ1 - np.pi / 2.0)
        + phi2
    )

    θ2_dd = (
        torque + (d2 / d1) * phi1 - phi2
    ) / (m2 * lc2**2 + I2 - d2**2 / d1)
    θ1_dd = -(d2 * θ2_dd + phi1) / d1

    # 5) Forward‐Euler integration
    new_θ1_dot = θ1_dot + θ1_dd * τ
    new_θ2_dot = θ2_dot + θ2_dd * τ
    new_θ1 = wrap_angle(θ1 + new_θ1_dot * τ)
    new_θ2 = wrap_angle(θ2 + new_θ2_dot * τ)

    # 6) Build the next observation (cos/sin + velocities)
    next_c1 = np.cos(new_θ1)
    next_s1 = np.sin(new_θ1)
    next_c2 = np.cos(new_θ2)
    next_s2 = np.sin(new_θ2)

    return np.array([next_c1, next_s1, next_c2, next_s2, new_θ1_dot, new_θ2_dot])


def acrobot_reward(state, action):
    """
    Stateless reward: −1.0 per time step, matching Gym’s default.
    """
    return -1.0


def acrobot_is_terminal(state):
    """
    Terminal condition when the “tip height” exceeds the threshold:
      -cos(θ1) - cos(θ1 + θ2) > 1.0
    This is exactly Gym’s test for “has the end-effector reached the goal?”
    """
    arr = np.asarray(state, dtype=float).ravel()
   # Recover angles
    θ1 = np.arctan2(arr[1], arr[0])  # atan2(sin θ₁, cos θ₁)
    θ2 = np.arctan2(arr[3], arr[2])  # atan2(sin θ₂, cos θ₂)
    return (-np.cos(θ1) - np.cos(θ1 + θ2)) > 1.0

def lunarlander_transition(state, action):
    from gym.envs.box2d.lunar_lander import LunarLander
    env = LunarLander()
    env.reset()
    env.lander.position = type(env.lander.position)(state[0], state[1])
    env.lander.linearVelocity = type(env.lander.linearVelocity)(state[2], state[3])
    env.lander.angle = state[4]
    env.lander.angularVelocity = state[5]
    env.lander.ground_contact = bool(state[6])
    env.lander.awake = True
    # This is a hack to trigger physics step with action
    env.step(action)
    obs = np.array([
        env.lander.position.x,
        env.lander.position.y,
        env.lander.linearVelocity.x,
        env.lander.linearVelocity.y,
        env.lander.angle,
        env.lander.angularVelocity,
        1.0 if env.lander.ground_contact else 0.0,
        0.0  # For second leg contact if needed
    ])
    return obs

def lunarlander_reward(state, action):
    return 0.0  # You may implement detailed reward if needed

def lunarlander_is_terminal(state):
    y = state[1]
    return y < 0  # crash if below ground

# ---------------------
# Central Test Function
# ---------------------

def test_environment(env_name: EnvName):
    env = gym.make(env_name.value)
    state, _ = env.reset()
    # if env_name == EnvName.PENDULUM:
    #     state = np.array([np.arctan2(state[1], state[0]), state[2]])

    transition_fn = {
        EnvName.CARTPOLE: cartpole_transition,
        # EnvName.PENDULUM: pendulum_transition,
        EnvName.MOUNTAINCAR: mountaincar_transition,
        # EnvName.ACROBOT: acrobot_transition,
        # EnvName.LUNARLANDER: lunarlander_transition,
    }[env_name]

    reward_fn = {
        EnvName.CARTPOLE: cartpole_reward,
        # EnvName.PENDULUM: pendulum_reward,
        EnvName.MOUNTAINCAR: mountaincar_reward,
        # EnvName.ACROBOT: acrobot_reward,
        # EnvName.LUNARLANDER: lunarlander_reward,
    }[env_name]

    done_fn = {
        EnvName.CARTPOLE: cartpole_is_terminal,
        # EnvName.PENDULUM: pendulum_is_terminal,
        EnvName.MOUNTAINCAR: mountaincar_is_terminal,
        # EnvName.ACROBOT: acrobot_is_terminal,
        # EnvName.LUNARLANDER: lunarlander_is_terminal,
    }[env_name]

    for i in range(2000):
        print(f"\n{env_name.value}| Step: {i}")
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated                

        # if env_name == EnvName.PENDULUM:
        #     next_state = np.array([np.arctan2(next_obs[1], next_obs[0]), next_obs[2]])            
        # else:
        next_state = next_obs
        print(f"Gym next state: {next_obs}")
        print(f"Gym next state: {state}")

        custom_next_state = transition_fn(state, action)
        custom_reward = reward_fn(state, action)
        custom_done = done_fn(custom_next_state)
        
        # if env_name == EnvName.PENDULUM:
        #     custom_next_state = [x.item() for x in custom_next_state]

        
        print(f"Action: {action}")
        print(f"Gym next: {next_obs}, Custom next: {custom_next_state}")
        print(f"Gym reward: {reward}, Custom reward: {custom_reward}")
        print(f"Gym done: {done}, Custom done: {custom_done}")

        assert np.allclose(next_state, custom_next_state, atol=1e-4), f"{env_name.value}: State mismatch"
        assert np.isclose(reward, custom_reward, atol=1e-4), f"{env_name.value}: Reward mismatch"
        assert done == custom_done, f"{env_name.value}: Terminal state mismatch"

        state = custom_next_state
        if done:
            print("Episode ended early due to terminal state.")
            break
        
        if truncated:
            print("Episode ended early due to truncated state.")
            break

# ---------------------
# Run All Tests
# ---------------------
if __name__ == "__main__":
    for env in EnvName:
        test_environment(env)
