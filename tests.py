import numpy as np
import gym

# CartPole Transition Function
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

# CartPole Reward Function
def cartpole_reward(state, action):
    x, x_dot, theta, theta_dot = state
    theta_threshold_radians = 12 * 2 * np.pi / 360
    x_threshold = 2.4

    done = bool(
        x < -x_threshold
        or x > x_threshold
        or theta < -theta_threshold_radians
        or theta > theta_threshold_radians
    )

    return 1.0 if not done else 0.0

# Pendulum Transition Function with Observation Output
def pendulum_transition_obs(state, action):
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

# Pendulum Reward Function
def pendulum_reward(state, action):
    th, thdot = state
    u = np.clip(action, -2.0, 2.0)
    cost = th ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)
    return -cost

# Test functions

def test_cartpole():
    env = gym.make("CartPole-v1")
    state, _ = env.reset()

    for i in range(500):
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)

        custom_next_state = cartpole_transition(state, action)
        custom_reward = cartpole_reward(custom_next_state, action)

        print("\nCartPole")
        print(f"Action: {action}")
        print(f"Gym next state:     {next_state}")
        print(f"Custom next state:  {custom_next_state}")
        print(f"Gym reward:         {reward}")
        print(f"Custom reward:      {custom_reward}")

        assert np.allclose(next_state, custom_next_state), "CartPole state mismatch"
        

        state = next_state

        if done:
            print(f"Episode ended early due to terminal state at step {i}.")
            break
        
        assert np.isclose(reward, custom_reward), "CartPole reward mismatch"

    print("CartPole tests passed.")

def test_pendulum():
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset()

    state = np.array([np.arctan2(obs[1], obs[0]), obs[2]])

    for i in range(500):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        custom_next_obs = pendulum_transition_obs(state, action)
        custom_next_obs = [x.item() for x in custom_next_obs]
        
        next_state = np.array([np.arctan2(next_obs[1], next_obs[0]), next_obs[2]])
        custom_reward = pendulum_reward(state, action)

        print("\nPendulum")
        print(f"Action: {action}")
        print(f"Gym next obs:       {next_obs}")
        print(f"Custom next obs:    {custom_next_obs}")
        print(f"Gym reward:         {reward}")
        print(f"Custom reward:      {custom_reward}")

        assert np.allclose(next_obs, custom_next_obs, atol=1e-4), "Pendulum observation mismatch"
        assert np.isclose(reward, custom_reward, atol=1e-4), "Pendulum reward mismatch"

        state = next_state

        if done:
            print(f"Episode ended early due to terminal state at step {i}.")
            break

    print("Pendulum tests passed.")

if __name__ == "__main__":
    test_cartpole()
    # test_pendulum()

