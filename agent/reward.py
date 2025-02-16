import numpy as np


def V2G_grid_reward(env, total_costs, user_satisfaction_list, *args):

    reward = total_costs

    for ev in env.departing_evs:
        reward += -10 * (ev.current_capacity - ev.desired_capacity)**2

    current_step = env.current_step - 1
    voltage_violation = np.reshape(env.node_voltage[:, current_step], (-1))

    v_m = env.node_voltage[:, current_step]

    loss_v = np.minimum(np.zeros_like(v_m), 0.05 - np.abs(1-v_m)).sum()
    return reward + 1000 * loss_v


def V2G_grid_simple_reward(env, total_costs, user_satisfaction_list, *args):

    reward = total_costs

    for ev in env.departing_evs:
        reward += -10 * (ev.current_capacity - ev.desired_capacity)**2

    current_step = env.current_step - 1
    v_m = env.node_voltage[:, current_step]

    loss_v = np.minimum(np.zeros_like(v_m), 0.05 - np.abs(1-v_m)).sum()

    return 1000 * loss_v
