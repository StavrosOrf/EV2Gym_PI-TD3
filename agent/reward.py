import numpy as np


def V2G_grid_full_reward(env, total_costs, user_satisfaction_list, *args):

    reward = total_costs
    
    verbose = False
    
    if verbose:
        print(f'!!! Costs: {total_costs}')
    
    user_costs = 0
    for ev in env.departing_evs:
        if verbose:
            print(f'!!! EV: {ev.current_capacity} | {ev.desired_capacity}')
        user_costs += -10 * (ev.current_capacity - ev.desired_capacity)**2
    
    if verbose:
        print(f'!!! User Satisfaction Penalty: {user_costs}')

    current_step = env.current_step - 1
    v_m = env.node_voltage[:, current_step]

    loss_v = np.minimum(np.zeros_like(v_m), 0.05 - np.abs(1-v_m)).sum()
    return reward + 1000 * loss_v + user_costs


def V2G_grid_simple_reward(env, total_costs, user_satisfaction_list, *args):

    reward = total_costs

    for ev in env.departing_evs:
        reward += -10 * (ev.current_capacity - ev.desired_capacity)**2

    current_step = env.current_step - 1
    v_m = env.node_voltage[:, current_step]

    loss_v = np.minimum(np.zeros_like(v_m), 0.05 - np.abs(1-v_m)).sum()

    return 1000 * loss_v
