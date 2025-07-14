import math
import numpy as np
import math

def V2G_grid_state_ModelBasedRL(env, *args):
    '''
    This is the state function for the V2GProfitMax scenario with loads
    '''

    state = [
        env.sim_date.weekday() / 7,
        math.sin(env.sim_date.hour/24*2*math.pi),
        math.cos(env.sim_date.hour/24*2*math.pi),
    ]

    step_ahead = 1

    # state.append(env.current_power_usage[env.current_step-1])

    charge_prices = env.charge_prices[0, env.current_step:
                                       env.current_step+step_ahead]

    if len(charge_prices) < step_ahead:
        charge_prices = np.append(
            charge_prices, np.zeros(step_ahead-len(charge_prices)))
    
    state.append(charge_prices)
    
    if env.current_step < env.simulation_length:
        setpoint = env.power_setpoints[env.current_step]
    else:
        setpoint = 0

    state.append(setpoint)

    state.append(env.current_power_usage[env.current_step-1])
    
    step = env.current_step
    if step == 0:
        state.append(env.node_active_power[1:, step])
        state.append(env.node_reactive_power[1:, step])
    else:
        state.append(env.node_active_power[1:, step-1])
        state.append(env.node_reactive_power[1:, step-1])

    # print('============\nstep:', env.current_step, '\n============')

    # For every transformer
    # for tr in env.transformers:
        # For every charging station connected to the transformer
    for cs in env.charging_stations:
        for EV in cs.evs_connected:
            # If there is an EV connected
            if EV is not None:
                state.append([
                    EV.current_capacity,
                    EV.time_of_departure - env.current_step + 1,
                    cs.connected_bus,
                ])

            # else if there is no EV connected put zeros
            else:
                state.append(np.zeros(3))

    state = np.array(np.hstack(state))

    return state

