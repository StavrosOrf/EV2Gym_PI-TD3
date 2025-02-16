import numpy as np

def V2G_grid_reward(env, total_costs, user_satisfaction_list, *args):
    
    reward = total_costs               
                
    for ev in env.departing_evs:
        reward += -10 * (ev.current_capacity - ev.desired_capacity)**2
        
    current_step = env.current_step - 1
    voltage_violation = np.reshape(env.node_voltage[:,current_step], (-1))        
   
    counter = np.where(voltage_violation > 1.05)[0]
    voltage_up_violation_counter = len(counter)
    
    counter = np.where(voltage_violation < 0.95)[0]
    voltage_down_violation_counter = len(counter)
            
    voltage_violation = np.sum(voltage_violation > 1.05) - 1.05 * voltage_up_violation_counter +\
                        np.sum(voltage_violation < 0.95) - 0.95 * voltage_down_violation_counter

    return reward - 1000 * voltage_violation

def V2G_grid_simple_reward(env, total_costs, user_satisfaction_list, *args):
    
    reward = total_costs               
                
    for ev in env.departing_evs:
        reward += -10 * (ev.current_capacity - ev.desired_capacity)**2
        
    current_step = env.current_step - 1
    voltage_violation = np.reshape(env.node_voltage[:,current_step], (-1))        
   
    counter = np.where(voltage_violation > 1.05)[0]
    voltage_up_violation_counter = len(counter)
    
    counter = np.where(voltage_violation < 0.95)[0]
    voltage_down_violation_counter = len(counter)
            
    voltage_violation = np.sum(voltage_violation > 1.05) - 1.05 * voltage_up_violation_counter +\
                        np.sum(voltage_violation < 0.95) - 0.95 * voltage_down_violation_counter

    return -1000 * voltage_violation

