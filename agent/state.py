import math
import numpy as np
import math


def V2G_grid_state(env, *args):
    '''
    This is the state function for the PST_V2GProfitMax scenario.
    '''

    state = [
        env.sim_date.weekday() / 7,
        math.sin(env.sim_date.hour/24*2*math.pi),
        math.cos(env.sim_date.hour/24*2*math.pi),
    ]

    # state.append(env.current_power_usage[env.current_step-1])

    if env.current_step < env.simulation_length:
        charge_prices = abs(env.charge_prices[0, env.current_step])
    else:
        charge_prices = 0

    state.append(charge_prices)

    state.append(env.node_voltage[:, env.current_step-1])
    state.append(env.node_active_power[:, env.current_step-1])

    # For every transformer
    for tr in env.transformers:

        # state.append(tr.get_power_limits(env.current_step,horizon=1))

        # For every charging station connected to the transformer
        for cs in env.charging_stations:
            if cs.connected_transformer == tr.id:
                # state.append(cs.min_charge_current)
                # state.append(cs.max_charge_current)
                # state.append(cs.n_ports)

                # For every EV connected to the charging station
                for EV in cs.evs_connected:
                    # If there is an EV connected
                    if EV is not None:
                        state.append([
                            EV.get_soc(),
                            EV.time_of_departure - env.current_step,
                        ])

                    # else if there is no EV connected put zeros
                    else:
                        state.append(np.zeros(2))

    state = np.array(np.hstack(state))

    return state


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


def PST_V2G_ProfitMaxGNN_state(env, *args):
    ''' 
    The state function of the profit maximization model with V2G capabilities for the GNN models.
    '''

    PST_V2G_ProfitMaxGNN_state.node_sizes = {
        'ev': 5, 'cs': 4, 'tr': 2, 'env': 6}

    # create the graph of the environment having as nodes the CPO, the transformers, the charging stations and the EVs connected to the charging stations

    ev_features = []
    cs_features = []
    tr_features = []
    env_features = []

    env_features = [
        env.sim_date.weekday() / 7,
        # turn hour and minutes in sin and cos
        math.sin(env.sim_date.hour/24*2*math.pi),
        math.cos(env.sim_date.hour/24*2*math.pi),
    ]

    if env.current_step < env.simulation_length:
        setpoint = env.power_setpoints[env.current_step]
    else:
        setpoint = 0

    env_features.append(setpoint)
    env_features.append(env.current_power_usage[env.current_step-1])

    node_counter = 0

    if env.current_step < env.simulation_length:
        env_features.append(abs(env.charge_prices[0, env.current_step]))
    else:
        env_features.append(0)

    env_features = [env_features]

    node_features = [env_features]
    node_types = [0]
    node_counter += 1
    node_names = ['env']

    ev_indexes = []
    cs_indexes = []
    tr_indexes = []
    env_indexes = [0]

    action_mapper = []  # It is a list that maps the node index to the action index

    edge_index_from = []
    edge_index_to = []

    port_counter = 0
    mapper = {}
    # Map tr.id, cs.id, ev.id to node index
    for cs in env.charging_stations:
        n_ports = cs.n_ports
        for i in range(n_ports):
            mapper[f'Tr_{cs.connected_transformer}_CS_{cs.id}_EV_{i}'] = port_counter + i

        port_counter += n_ports

    for tr in env.transformers:
        # If EV is connected to the charging station that is connected to the transformer
        # Then include transformer id, EV id, EV soc, EV total energy exchanged, EV max charge power, EV min charge power, time of arrival
        registered_tr = False

        for cs in env.charging_stations:
            registered_CS = False

            if cs.connected_transformer == tr.id:
                for EV in cs.evs_connected:
                    if EV is not None:

                        if not registered_CS:
                            registered_CS = True

                            charger_features = [cs.min_charge_current,
                                                cs.max_charge_current,
                                                cs.n_ports,
                                                cs.id
                                                ]

                            if not registered_tr:

                                node_features.append([tr.max_power[env.current_step] -
                                                      tr.inflexible_load[env.current_step] +
                                                      tr.solar_power[env.current_step],
                                                      tr.id
                                                      ])
                                tr_features.append([tr.max_power[env.current_step] -
                                                    tr.inflexible_load[env.current_step] +
                                                    tr.solar_power[env.current_step],
                                                    tr.id
                                                    ])

                                tr_indexes.append(node_counter)
                                node_counter += 1
                                node_types.append(1)
                                node_names.append(f'Tr_{tr.id}')
                                tr_node_index = len(node_names)-1

                                edge_index_from.append(0)
                                edge_index_to.append(tr_node_index)

                                edge_index_from.append(tr_node_index)
                                edge_index_to.append(0)

                                registered_tr = True

                            node_features.append(charger_features)
                            cs_features.append(charger_features)

                            cs_indexes.append(node_counter)
                            node_counter += 1
                            node_types.append(2)
                            node_names.append(f'Tr_{tr.id}_CS_{cs.id}')
                            cs_node_index = len(node_names)-1

                            edge_index_from.append(tr_node_index)
                            edge_index_to.append(cs_node_index)

                            edge_index_from.append(cs_node_index)
                            edge_index_to.append(tr_node_index)

                            registered_CS = True

                        node_features.append([EV.get_soc(),
                                              EV.time_of_departure - env.current_step,
                                              EV.id,
                                              cs.id,
                                              tr.id
                                              ])
                        ev_features.append([EV.get_soc(),
                                            EV.time_of_departure - env.current_step,
                                            EV.id,
                                            cs.id,
                                            tr.id
                                            ])

                        ev_indexes.append(node_counter)
                        node_counter += 1

                        node_types.append(3)
                        action_mapper.append(
                            mapper[f'Tr_{tr.id}_CS_{cs.id}_EV_{EV.id}'])
                        node_names.append(f'Tr_{tr.id}_CS_{cs.id}_EV_{EV.id}')
                        ev_node_index = len(node_names)-1

                        edge_index_from.append(cs_node_index)
                        edge_index_to.append(ev_node_index)

                        edge_index_from.append(ev_node_index)
                        edge_index_to.append(cs_node_index)

            # map the edge node names from edge_index_from and edge_index_to to integers

    edge_index = [edge_index_from, edge_index_to]

    data = Data(ev_features=np.array(ev_features).reshape(-1, 5).astype(float),
                cs_features=np.array(cs_features).reshape(-1, 4).astype(float),
                tr_features=np.array(tr_features).reshape(-1, 2).astype(float),
                env_features=np.array(
                    env_features).reshape(-1, 6).astype(float),
                edge_index=np.array(edge_index).astype(int),
                node_types=np.array(node_types).astype(int),
                sample_node_length=[len(node_features)],
                action_mapper=action_mapper,
                ev_indexes=np.array(ev_indexes),
                cs_indexes=np.array(cs_indexes),
                tr_indexes=np.array(tr_indexes),
                env_indexes=np.array(env_indexes),
                )

    return data


def PST_V2G_ProfitMax_state_to_GNN(state, config, *args):
    '''
    This function converts the state of the PST_V2GProfitMax scenario to a GNN state similar to the output of the PST_V2G_ProfitMaxGNN_state function.

    Input:
        state: np.array
        config: the simulation config, which provides structural information

    Output:
        data: torch_geometric.data.Data
    '''

    PST_V2G_ProfitMax_state_to_GNN.node_sizes = {
        'ev': 5, 'cs': 4, 'tr': 2, 'env': 6}

    assert config['number_of_ports_per_cs'] == 1, 'This function only supports one port per charging station.'
    idx = 0

    # Extract environment features
    env_features = state[idx:idx+6]
    idx += 6

    node_features = [env_features]
    node_types = [0]  # 0 for env node
    node_names = ['env']
    node_counter = 1

    edge_index_from = []
    edge_index_to = []

    ev_features = []
    cs_features = []
    tr_features = []

    ev_indexes = []
    cs_indexes = []
    tr_indexes = []
    env_indexes = [0]

    action_mapper = []
    cs_counter = -1
    ev_counter = 0

    for tr in range(config['number_of_transformers']):
        any_evs_per_tr = False
        # Get transformer feature from state
        tr_feature = state[idx]
        idx += 1

        tr_features.append([tr_feature, tr])
        node_features.append([tr_feature, tr])
        node_types.append(1)  # 1 for transformer node
        node_names.append(f'Tr_{tr}')
        tr_node_index = node_counter
        tr_indexes.append(tr_node_index)
        node_counter += 1

        # Add edge between env and transformer
        edge_index_from.append(0)
        edge_index_to.append(tr_node_index)
        edge_index_from.append(tr_node_index)
        edge_index_to.append(0)

        chargers_per_tr = int(
            config['number_of_charging_stations'])/int(config['number_of_transformers'])

        if chargers_per_tr != int(chargers_per_tr):
            raise ValueError(
                'The number of charging stations must be divisible by the number of transformers.')

        chargers_per_tr = int(chargers_per_tr)
        for cs in range(chargers_per_tr):
            cs_counter += 1
            # Get charging station features from state
            cs_min_charge_current = state[idx]
            idx += 1
            cs_max_charge_current = state[idx]
            idx += 1
            cs_n_ports = int(state[idx])
            idx += 1
            # print(f'cs {cs} : {cs_min_charge_current} {cs_max_charge_current} {cs_n_ports}')
            # check if EVs are connected to the charging station
            if state[idx] == 0 and state[idx+1] == 0:
                idx += 2
                ev_counter += cs_n_ports
                continue

            cs_features.append(
                [cs_min_charge_current, cs_max_charge_current, cs_n_ports, cs_counter])
            node_features.append(
                [cs_min_charge_current, cs_max_charge_current, cs_n_ports, cs_counter])
            node_types.append(2)  # 2 for charging station node
            node_names.append(f'Tr_{tr}_CS_{cs_counter}')
            cs_node_index = node_counter
            cs_indexes.append(cs_node_index)
            node_counter += 1

            # Add edge between transformer and charging station
            edge_index_from.append(tr_node_index)
            edge_index_to.append(cs_node_index)
            edge_index_from.append(cs_node_index)
            edge_index_to.append(tr_node_index)

            # For each port (EV slot) in the charging station
            for port_i in range(cs_n_ports):
                EV_soc = state[idx]
                idx += 1
                EV_tod = state[idx]
                idx += 1

                EV_id = port_i  # Using port index as EV ID for simplicity
                ev_features.append([EV_soc, EV_tod, EV_id, cs_counter, tr])
                node_features.append([EV_soc, EV_tod, EV_id, cs_counter, tr])
                node_types.append(3)  # 3 for EV node
                node_names.append(f'Tr_{tr}_CS_{cs_counter}_EV_{EV_id}')
                ev_node_index = node_counter
                ev_indexes.append(ev_node_index)
                action_mapper.append(ev_counter)
                node_counter += 1

                # Add edge between charging station and EV
                edge_index_from.append(cs_node_index)
                edge_index_to.append(ev_node_index)
                edge_index_from.append(ev_node_index)
                edge_index_to.append(cs_node_index)

                ev_counter += 1
                any_evs_per_tr = True

        if not any_evs_per_tr:
            edge_index_from = edge_index_from[:-2]
            edge_index_to = edge_index_to[:-2]
            tr_features = tr_features[:-1]
            tr_indexes = tr_indexes[:-1]
            node_names = node_names[:-1]
            node_types = node_types[:-1]
            node_features = node_features[:-1]
            node_counter -= 1

    # print(f'idx: {idx}')
    # print(f'len(state): {len(state)}')
    # if idx != len(state):
    #     raise ValueError('The state was not fully processed.')

    # if len(ev_features) == 0:
    #     edge_index_from = []
    #     edge_index_to = []
    #     tr_features = []
    #     tr_indexes = []
    #     node_features = [env_features]
    #     node_types = [0]

    # Construct edge_index tensor
    edge_index = np.array([edge_index_from, edge_index_to], dtype=int)

    # Convert lists to numpy arrays
    ev_features_array = np.array(
        ev_features, dtype=float) if ev_features else np.empty((0, 5))
    cs_features_array = np.array(
        cs_features, dtype=float) if cs_features else np.empty((0, 4))
    tr_features_array = np.array(
        tr_features, dtype=float) if tr_features else np.empty((0, 2))
    env_features_array = np.array([env_features], dtype=float)

    node_types_array = np.array(node_types, dtype=int)
    action_mapper_array = np.array(action_mapper, dtype=int)
    ev_indexes_array = np.array(ev_indexes, dtype=int)
    cs_indexes_array = np.array(cs_indexes, dtype=int)
    tr_indexes_array = np.array(tr_indexes, dtype=int)
    env_indexes_array = np.array(env_indexes, dtype=int)

    data = Data(
        ev_features=ev_features_array,
        cs_features=cs_features_array,
        tr_features=tr_features_array,
        env_features=env_features_array,
        edge_index=edge_index,
        node_types=node_types_array,
        sample_node_length=[len(node_features)],
        action_mapper=action_mapper_array,
        ev_indexes=ev_indexes_array,
        cs_indexes=cs_indexes_array,
        tr_indexes=tr_indexes_array,
        env_indexes=env_indexes_array,
    )

    return data
