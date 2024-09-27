import numpy as np  

# guide_policy.py

def guide_policy(world, gp_type):
    """Factory function to select the appropriate policy based on the version"""
    if gp_type == "formation":
        return guide_policy_formation(world)
    elif gp_type == "encirclement":
        return guide_policy_encirclement(world)
    elif gp_type == "navigation":
        return guide_policy_navigation(world)
    else:
        raise ValueError(f"Unknown policy version: {gp_type}")

def guide_policy_formation(world):
    egos = world.egos
    dynamic_obstacles = world.dynamic_obstacles
    obstacles = world.obstacles
    num_egos = len(egos)
    U = np.zeros((num_egos, 2, 1))

    edge_list = world.edge_list.tolist()
    edge_num = len(edge_list[1])  # each edge is calculated twice

    k1 = 0.3  # pos coefficient
    k2 = 0.1  # vel coefficient
    k3 = 0.3  # neighbor coefficient
    k4 = 0.8  # goal coefficient
    # Formation control
    for i, ego in enumerate(egos):
        if ego.is_leader:
            leader = ego
    for i, ego in enumerate(egos):
        # Get the neighbors of the ego
        neighbors_id = []  # the neighbor id of all entities, global id
        for j in range(edge_num):
            if edge_list[0][j] == ego.global_id:
                neighbors_id.append(edge_list[1][j])
            if edge_list[1][j] > ego.global_id:
                break
        nieghbors_ego = [e for e in egos if e.global_id in neighbors_id]
        neighbors_dobs = [d for d in dynamic_obstacles if d.global_id in neighbors_id]
        nieghbors_obs = [o for o in obstacles if o.global_id in neighbors_id]

        sum_epj = np.array([0., 0.])
        sum_evj = np.array([0., 0.])
        for nb_ego in nieghbors_ego:
            sum_epj = sum_epj + k3 * ((ego.state.p_pos - ego.formation_vector) - (nb_ego.state.p_pos - nb_ego.formation_vector))
            sum_evj = sum_evj + k3 * (ego.state.p_pos - nb_ego.state.p_pos)

        epL = ego.state.p_pos - leader.state.p_pos - ego.formation_vector
        evL = ego.state.p_vel - leader.state.p_vel
        v_L_dot = leader.action.u if leader.action.u is not None else np.array([0., 0.])

        u_i = - k1 * (epL + k3 * sum_epj) - k2 * (evL + k3 * sum_evj) + v_L_dot

        if ego.is_leader:
            u_i = u_i + k4 * (ego.goal - ego.state.p_pos)

        u_i = limit_action_inf_norm(u_i, 1)

        U[i] = u_i.reshape(2,1)

    return U


def guide_policy_encirclement(world):
    agents = world.agents

def guide_policy_navigation(world):
    agents = world.agents


def limit_action_inf_norm(action, max_limit):
    action = np.float32(action)
    action_ = action
    if abs(action[0]) > abs(action[1]):
        if abs(action[0])>max_limit:
            action_[1] = max_limit*action[1]/abs(action[0])
            action_[0] = max_limit if action[0] > 0 else -max_limit
        else:
            pass
    else:
        if abs(action[1])>max_limit:
            action_[0] = max_limit*action[0]/abs(action[1])
            action_[1] = max_limit if action[1] > 0 else -max_limit
        else:
            pass
    return action_
