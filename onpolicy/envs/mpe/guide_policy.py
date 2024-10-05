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
    elif gp_type == "formation_rvo":
        return guide_policy_formation_rvo(world)
    elif gp_type == "encirclement_rvo":
        return guide_policy_encirclement_rvo(world)
    elif gp_type == "navigation_rvo":
        return guide_policy_navigation_rvo(world)
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

    k1 = 0.5  # pos coefficient
    k2 = 0.1  # vel coefficient
    k3 = 0.3  # neighbor coefficient
    k4 = 0.8  # goal coefficient
    k_obs = 0.6  # obstacle coefficient
    k_b = 0.5  # damping coefficient
    # Formation control
    for i, ego in enumerate(egos):
        if ego.is_leader:
            leader = ego
    for i, ego in enumerate(egos):
        # Get the neighbors of the ego
        neighbors_id = []  # the neighbor id of all entities, global id
        for j in range(edge_num):
            if int(edge_list[0][j]) == ego.global_id:
                neighbors_id.append(edge_list[1][j])
            if int(edge_list[0][j]) > ego.global_id:
                break

        # print("ego", i, "neighbors_id", neighbors_id)
        neighbors_ego = [e for e in egos if e.global_id in neighbors_id]
        neighbors_dobs = [d for d in dynamic_obstacles if d.global_id in neighbors_id]
        neighbors_obs = [o for o in obstacles if o.global_id in neighbors_id]

        sum_epj = np.array([0., 0.])
        sum_evj = np.array([0., 0.])
        for nb_ego in neighbors_ego:
            sum_epj = sum_epj + k3 * ((ego.state.p_pos - ego.formation_vector) - (nb_ego.state.p_pos - nb_ego.formation_vector))
            sum_evj = sum_evj + k3 * (ego.state.p_pos - nb_ego.state.p_pos)

        epL = ego.state.p_pos - leader.state.p_pos - ego.formation_vector
        evL = ego.state.p_vel - leader.state.p_vel
        v_L_dot = leader.action.u if leader.action.u is not None else np.array([0., 0.])

        f_fom = - k1 * (epL + k3 * sum_epj) - k2 * (evL + k3 * sum_evj) + v_L_dot

        f_obs = np.array([0., 0.])
        for nb_obs in neighbors_obs:
            d_ij = ego.state.p_pos - nb_obs.state.p_pos
            norm_d_ij = np.linalg.norm(d_ij)
            L_min = ego.R + nb_obs.R + nb_obs.delta
            Ls = L_min + 0.5
            if norm_d_ij < Ls:
                f_obs = f_obs + k_obs*(Ls-norm_d_ij)/norm_d_ij*d_ij

        f_dobs = np.array([0., 0.])
        for nb_dobs in neighbors_dobs:
            r_ij = ego.state.p_pos - nb_dobs.state.p_pos
            norm_r_ij = np.linalg.norm(r_ij)
            relative_velocity = ego.state.p_vel - nb_dobs.state.p_vel
            L_min = ego.R + nb_dobs.R + nb_dobs.delta
            Ls = L_min + 0.5  
            if norm_r_ij < Ls:
                relative_speed_in_r_dir = np.dot(relative_velocity, r_ij) / norm_r_ij
                if relative_speed_in_r_dir < 0:
                    f_dobs = f_dobs + k_obs * (Ls - norm_r_ij) / norm_r_ij * r_ij

        f_egos = np.array([0., 0.])
        for nb_ego in neighbors_ego:
            d_ij = ego.state.p_pos - nb_ego.state.p_pos
            norm_d_ij = np.linalg.norm(d_ij)
            L_min = ego.R + nb_ego.R
            Ls = L_min + 0.25
            if norm_d_ij < Ls:
                f_egos = f_egos + k_obs*(Ls-norm_d_ij)/norm_d_ij*d_ij

        u_i = f_fom + f_egos + f_obs + f_dobs - k_b*ego.state.p_vel


        if ego.is_leader:
            u_i = u_i + k4 * (ego.goal - ego.state.p_pos)

        u_i = limit_action_inf_norm(u_i, 1)

        U[i] = u_i.reshape(2,1)

    return U

def guide_policy_encirclement(world):
    egos = world.egos
    target = world.targets[0]  # only one target
    dynamic_obstacles = world.dynamic_obstacles
    obstacles = world.obstacles
    num_egos = len(egos)
    U = np.zeros((num_egos, 2, 1))

    edge_list = world.edge_list.tolist()
    edge_num = len(edge_list[1])  # each edge is calculated twice

    d_cap = egos[0].d_cap
    L = 2*d_cap*np.sin(np.pi/len(egos))
    k_ic = 1.2 # 2.0
    k_icv = 1.0  #
    k_ij = 1.5  # 4.5
    k_b = 0.8  # 速度阻尼
    k_obs = 1.5

    for i, ego in enumerate(egos):
        # Get the neighbors of the ego
        neighbors_id = []  # the neighbor id of all entities, global id
        for j in range(edge_num):
            if int(edge_list[0][j]) == ego.global_id:
                neighbors_id.append(edge_list[1][j])
            if int(edge_list[0][j]) > ego.global_id:
                break

        # print("ego", i, "neighbors_id", neighbors_id)
        neighbors_ego = [e for e in egos if e.global_id in neighbors_id]
        neighbors_dobs = [d for d in dynamic_obstacles if d.global_id in neighbors_id]
        neighbors_obs = [o for o in obstacles if o.global_id in neighbors_id]

        # 与目标之间的吸引力
        r_ic = target.state.p_pos - ego.state.p_pos
        norm_r_ic = np.linalg.norm(r_ic)
        vel_vec = target.state.p_vel - ego.state.p_vel
        if norm_r_ic - d_cap > 0:
            if norm_r_ic - d_cap > 1.5:
                f_c = 1.5/norm_r_ic*r_ic + k_icv*vel_vec
            else:
                f_c = k_ic*(norm_r_ic - d_cap)/norm_r_ic*r_ic + k_icv*vel_vec
        else:  # 不能穿过目标
            f_c = 5 * k_ic * (norm_r_ic - d_cap) / norm_r_ic * r_ic + k_icv * vel_vec

        f_r = np.array([0, 0])
        for adv in neighbors_ego:
            r_ij = ego.state.p_pos - adv.state.p_pos
            norm_r_ij = np.linalg.norm(r_ij)
            if norm_r_ij < L:
                f_ = k_ij*(L - norm_r_ij)/norm_r_ij*r_ij
                if np.dot(f_, r_ic) < 0 and norm_r_ij > 2*L/3:  # 把与目标方向相反的部分力给抵消了
                    f_ = f_ - np.dot(f_, r_ic) / np.dot(r_ic, r_ic) * r_ic
                f_r = f_r + f_

        f_obs = np.array([0, 0])
        for nb_obs in neighbors_obs:
            d_ij = ego.state.p_pos - nb_obs.state.p_pos
            norm_d_ij = np.linalg.norm(d_ij)
            L_min =egos[0].R + nb_obs.R + nb_obs.delta
            Ls = L_min+0.3
            if norm_d_ij < Ls:
                f_obs = f_obs + k_obs*(Ls-norm_d_ij)/norm_d_ij*d_ij

        f_dobs = np.array([0., 0.])
        for nb_dobs in neighbors_dobs:
            r_ij = ego.state.p_pos - nb_dobs.state.p_pos
            norm_r_ij = np.linalg.norm(r_ij)
            relative_velocity = ego.state.p_vel - nb_dobs.state.p_vel
            L_min = ego.R + nb_dobs.R + nb_dobs.delta
            Ls = L_min + 0.5  
            if norm_r_ij < Ls:
                relative_speed_in_r_dir = np.dot(relative_velocity, r_ij) / norm_r_ij
                if relative_speed_in_r_dir < 0:
                    f_dobs = f_dobs + k_obs * (Ls - norm_r_ij) / norm_r_ij * r_ij


        u_i = f_c + f_r + f_obs + f_dobs - k_b*ego.state.p_vel

        u_i = limit_action_inf_norm(u_i, 1)

        U[i] = u_i.reshape(2,1)

    return U

def guide_policy_navigation(world):
    egos = world.egos
    dynamic_obstacles = world.dynamic_obstacles
    obstacles = world.obstacles
    num_egos = len(egos)
    U = np.zeros((num_egos, 2, 1))

    edge_list = world.edge_list.tolist()
    edge_num = len(edge_list[1])  # each edge is calculated twice

    k1 = 0.5  # goal coefficient
    k_obs = 1.5  # obstacle coefficient
    k_b = 1.6  # damping coefficient

    for i, ego in enumerate(egos):
        # Get the neighbors of the ego
        neighbors_id = []  # the neighbor id of all entities, global id
        for j in range(edge_num):
            if int(edge_list[0][j]) == ego.global_id:
                neighbors_id.append(edge_list[1][j])
            if int(edge_list[0][j]) > ego.global_id:
                break

        f_goal = k1 * (ego.goal - ego.state.p_pos)

        # print("ego", i, "neighbors_id", neighbors_id)
        neighbors_ego = [e for e in egos if e.global_id in neighbors_id]
        neighbors_dobs = [d for d in dynamic_obstacles if d.global_id in neighbors_id]
        neighbors_obs = [o for o in obstacles if o.global_id in neighbors_id]

        f_obs = np.array([0., 0.])
        for nb_obs in neighbors_obs:
            d_ij = ego.state.p_pos - nb_obs.state.p_pos
            norm_d_ij = np.linalg.norm(d_ij)
            L_min = ego.R + nb_obs.R + nb_obs.delta
            Ls = L_min + 0.7
            if norm_d_ij < Ls:
                f_obs = f_obs + k_obs*(Ls-norm_d_ij)/norm_d_ij*d_ij

        f_dobs = np.array([0., 0.])
        for nb_dobs in neighbors_dobs:
            r_ij = ego.state.p_pos - nb_dobs.state.p_pos
            norm_r_ij = np.linalg.norm(r_ij)
            relative_velocity = ego.state.p_vel - nb_dobs.state.p_vel
            L_min = ego.R + nb_dobs.R + nb_dobs.delta
            Ls = L_min + 0.5  
            if norm_r_ij < Ls:
                relative_speed_in_r_dir = np.dot(relative_velocity, r_ij) / norm_r_ij
                if relative_speed_in_r_dir < 0:
                    f_dobs = f_dobs + k_obs * (Ls - norm_r_ij) / norm_r_ij * r_ij

        f_egos = np.array([0., 0.])
        for nb_ego in neighbors_ego:
            d_ij = ego.state.p_pos - nb_ego.state.p_pos
            norm_d_ij = np.linalg.norm(d_ij)
            L_min = ego.R + nb_ego.R
            Ls = L_min + 0.3
            if norm_d_ij < Ls:
                f_egos = f_egos + k_obs*(Ls-norm_d_ij)/norm_d_ij*d_ij

        u_i = f_goal + f_egos + f_obs + f_dobs - k_b*ego.state.p_vel

        u_i = limit_action_inf_norm(u_i, 1)

        U[i] = u_i.reshape(2,1)

    return U

def guide_policy_formation_rvo(world):
    pass

def guide_policy_encirclement_rvo(world):
    pass

def guide_policy_navigation_rvo(world):
    pass

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
