import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Target, Obstacle, DynamicObstacle
from onpolicy.envs.mpe.scenario import BaseScenario
from onpolicy import global_var as glv
from scipy import sparse
import sys
from util import *

'''
4 egos
4 obstacles
4 dynamic obstacles
'''

class Scenario(BaseScenario):
    
    def __init__(self) -> None:
        super().__init__()

    # 设置agent,landmark的数量，运动属性。
    def make_world(self,args):
        self.cp = args.cp
        self.use_CL = args.use_curriculum  # 是否使用课程式训练(render时改为false)
        self.num_egos = args.num_agents  # formation agents
        self.num_target = args.num_target
        self.num_obs = args.num_obstacle
        self.num_dynamic_obs = args.num_dynamic_obs
        if not hasattr(args, "max_edge_dist"):
            self.max_edge_dist = 1
            print("_" * 60)
            print(
                f"Max Edge Distance for graphs not specified. "
                f"Setting it to {self.max_edge_dist}"
            )
            print("_" * 60)
        else:
            self.max_edge_dist = args.max_edge_dist  # setting the max edge distance for the graph

        world = World()
        world.world_length = args.episode_length
        world.graph_mode = args.graph_mode
        world.collaborative = True
        world.cache_dists = True # cache the distances between all entities
        # set any world properties first

        world.max_edge_dist = self.max_edge_dist
        world.egos = [Agent() for i in range(self.num_egos)]
        world.obstacles = [Obstacle() for i in range(self.num_obs)]
        world.dynamic_obstacles = [DynamicObstacle() for i in range(self.num_dynamic_obs)]
        world.agents = world.egos + world.dynamic_obstacles

        global_id = 0
        for i, ego in enumerate(world.egos):
            ego.id = i
            ego.size = 0.12
            ego.R = ego.size
            ego.color = np.array([0.95, 0.45, 0.45])
            ego.max_speed = 0.5
            ego.max_accel = 0.5
            ego.global_id = global_id
            global_id += 1

        for i, d_obs in enumerate(world.dynamic_obstacles):
            d_obs.id = i
            d_obs.color = np.array([0.95, 0.65, 0.0])
            d_obs.size = 0.12
            d_obs.R = d_obs.size
            d_obs.max_speed = 0.3
            d_obs.max_accel = 0.5
            d_obs.t = 0  # open loop, record time
            d_obs.global_id = global_id
            global_id += 1

        for i, obs in enumerate(world.obstacles):
            obs.id = i
            obs.color = np.array([0.45, 0.45, 0.95])
            obs.global_id = global_id
            global_id += 1

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        init_pos_ego = np.array([[0., 0.], [-1.0, 0.], [0., 1.0], [1.0, 0.]])
        init_pos_ego = init_pos_ego + np.random.randn(*init_pos_ego.shape)*0.05
        H = np.array([[0., 0.], [-1.0, 0.], [0., 1.0], [1.0, 0.]])
        for i, ego in enumerate(world.egos):
            if i==0:
                ego.is_leader = True
                ego.goal = np.array([0., 8.])
            else:
                ego.goal = np.array([0., 8.]) + H[i]
            ego.done = False
            ego.state.p_pos = init_pos_ego[i]
            ego.state.p_vel = np.array([0.0, 0.0])
            ego.state.V = np.linalg.norm(ego.state.p_vel)
            ego.state.phi = np.pi
            ego.formation_vector = H[i]

        init_pos_d_obs = np.array([[-3., 5.], [3., 3.5], [-3., 8.], [3., 6.5]])
        init_direction = np.array([[1., -0.5], [-1., -0.5], [1., -0.5], [-1., -0.5]])
        for i, d_obs in enumerate(world.dynamic_obstacles):
            d_obs.done = False
            d_obs.t = 0
            d_obs.delta = 0.08
            d_obs.state.p_pos = init_pos_d_obs[i]
            d_obs.direction = init_direction[i]
            d_obs.state.p_vel = d_obs.direction*d_obs.max_speed/np.linalg.norm(d_obs.direction)
            d_obs.action_callback = dobs_policy

        init_pos_obs = np.array([[-1.5, 1.5], [-0.8, 3.8], [0.4, 2.6], [1.8, 0.9]])
        sizes_obs = np.array([0.19, 0.25, 0.24, 0.22])
        for i, obs in enumerate(world.obstacles):
            obs.done = False
            obs.state.p_pos = init_pos_obs[i]
            obs.state.p_vel = np.array([0.0, 0.0])
            obs.R = sizes_obs[i]
            obs.delta = 0.08
            obs.Ls = obs.R + obs.delta  

        world.calculate_distances()
        self.update_graph(world)

    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        dist = np.linalg.norm(agent1.state.p_pos - agent2.state.p_pos)
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def target_defender(self, world):
        return [agent for agent in world.agents if not agent.name=='ego']

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def set_CL(self, CL_ratio):
        d_cap = 1.5
        if CL_ratio < self.cp:
            # print('in here Cd')
            self.d_cap = d_cap*(self.cd + (1-self.cd)*CL_ratio/self.cp)
        else:
            self.d_cap = d_cap

    # reward function
    def reward(self, agent, world):
        '''
        思路：主要是碰撞、队形奖励。如果是leader，到达目标点奖励，且目标点给予大家更大的队形奖励。
        此处的奖励是针对每个agent的。
        '''
        rew = 1
        return rew

    # observation for policy agents
    def observation(self, agent, world):
        obs = np.array([1, 1])
        return obs

    def done(self, agent, world):  # 
        if agent.is_leader:
            dist = np.linalg.norm(agent.state.p_pos - agent.goal)
            if dist < 0.1:
                agent.done = True
                return True
        else:
            for ego in world.egos:
                if ego.is_leader:
                    if ego.done:
                        agent.done = True
                        return True
        agent.done = False

        return False
    
    def update_graph(self, world: World):
        """
        Construct a graph from the cached distances.
        Nodes are entities in the environment
        Edges are constructed by thresholding distances
        """
        dists = world.cached_dist_mag
        # just connect the ones which are within connection
        # distance and do not connect to itself
        connect = np.array((dists <= self.max_edge_dist) * (dists > 0)).astype(int)
        sparse_connect = sparse.csr_matrix(connect)
        sparse_connect = sparse_connect.tocoo()
        row, col = sparse_connect.row, sparse_connect.col
        edge_list = np.stack([row, col])
        world.edge_list = edge_list
        world.edge_weight = dists[row, col]   

def dobs_policy(agent, obstacles):
    action = agent.action
    dt = 0.1
    agent.t += dt
    if agent.t > 20:
        agent.done = True
    if agent.done:
        target_v = np.linalg.norm(agent.state.p_vel)
        if target_v < 1e-3:
            acc = np.array([0,0])
        else:
            acc = -agent.state.p_vel/target_v*agent.max_accel
        a_x, a_y = acc[0], acc[1]
        v_x = agent.state.p_vel[0] + a_x*dt
        v_y = agent.state.p_vel[1] + a_y*dt
        escape_v = np.array([v_x, v_y])
    else:
        max_speed = agent.max_speed
        esp_direction = agent.direction/np.linalg.norm(agent.direction)

        # with obstacles
        d_min = 1.0  # only consider the nearest obstacle, within 1.0m
        for obs in obstacles:
            dist_ = np.linalg.norm(agent.state.p_pos - obs.state.p_pos)
            if dist_ < d_min:
                d_min = dist_
                nearest_obs = obs
        if d_min < 1.0:
            d_vec_ij = agent.state.p_pos - nearest_obs.state.p_pos
            d_vec_ij = 0.5 * d_vec_ij / np.linalg.norm(d_vec_ij) / (np.linalg.norm(d_vec_ij) - nearest_obs.R - agent.R)
            if np.dot(d_vec_ij, esp_direction) < 0:
                d_vec_ij = d_vec_ij - np.dot(d_vec_ij, esp_direction) / np.dot(esp_direction, esp_direction) * esp_direction
        else:
            d_vec_ij = np.array([0, 0])
        esp_direction = esp_direction + d_vec_ij

        # with dynamic obstacles

        esp_direction = esp_direction/np.linalg.norm(esp_direction)
        a_x, a_y = esp_direction[0]*agent.max_accel, esp_direction[1]*agent.max_accel
        v_x = agent.state.p_vel[0] + a_x*dt
        v_y = agent.state.p_vel[1] + a_y*dt
        # 检查速度是否超过上限
        if abs(v_x) > max_speed:
            v_x = max_speed if agent.state.p_vel[0]>0 else -max_speed
        if abs(v_y) > max_speed:
            v_y = max_speed if agent.state.p_vel[1]>0 else -max_speed
        escape_v = np.array([v_x, v_y])

        # print("exp_direction:", esp_direction)

    action.u = escape_v
    return action