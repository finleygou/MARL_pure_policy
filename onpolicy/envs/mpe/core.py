import numpy as np
import seaborn as sns
from .scenarios.util import *

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        # physical angle
        self.phi = 0  # 0-2pi
        # physical angular velocity
        self.p_omg = 0
        self.last_a = np.array([0, 0])
        # norm of physical velocity
        self.V = 0
        # 控制量（非加速度）：只需记录target，以便求attacker的policy_u
        self.controller = 0

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # index among all entities (important to set for distance caching)
        self.i = 0
        # name
        self.name = ''
        # properties:
        self.size = 1.0
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # entity can pass through non-hard walls
        self.ghost = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.max_angular = None
        self.max_accel = None
        self.accel = None
        # state: including internal/mental state p_pos, p_vel
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        # commu channel
        self.channel = None

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()
        self.R = None
        self.delta = None
        self.Ls = None
        self.movable = False

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()

        # agents are dummy
        self.dummy = False
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state: including communication state(communication utterance) c and internal/mental state p_pos, p_vel
        self.state = AgentState()
        # action: physical action u & communication action c
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # zoe 20200420
        self.goal = None
        # finley
        self.done = False
        self.policy_action = np.array([0,0])
        self.network_action = np.array([0,0])

class Target(Agent):
    def __init__(self):
        super(Target, self).__init__()
        self.name = 'target'
        self.id = None
        self.attackers = []  # attackers that are aiming him
        self.defenders = []  # defenders that help you avoid attacker
        self.cost = [] # to store cost of each tad cost
        self.attacker = None # local TAD combination
        self.defender = None

class Attacker(Agent):
    def __init__(self):
        super(Attacker, self).__init__()
        self.name = 'attacker'
        self.id = None
        self.true_target = None
        self.fake_target = None
        self.defenders = []
        self.flag_kill = False  # successful kill a target
        self.flag_dead = False  # being killed by defender
        self.last_belief = None
        self.is_locked = False  # whether lock the true target

class Defender(Agent):
    def __init__(self):
        super(Defender, self).__init__()
        self.name = 'defender'
        self.id = None
        self.attacker = None # attacker to defend at
        self.target = None # the target that your attacker is aiming at


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.walls = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping（阻尼）
        self.damping = 0 # 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None
        # finley
        self.world_length = 200
        self.world_step = 0
        self.num_agents = 0
        self.num_landmarks = 0

        self.targets = []
        self.defenders= []
        self.attackers = []
        self.cnt_dead = 0
        self.attacker_belief = []

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def calculate_distances(self):
        """
        cached_dist_vect: 类似图论矩阵, 记录i_a和i_b相对位置关系的向量
        """
        if self.cached_dist_vect is None:
            # initialize distance data structure
            self.cached_dist_vect = np.zeros((len(self.entities),
                                              len(self.entities),
                                              self.dim_p))
            # calculate minimum distance for a collision between all entities （size相加?）
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))  # N*N数组，N为智能体个数
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist
            # 实对称距离矩阵

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)

        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)  # bool

    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], 'dummy'):
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], 'adversary'):
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        # r g b
        dummy_colors = [(0.25, 0.75, 0.25)] * n_dummies
        # sns.color_palette("OrRd_d", n_adversaries)
        adv_colors = [(0.75, 0.25, 0.25)] * n_adversaries
        # sns.color_palette("GnBu_d", n_good_agents)
        good_colors = [(0.25, 0.25, 0.75)] * n_good_agents
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color

    # landmark color
    def assign_landmark_colors(self):
        for landmark in self.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])

    # update state of the world
    def step(self):

        self.world_step += 1
        # print("world step is {} ".format(self.world_step))

        # if self.world_step > 30:
        #     self.attackers[0].true_target = 0
        #     self.attackers[0].fake_target = 0

        # if self.world_step > 32:
        #     self.attackers[0].true_target = 0
        #     self.attackers[0].fake_target = 0
        #     self.attackers[1].true_target = 2
        #     self.attackers[1].fake_target = 2
        #     self.attackers[2].true_target = 3
        #     self.attackers[2].fake_target = 3
        #     self.attackers[3].true_target = 1
        #     self.attackers[3].fake_target = 1
        #     self.attackers[4].true_target = 3
        #     self.attackers[4].fake_target = 3

        # set actions for scripted agents
        for i, agent in enumerate(self.agents):
            if agent.name == 'target':
                action = agent.action_callback(agent, self.attackers[agent.attacker], self.defenders[agent.defender])
                agent.action.u = action
                # print("agent {} action is {}".format(agent.id, action))
            elif agent.name == 'attacker':
                action = agent.action_callback(self.targets[agent.fake_target], agent)
                agent.action.u = action
                # print("agent {} action is {}".format(agent.id, action))
            elif agent.name == 'defender':
                action = agent.action_callback(self.targets[self.attackers[agent.attacker].fake_target], self.attackers[agent.attacker], agent)
                agent.action.u = action
                # print("agent {} action is {}".format(agent.id, action))
            
        
        # gather forces applied to entities
        u = [None] * len(self.agents)  # 空数组, 存储所有TAD的action
        # apply agent physical controls
        u = self.apply_action_force(u)
        # integrate physical state
        self.integrate_state(u)

        # # calculate and store distances between all entities
        # if self.cache_dists:
        #     self.calculate_distances()

    # gather agent action forces
    def apply_action_force(self, u):
        # set applied forces
        '''
        for adversary agents, u = [ax, ay]; 
        for escaping agents, u = [Vx, Vy];
        '''
        for i, agent in enumerate(self.agents):
            u[i] = agent.action.u
        return u

    def integrate_state(self, u):  # u:[[1*2]...] 1*2n, [[ax, ay]...]
        for i, agent in enumerate(self.agents):
            # if agent.name == 'defender':
            #     agent.state.p_vel = np.array([0, 0])

            v = agent.state.p_vel + u[i] * self.dt
            if np.linalg.norm(v) != 0:
                v = v / np.linalg.norm(v) * agent.max_speed

            # print("{} {} done is {}".format(agent.name, agent.id, agent.done))
            if agent.done:
                agent.state.p_vel = np.array([0, 0]) # 速度清零
            else:
                v_x, v_y = v[0], v[1]
                theta = np.arctan2(v_y, v_x)
                if theta < 0:
                    theta += np.pi * 2
                # update phi
                agent.state.phi = theta
                agent.state.p_vel = np.array([v_x, v_y])
            agent.state.p_pos += agent.state.p_vel * self.dt

            # print("{} {} pos is {}".format(agent.name, agent.id, np.linalg.norm(agent.state.p_pos)))
