import csv
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from .multi_discrete import MultiDiscrete
from onpolicy import global_var as glv
from .scenarios.util import GetAcuteAngle

# update bounds to center around agent
cam_range = 8
INFO = []  # render时可视化数据用

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,  # 以上callback是通过MPE_env跑通的
                 done_callback=None, update_belief=None, 
                 post_step_callback=None,shared_viewer=True, 
                 discrete_action=False):
        # discrete_action为false,即指定动作为Box类型

        # set CL
        self.use_policy = 1
        self.use_CL = 0
        self.CL_ratio = 0
        self.Cp = 0.6  # 1.0 # 0.3
        self.JS_thre = 0

        # terminate
        self.is_ternimate = False

        self.world = world
        self.world_length = self.world.world_length
        self.current_step = 0
        self.agents = self.world.attackers
        # set required vectorized gym env property
        self.n = len(self.world.attackers)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback  
        self.post_step_callback = post_step_callback
        self.update_belief = update_belief

        # environment parameters
        # self.discrete_action_space = True
        self.discrete_action_space = discrete_action

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(
            world, 'discrete_action') else False
        # in this env, force_discrete_action == False because world do not have discrete_action

        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(
            world, 'collaborative') else False
        #self.shared_reward = False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in self.agents:
            # action space
            total_action = [[0, len(self.world.targets)-1], [0,1]]
            u_action_space = MultiDiscrete(total_action)
            self.action_space.append(u_action_space)
            
            # observation space
            obs_dim = len(observation_callback(agent, self.world))  # callback from senario, changeable
            share_obs_dim += obs_dim  # simple concatenate
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))  # [-inf,inf]
        
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]
        
        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    # step  this is  env.step()
    def step(self, action_n):  # action_n: action for all policy agents, concatenated, from MPErunner
        self.current_step += 1
        obs_n = []
        reward_n = []  # concatenated reward for each agent
        done_n = []
        info_n = []
        start_ratio = 0.80
        self.JS_thre = int(self.world_length*start_ratio*set_JS_curriculum(self.CL_ratio/self.Cp))

        # set action for poliy agents
        for i, agent in enumerate(self.agents):  # attacker
            self._set_action(action_n[i], agent, self.action_space[i])
        
        # advance world state
        self.world.step()  # core.step(), after done, all stop. 不能传参

        # record observation for each agent
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append([self._get_reward(agent)])
            done_n.append(self._get_done(agent))
            info = {'individual_reward': self._get_reward(agent)}
            env_info = self._get_info(agent)
            if 'fail' in env_info.keys():
                info['fail'] = env_info['fail']
            info_n.append(info)

        # all agents get total reward in cooperative case, if shared reward, all agents have the same reward, and reward is sum
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.n  # [[reward] [reward] [reward] ...]

        if self.post_step_callback is not None:
            self.post_step_callback(self.world)

        # supervise dones number and belief update
        terminate = []
        current_dead = 0
        attacker_belief = []
        for i, agent in enumerate(self.world.agents):
            if agent.name=='target':
                terminate.append(agent.done)
            if agent.name=='attacker':
                attacker_belief.append(agent.fake_target)
                agent.last_belief = agent.fake_target
            if agent.done:
                current_dead += 1
        
        self.is_ternimate = True if all(terminate) else False
        if self.is_ternimate:
            # 所有target都被kill
            done_n = [True] * self.n
        
        # re-assign goals for TADs
        if self.update_belief is not None and not all(done_n):  # 若全部targets or attackers都被kill，则不需要更新
            # self.current_step%10==0
            if  not self.world.attacker_belief == attacker_belief or current_dead > self.world.cnt_dead:
                # if there is change in attacker belief or some agent is killed
                self.update_belief(self.world)
                # print("update belief")

        self.world.cnt_dead = current_dead
        self.world.attacker_belief = attacker_belief

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        self.current_step = 0
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.attackers

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent, means it is dead
    # if all agents are done, then the episode is done before episode length is reached. in envwrapper
    def _get_done(self, agent):
        if self.done_callback is None:
            if self.current_step >= self.world_length:
                return True
            else:
                return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        pass
        # # process action
        # if isinstance(action_space, MultiDiscrete):
        #     attacker_belief = int(action[0])
        #     is_locked = action[1]
        #     if not agent.is_locked:
        #         if is_locked:
        #             agent.fake_target = agent.true_target
        #             agent.is_locked = True
        #         else:
        #             agent.fake_target = attacker_belief
        #     else:
        #         agent.fake_target = agent.true_target

    def _set_CL(self, CL_ratio):
        # 通过多进程set value，与env_wraapper直接关联，不能改。
        # 此处glv是这个进程中的！与mperunner中的并不共用。
        glv.set_value('CL_ratio', CL_ratio)
        self.CL_ratio = glv.get_value('CL_ratio')

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def render(self, mode='human', close=False):
        if close:
            # close any existic renderers
            for i, viewer in enumerate(self.viewers):
                if viewer is not None:
                    viewer.close()
                self.viewers[i] = None
            return []
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent:
                        continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' +
                                agent.name + ': ' + word + '   ')
            print(message)
        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from . import rendering
                print('sucessfully imported rendering')
                self.viewers[i] = rendering.Viewer(700, 700)
        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from . import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            self.line = {}
            self.comm_geoms = []
            for entity in self.world.entities:
                geom = rendering.make_circle(0.1)  # entity.size
                xform = rendering.Transform()

                entity_comm_geoms = []
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)

                    if not entity.silent:
                        dim_c = self.world.dim_c  # 0
                        # make circles to represent communication
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = (entity.size / dim_c)
                            offset.set_translation(ci * comm_size * 2 -
                                                   entity.size + comm_size, 0)
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)

                else:
                    geom.set_color(*entity.color)
                    if entity.channel is not None:
                        dim_c = self.world.dim_c
                        # make circles to represent communication
                        for ci in range(dim_c):
                            comm = rendering.make_circle(entity.size / dim_c)
                            comm.set_color(1, 1, 1)
                            comm.add_attr(xform)
                            offset = rendering.Transform()
                            comm_size = (entity.size / dim_c)
                            offset.set_translation(ci * comm_size * 2 -
                                                   entity.size + comm_size, 0)
                            comm.add_attr(offset)
                            entity_comm_geoms.append(comm)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
                self.comm_geoms.append(entity_comm_geoms)
            
            for wall in self.world.walls:
                corners = ((wall.axis_pos - 0.5 * wall.width, wall.endpoints[0]),
                           (wall.axis_pos - 0.5 *
                            wall.width, wall.endpoints[1]),
                           (wall.axis_pos + 0.5 *
                            wall.width, wall.endpoints[1]),
                           (wall.axis_pos + 0.5 * wall.width, wall.endpoints[0]))
                if wall.orient == 'H':
                    corners = tuple(c[::-1] for c in corners)
                geom = rendering.make_polygon(corners)
                if wall.hard:
                    geom.set_color(*wall.color)
                else:
                    geom.set_color(*wall.color, alpha=0.5)
                self.render_geoms.append(geom)

            # add geoms to viewer
            # for viewer in self.viewers:
            #     viewer.geoms = []
            #     for geom in self.render_geoms:
            #         viewer.add_geom(geom)
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                for entity_comm_geoms in self.comm_geoms:
                    for geom in entity_comm_geoms:
                        viewer.add_geom(geom)
        results = []
        for i in range(len(self.viewers)):
            from . import rendering

            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(
                -5, 30, -15, 15)
            # x_left, x_right, y_bottom, y_top
            
            
            ############################### csv save
            data_ = ()
            # for j in range(len(self.world.agents)):
            #     data_ = data_ + (j, self.world.agents[j].state.p_pos[0], self.world.agents[j].state.p_pos[1])
            # data_ = data_ + (self.q_md, self.q_md_dot)
            # INFO.append(data_)
            # #csv
            

            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                # 绘制agent速度
                self.line[e] = self.viewers[i].draw_line(entity.state.p_pos, entity.state.p_pos+entity.state.p_vel*1.0)

                # if entity.name == 'attacker' and not entity.done:
                #     self.line[e] = self.viewers[i].draw_line(entity.state.p_pos, self.world.targets[entity.fake_target].state.p_pos)

                if 'agent' in entity.name:
                    self.render_geoms[e].set_color(*entity.color, alpha=0.5)
                    self.line[e].set_color(*entity.color, alpha=0.5)

                    if not entity.silent:
                        for ci in range(self.world.dim_c):
                            color = 1 - entity.state.c[ci]
                            self.comm_geoms[e][ci].set_color(
                                color, color, color)
                else:
                    self.render_geoms[e].set_color(*entity.color)
                    if entity.channel is not None:
                        for ci in range(self.world.dim_c):
                            color = 1 - entity.channel[ci]
                            self.comm_geoms[e][ci].set_color(
                                color, color, color)
            
            m = len(self.line)
            for k, attacker in enumerate(self.world.attackers):
                if not attacker.done:
                    self.line[m+k] = self.viewers[i].draw_line(attacker.state.p_pos, self.world.targets[attacker.fake_target].state.p_pos)
                    self.line[m+k].set_color(*attacker.color, alpha=0.5)

            m = len(self.line)
            for k, defender in enumerate(self.world.defenders):
                if not defender.done:
                    self.line[m+k] = self.viewers[i].draw_line(defender.state.p_pos, self.world.attackers[defender.attacker].state.p_pos)
                    self.line[m+k].set_color(*defender.color, alpha=0.5)

            # render to display or array
            results.append(self.viewers[i].render(
                return_rgb_array=mode == 'rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(
                        distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx

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

def set_JS_curriculum(CL_ratio):
    # func_ = 1-CL_ratio
    k = 2.0
    delta = 1-(np.exp(-k*(-1))-np.exp(k*(-1)))/(np.exp(-k*(-1))+np.exp(k*(-1)))
    x = 2*CL_ratio-1
    y_mid = (np.exp(-k*x)-np.exp(k*x))/(np.exp(-k*x)+np.exp(k*x))-delta*x**3
    func_ = (y_mid+1)/2
    return func_
