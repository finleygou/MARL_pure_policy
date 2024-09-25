import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Target, Attacker, Defender
from onpolicy.envs.mpe.scenario import BaseScenario
from onpolicy import global_var as glv
from util import *

'''
距离单位: km
时间单位: s
'''
Mach = 0.34  # km/s
G = 9.8e-3  # km/s^2

global a1, b1, c1, d1, N_m
a1 = 0
b1 = 1e10
c1 = 0.5
d1 = 0.5
N_m = 4

class Scenario(BaseScenario):
    
    def __init__(self) -> None:
        super().__init__()
        self.cd = 1.0  # 取消Cd
        self.cp = 0.4
        self.cr = 1.0  # 取消Cr
        self.d_cap = 1.0 # 期望围捕半径,动态变化,在set_CL里面
        self.init_target_pos = 2.0
        self.use_CL = 0  # 是否使用课程式训练(render时改为false)
        self.assign_list = [] # 为attackers初始化分配target, 长度为num_A, 值为target的id
        self.init_assign = True  # 仅在第一次分配时使用

    # 设置agent,landmark的数量，运动属性。
    def make_world(self,args):
        self.num_target = args.num_target
        self.num_attacker = args.num_attacker
        self.num_defender = args.num_defender

        world = World()
        world.collaborative = True
        # set any world properties first
        # add agents
        world.targets = [Target() for i in range(self.num_target)]  # 3
        world.attackers = [Attacker() for i in range(self.num_attacker)]  # 3
        world.defenders = [Defender() for i in range(self.num_defender)]  # 4
        world.agents = world.targets+world.attackers+world.defenders

        for i, target in enumerate(world.targets):
            target.id = i
            target.size = 0.1
            target.color = np.array([0.45, 0.95, 0.45]) #greem
            target.max_speed = 0.8*Mach
            target.max_accel = 5*G
            target.action_callback = target_policy

        for i, attacker in enumerate(world.attackers):
            attacker.id = i
            attacker.size = 0.1
            attacker.color = np.array([0.95, 0.45, 0.45])
            attacker.max_speed = 3*Mach
            attacker.max_accel = 15*G
            attacker.action_callback = attacker_policy

        for i, defender in enumerate(world.defenders):
            defender.id = i
            defender.size = 0.1
            defender.color = np.array([0.45, 0.45, 0.95])
            defender.max_speed = 2*Mach
            defender.max_accel = 15*G
            defender.action_callback = defender_policy

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        self.init_assign = True
        self.assign_list = rand_assign_targets(self.num_target, self.num_attacker)
        # print('init assign_list is:', self.assign_list)
        self.assign_list = [1, 0, 2, 3, 2]

        # properties and initial states for agents
        for i, target in enumerate(world.targets):
            if i == 0: 
                target.done = False
                target.state.p_pos = np.array([0.0, 4.5])
                target.state.p_vel = np.array([target.max_speed, 0.0])
                target.state.V = np.linalg.norm(target.state.p_vel)
                target.state.phi = 0.
                target.attacker = [j for j in range(self.num_attacker) if self.assign_list[j]==i]  # the id of attacker in world.attackers
                # target.defender = 0  # init in update_belief
            elif i == 1:
                target.done = False
                target.state.p_pos = np.array([-1.0, 1.5])
                target.state.p_vel = np.array([target.max_speed, 0.0])
                target.state.V = np.linalg.norm(target.state.p_vel)
                target.state.phi = 0.
                target.attacker = [j for j in range(self.num_attacker) if self.assign_list[j]==i]
            elif i == 2:
                target.done = False
                target.state.p_pos = np.array([-0.5, -1.8])
                target.state.p_vel = np.array([target.max_speed, 0.0])
                target.state.V = np.linalg.norm(target.state.p_vel)
                target.state.phi = 0.
                target.attacker = [j for j in range(self.num_attacker) if self.assign_list[j]==i]
            elif i == 3:
                target.done = False
                target.state.p_pos = np.array([1.2, -4.2])
                target.state.p_vel = np.array([target.max_speed, 0.0])
                target.state.V = np.linalg.norm(target.state.p_vel)
                target.state.phi = 0.
                target.attacker = [j for j in range(self.num_attacker) if self.assign_list[j]==i]

        for i, attacker in enumerate(world.attackers):
            if i == 0: 
                attacker.done = False
                attacker.state.p_pos = np.array([20.0, 6.])
                attacker.state.p_vel = np.array([-attacker.max_speed, 0.0])
                attacker.state.V = np.linalg.norm(attacker.state.p_vel)
                attacker.state.phi = np.pi
                attacker.true_target = self.assign_list[i]
                attacker.fake_target = self.assign_list[i]
            elif i == 1:
                attacker.done = False
                attacker.state.p_pos = np.array([20.0, 3.])
                attacker.state.p_vel = np.array([-attacker.max_speed, 0.0])
                attacker.state.V = np.linalg.norm(attacker.state.p_vel)
                attacker.state.phi = np.pi
                attacker.true_target = self.assign_list[i]
                attacker.fake_target = self.assign_list[i]
            elif i == 2:
                attacker.done = False
                attacker.state.p_pos = np.array([20.0, 0.])
                attacker.state.p_vel = np.array([-attacker.max_speed, 0.0])
                attacker.state.V = np.linalg.norm(attacker.state.p_vel)
                attacker.state.phi = np.pi
                attacker.true_target = self.assign_list[i]
                attacker.fake_target = self.assign_list[i]
            elif i == 3:
                attacker.done = False
                attacker.state.p_pos = np.array([20.0, -3.])
                attacker.state.p_vel = np.array([-attacker.max_speed, 0.0])
                attacker.state.V = np.linalg.norm(attacker.state.p_vel)
                attacker.state.phi = np.pi
                attacker.true_target = self.assign_list[i]
                attacker.fake_target = self.assign_list[i]
            elif i == 4:
                attacker.done = False
                attacker.state.p_pos = np.array([20.0, -6.])
                attacker.state.p_vel = np.array([-attacker.max_speed, 0.0])
                attacker.state.V = np.linalg.norm(attacker.state.p_vel)
                attacker.state.phi = np.pi
                attacker.true_target = self.assign_list[i]
                attacker.fake_target = self.assign_list[i]
        
        for i, defender in enumerate(world.defenders):
            if i == 0: 
                defender.done = False
                defender.state.p_pos = np.array([8., 5.2])
                defender.state.p_vel = np.array([defender.max_speed, 0.0])
                defender.state.V = np.linalg.norm(defender.state.p_vel)
                defender.state.phi = 0
            elif i == 1:
                defender.done = False
                defender.state.p_pos = np.array([6.5, 3.])
                defender.state.p_vel = np.array([defender.max_speed, 0.0])
                defender.state.V = np.linalg.norm(defender.state.p_vel)
                defender.state.phi = 0
            elif i == 2:
                defender.done = False
                defender.state.p_pos = np.array([6., -3.2])
                defender.state.p_vel = np.array([defender.max_speed, 0.0])
                defender.state.V = np.linalg.norm(defender.state.p_vel)
                defender.state.phi = 0
            elif i == 3:
                defender.done = False
                defender.state.p_pos = np.array([7.5, -6.])
                defender.state.p_vel = np.array([defender.max_speed, 0.0])
                defender.state.V = np.linalg.norm(defender.state.p_vel)
                defender.state.phi = 0
            elif i == 4:   
                defender.done = False
                defender.state.p_pos = np.array([7., 0.1])
                defender.state.p_vel = np.array([defender.max_speed, 0.0])
                defender.state.V = np.linalg.norm(defender.state.p_vel)
                defender.state.phi = 0
            elif i == 5:
                defender.done = False
                defender.state.p_pos = np.array([9., 3.6])
                defender.state.p_vel = np.array([defender.max_speed, 0.0])
                defender.state.V = np.linalg.norm(defender.state.p_vel)
                defender.state.phi = 0
            elif i == 6:
                defender.done = False
                defender.state.p_pos = np.array([9., -4.])
                defender.state.p_vel = np.array([defender.max_speed, 0.0])
                defender.state.V = np.linalg.norm(defender.state.p_vel)
                defender.state.phi = 0

        self.update_belief(world)

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
        return [agent for agent in world.agents if not agent.name=='attacker']

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
    
    # agent 和 adversary 分别的reward
    def reward(self, agent, world):
        if agent.name == 'attacker':
            main_reward = self.attacker_reward(agent, world)  # else self.agent_reward(agent, world)
        else:
            print('reward error')
        return main_reward

    # individual adversary award
    def attacker_reward(self, attacker, world):  # agent here is adversary
        # 暂时假定只有一组TAD
        # 只考虑attacker、target或defender、attacker间的碰撞

        return 1


    # observation for adversary agents
    def observation(self, agent, world):
        return np.zeros((1))

    def done(self, agent, world):  # 
        # for attackers only
        if self.is_collision(agent, world.targets[agent.true_target]) and not agent.done:
            agent.done = True  # 在env中改变dones
            agent.flag_kill = True
            world.targets[agent.true_target].done = True
            return True
        
        for i, defender_id in enumerate(agent.defenders):
            if self.is_collision(agent, world.defenders[defender_id]) and not agent.done:
                agent.done = True
                world.defenders[defender_id].done = True
                return True
        
        # 要加下面这句话，否则done一次后前面两个if不会进入，立刻又变成False了。
        if agent.done:
            return True
        
        agent.done = False
        return False

    def update_belief(self, world):
        target_list = []
        defender_list = []
        attacker_list = []
        target_id_list = []
        defender_id_list = []
        attacker_id_list = []  
        for target in world.targets:
            if not target.done:
                target_list.append(target)
                target_id_list.append(target.id)
                target.defenders = []
                target.attackers = [] # 重新分配ADs
                target.cost = []
        for defender in world.defenders:
            if not defender.done:
                defender_list.append(defender)
                defender_id_list.append(defender.id)
        for attacker in world.attackers:
            if not attacker.done:
                attacker.defenders = []
                attacker_list.append(attacker)
                attacker_id_list.append(attacker.id)
        
        if len(defender_list) > 0:
            # calculate the cost matrix
            T = np.zeros((len(defender_list), len(attacker_list)))
            if self.init_assign:
                for i, defender in enumerate(defender_list):
                    for j, attacker in enumerate(attacker_list):
                        T[i, j] = get_init_cost(attacker, defender, world.targets[attacker.fake_target])
                self.init_assign = False      
            else:
                for i, defender in enumerate(defender_list):
                    for j, attacker in enumerate(attacker_list):
                        T[i, j] = get_energy_cost(attacker, defender, world.targets[attacker.fake_target])
            # print('T is:', T)
            # print('TAD list are:', target_id_list, defender_id_list, attacker_id_list)
            assign_result = target_assign(T)  # |D|*|A|的矩阵
            # print('assign_result is:', assign_result)

            '''
            如果assign报错，检查'TAD list，查看A的list是否为空。如果全部A被拦截，不需要update belief
            '''
            # update belief list of TDs according to assign_result
            for i, defender in enumerate(defender_list):  # 遍历行
                for j in range(len(attacker_list)):
                    if assign_result[i, j] == 1:
                        defender.attacker = attacker_list[j].id
                        defender.target = attacker_list[j].fake_target
                        attacker_list[j].defenders.append(defender.id)
                        target = world.targets[attacker_list[j].fake_target]
                        target.defenders.append(defender.id)
                        target.attackers.append(attacker_list[j].id)
                        target.cost.append(T[i, j])
            
            # 为target从list中选择AD
            for i, target in enumerate(target_list):
                if len(target.cost)>0:
                    target.defender = target.defenders[np.argmin(target.cost)]
                    target.attacker = target.attackers[np.argmin(target.cost)]
                else:
                    # 有的target已经不需要AD了，AD太少了
                    target.defender = np.random.choice(defender_id_list)
                    target.attacker = np.random.choice(attacker_id_list)

            
            # print('T believes are:', )


'''
low-level policy for TADs
'''
def target_policy(target, attacker, defender):
    global a1, b1, c1, d1, N_m

    if target.done or attacker.done or defender.done:
        return np.array([0, 0])

    # 期望攻击角度
    q_md_expect = 0.

    # 各方状态
    V_m = attacker.state.p_vel
    V_t = target.state.p_vel
    V_d = defender.state.p_vel
    e_vm = V_m / np.linalg.norm(V_m)
    e_vt = V_t / np.linalg.norm(V_t)
    e_vd = V_d / np.linalg.norm(V_d)
    x_mt = target.state.p_pos - attacker.state.p_pos
    r_mt = np.linalg.norm(x_mt)
    e_mt = x_mt / r_mt
    x_md = defender.state.p_pos - attacker.state.p_pos
    r_md = np.linalg.norm(x_md)
    e_md = x_md / r_md
    q_md = np.arctan2(x_md[1], x_md[0])
    q_md = q_md + np.pi * 2 if q_md < 0 else q_md  # 0~2pi

    # TAD基本关系
    r_mt_dot = np.dot(V_t - V_m, e_mt)
    q_mt_dot = np.cross(e_mt, V_t - V_m) / r_mt

    r_md_dot = np.dot(V_d - V_m, e_md) # negative
    q_md_dot = np.cross(e_md, V_d - V_m) / r_md

    # 旧的状态量
    x_11 = q_md - q_md_expect
    x_12 = q_md_dot

    # 中间参数
    M1 = N_m * np.dot(e_md, e_vm) / (5 * np.dot(e_mt, e_vm))
    P1 = N_m * np.dot(e_md, e_vm) * r_mt_dot * q_mt_dot / (np.dot(e_mt, e_vm) * r_md)
    L1 = 1 / c1 + M1 * M1 / d1
    A1 = L1 * a1 / (8 * r_md_dot * r_md_dot) * (1 - 2 * np.e * np.e + np.power(np.e, 4))
    B1 = L1 * b1 / (4 * r_md * r_md_dot) * (1 - np.power(np.e, 4)) + 1
    C1 = np.e * np.e * x_12 - r_md * P1 * (1 - np.e * np.e) / (2 * r_md_dot)
    D1 = L1 * a1 * r_md / (16 * np.power(r_md_dot, 3)) * (4 + np.power(np.e, -2) - np.e * np.e) - 1
    E1 = r_md / (2 * r_md_dot) * (np.power(np.e, -2) - 1) + L1 * b1 / (8 * r_md_dot * r_md_dot) * (np.e * np.e + np.power(np.e, -2) - 2)
    F1 = - x_11 - r_md * r_md * P1 * (1 + np.power(np.e, -2)) / (4 * r_md_dot * r_md_dot)

    # 新的状态量
    x_11_tf = (B1 * F1 - C1 * E1) / (B1 * D1 - A1 * E1)
    x_12_tf = (C1 * D1 - A1 * F1) / (B1 * D1 - A1 * E1)

    # 协态量
    # lamda_11 = a1 * x_11_tf
    lamda_12 = a1 * x_11_tf * r_md / (2 * r_md_dot) * (1 - np.e * np.e) + b1 * x_12_tf * np.e * np.e

    # 加速度
    v_q = - M1 * lamda_12 / (d1 * r_md)
    target.state.controller = v_q
    # target
    e_vq = - np.array([- e_mt[1], e_mt[0]])  # 和attacker不同，因为方向是mt，不是tm，故加负号。
    if np.linalg.norm(V_t) == 0:  # done
        a_t = np.array([0, 0])
    else:
        e_at = np.array([- e_vt[1], e_vt[0]])
        if GetAcuteAngle(e_at, e_vq) > np.pi / 2:
            e_at = np.array([e_vt[1], - e_vt[0]])
        a_t_value = np.clip(v_q / np.abs(np.dot(-e_mt, e_vt)), - target.max_accel, target.max_accel)
        a_t = np.multiply(e_at, a_t_value)

    return a_t

def defender_policy(target, attacker, defender):
    global a1, b1, c1, d1, N_m

    if target.done or attacker.done or defender.done:
        return np.array([0, 0])

    # 期望攻击角度
    q_md_expect = 0.

    # 各方状态
    V_m = attacker.state.p_vel
    V_t = target.state.p_vel
    V_d = defender.state.p_vel
    e_vm = V_m / np.linalg.norm(V_m)
    e_vt = V_t / np.linalg.norm(V_t)
    e_vd = V_d / np.linalg.norm(V_d)
    x_mt = target.state.p_pos - attacker.state.p_pos
    r_mt = np.linalg.norm(x_mt)
    e_mt = x_mt / r_mt
    x_md = defender.state.p_pos - attacker.state.p_pos
    r_md = np.linalg.norm(x_md)
    e_md = x_md / r_md
    q_md = np.arctan2(x_md[1], x_md[0])
    q_md = q_md + np.pi * 2 if q_md < 0 else q_md  # 0~2pi

    # TAD基本关系
    r_mt_dot = np.dot(V_t - V_m, e_mt)
    q_mt_dot = np.cross(e_mt, V_t - V_m) / r_mt

    r_md_dot = np.dot(V_d - V_m, e_md) # negative
    q_md_dot = np.cross(e_md, V_d - V_m) / r_md

    # 旧的状态量
    x_11 = q_md - q_md_expect
    x_12 = q_md_dot

    # 中间参数
    M1 = N_m * np.dot(e_md, e_vm) / (5 * np.dot(e_mt, e_vm))
    P1 = N_m * np.dot(e_md, e_vm) * r_mt_dot * q_mt_dot / (np.dot(e_mt, e_vm) * r_md)
    L1 = 1 / c1 + M1 * M1 / d1
    A1 = L1 * a1 / (8 * r_md_dot * r_md_dot) * (1 - 2 * np.e * np.e + np.power(np.e, 4))
    B1 = L1 * b1 / (4 * r_md * r_md_dot) * (1 - np.power(np.e, 4)) + 1
    C1 = np.e * np.e * x_12 - r_md * P1 * (1 - np.e * np.e) / (2 * r_md_dot)
    D1 = L1 * a1 * r_md / (16 * np.power(r_md_dot, 3)) * (4 + np.power(np.e, -2) - np.e * np.e) - 1
    E1 = r_md / (2 * r_md_dot) * (np.power(np.e, -2) - 1) + L1 * b1 / (8 * r_md_dot * r_md_dot) * (np.e * np.e + np.power(np.e, -2) - 2)
    F1 = - x_11 - r_md * r_md * P1 * (1 + np.power(np.e, -2)) / (4 * r_md_dot * r_md_dot)

    # 新的状态量
    x_11_tf = (B1 * F1 - C1 * E1) / (B1 * D1 - A1 * E1)
    x_12_tf = (C1 * D1 - A1 * F1) / (B1 * D1 - A1 * E1)

    # 协态量
    # lamda_11 = a1 * x_11_tf
    lamda_12 = a1 * x_11_tf * r_md / (2 * r_md_dot) * (1 - np.e * np.e) + b1 * x_12_tf * np.e * np.e

    # 加速度
    w_q = lamda_12 / (c1 * r_md)
    e_wq = - np.array([- e_md[1], e_md[0]])
    if np.linalg.norm(V_d) == 0:
        a_d = np.array([0, 0])
    else:
        e_ad = np.array([- e_vd[1], e_vd[0]])
        if GetAcuteAngle(e_ad, e_wq) > np.pi / 2:
            e_ad = np.array([e_ad[1], - e_ad[0]])
        a_d_value = np.clip(w_q / np.abs(np.dot(-e_md, e_vd)), - defender.max_accel, defender.max_accel)
        a_d = np.multiply(e_ad, a_d_value)

    # a_d = np.array([0, 0])

    return a_d

def attacker_policy(target, attacker):
    global N_m
    # target = world.targets[attacker.fake_target]

    if target.done or attacker.done:
        return np.array([0, 0])

    # TAD基本关系
    '''
    V_m = attacker.state.V
    V_t = target.state.V
    theta_m = attacker.state.phi
    theta_t = target.state.phi
    x_mt = attacker.state.p_pos - target.state.p_pos
    r_mt = np.linalg.norm(x_mt)
    q_mt = np.arctan2(x_mt[1], x_mt[0])  # -pi~pi
    q_mt = q_mt + np.pi * 2 if q_mt < 0 else q_mt  # 0~2pi

    # TAD运动关系(只用到了AT)
    q_mt_dot = (V_t * np.sin(q_mt - theta_t) - V_m * np.sin(q_mt - theta_m)) / r_mt
    r_mt_dot = V_t * np.cos(q_mt - theta_t) - V_m * np.cos(q_mt - theta_m)  # nagetive
    # print(r_mt_dot)
    '''
    # 要有向量的思想，起点和终点
    V_m = attacker.state.p_vel
    V_t = target.state.p_vel
    x_mt = target.state.p_pos - attacker.state.p_pos
    r_mt = np.linalg.norm(x_mt)
    e_mt = x_mt / r_mt
    r_mt_dot = np.dot(V_t - V_m, e_mt)
    q_mt_dot = np.cross(e_mt, V_t - V_m) / r_mt

    # attacker的加速度
    u_q = - N_m * r_mt_dot * q_mt_dot # PNG
    # u_q = - N_m * r_mt_dot * q_mt_dot - 0.2 * N_m * target.state.controller # APNG
    '''
    APNG效果不好，target的动作会导致attacker的动作不稳定
    '''
    e_uq = np.array([- e_mt[1], e_mt[0]])  # 单位方向向量
    if np.linalg.norm(V_m) == 0:
        a_m = np.array([0, 0])
    else:
        e_v = V_m / np.linalg.norm(V_m)
        e_am = np.array([- e_v[1], e_v[0]])  # v方向逆转90°，单位方向向量
        if GetAcuteAngle(e_am, e_uq) > np.pi / 2:
            e_am = np.array([e_v[1], - e_v[0]])
        a_m_value = np.clip(u_q / np.abs(np.dot(e_uq, e_am)), - attacker.max_accel, attacker.max_accel)
        a_m = np.multiply(e_am, a_m_value)

    return a_m

