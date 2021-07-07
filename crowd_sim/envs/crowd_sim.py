import logging
import copy
import gym
import math
import matplotlib as mpl
import matplotlib.lines as mlines
import configparser
import matplotlib.pylab as plt
import pyglet

import numpy as np
import rvo2
from matplotlib import patches
from numpy.linalg import norm
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.utils.robot import SARobot
from crowd_sim.envs.utils.info import Timeout, Danger, ReachGoal, Collision,\
    Nothing, Violation, Annoying
from crowd_sim.envs.utils.utils import point_to_segment_dist, rotate_coordinate


class CrowdSimConfig():
    def __init__(self, file):
        config = configparser.RawConfigParser()
        config.read(file)

        self.time_limit = config.getint('env', 'time_limit')
        self.time_step = config.getfloat('env', 'time_step')
        self.randomize_attributes = config.getboolean('env', 'randomize_attributes')
        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': np.iinfo(np.uint32).max - 2000, 'val': config.getint('env', 'val_size'),
                          'test': config.getint('env', 'test_size')}

        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')
        self.violation_penalty = config.getfloat('reward', 'violation_penalty')
        self.annoying_penalty = config.getfloat('reward', 'annoying_penalty')
        self.use_annoying = config.getboolean('reward', 'use_annoying')

        self.square_width = config.getfloat('sim', 'square_width')
        self.circle_radius = config.getfloat('sim', 'circle_radius')
        self.human_num = config.getint('sim', 'human_num')
        self.train_val_sim = config.get('sim', 'train_val_sim')
        self.test_sim = config.get('sim', 'test_sim')

        self.human_initial_px = config.getfloat('humans', 'initial_px')
        self.human_initial_py = config.getfloat('humans', 'initial_py')


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.humans = None
        self.global_time = None
        self.human_times = None
        # reward function
        self.success_reward = None
        self.collision_penalty = None
        self.discomfort_dist = None
        self.discomfort_penalty_factor = None
        # simulation configuration
        self.case_capacity = None
        self.case_size = None
        self.randomize_attributes = None
        self.train_val_sim = None
        self.test_sim = None
        self.square_width = None
        self.circle_radius = None
        self.human_num = None
        # for visualization
        self.state = None
        self.action_values = None
        self.attention_weights = None
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        self.renderer = MatplotRenderer()
        self.rs = np.random.RandomState(None)

    def configure(self, file):
        """
        config is a parser in this case
        """
        config = CrowdSimConfig(file)
        self.time_limit = config.time_limit
        self.time_step = config.time_step

        self.randomize_attributes = config.randomize_attributes
        self.success_reward = config.success_reward
        self.collision_penalty = config.collision_penalty
        self.discomfort_dist = config.discomfort_dist
        self.discomfort_penalty_factor = config.discomfort_penalty_factor
        self.violation_penalty = config.violation_penalty
        self.annoying_penalty = config.annoying_penalty
        self.use_annoying = config.use_annoying

        self.square_width = config.square_width
        self.circle_radius = config.circle_radius
        self.human_num = config.human_num

        self.case_capacity = config.case_capacity
        self.case_size = config.case_size

        self.train_val_sim = config.train_val_sim
        self.test_sim = config.test_sim

        self.human_initial_px = config.human_initial_px
        self.human_initial_py = config.human_initial_py

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")

        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_robot(self, robot):
        self.robot = robot

    def set_humans(self, humans):
        self.humans = humans

    def set_interactive_human(self):
        self.renderer.connect(self.on_click)
        self.humans[0].set(-4, 0, 4, 0, 0, 0, 0)

    def generate_static_human(self, i):
        if i == self.human_num - 1:
            self.humans[i].set(0, 0, 0, 0, 0, 0, 0, radius=.3, v_pref=0.)
        else:
            self.generate_circle_crossing_human(i)

    def generate_fixed_human(self, i):
        # v.1
        px = self.human_initial_px
        py = self.human_initial_py

        self.humans[i].set(
            px, py,
            -px, -py,
            0, 0, 0
        )

    def generate_fixed_opposite_human(self, i):
        px = self.robot.px - math.cos(math.radians(5))
        py = self.robot.py - math.sin(math.radians(5))
        self.humans[i].set(
            -px, -py,
            px, py,
            0, 0, 0
        )

    def generate_random_human_position(self, human_num, rule):
        """
        Generate human position according to certain rule
        Rule square_crossing: generate start/goal position at two sides of y-axis
        Rule circle_crossing: generate start position on a circle, goal position is at the opposite side

        :param human_num:
        :param rule:
        :return:
        """
        # initial min separation distance to avoid danger penalty at beginning
        if rule == 'square_crossing':
            for i in range(human_num):
                self.generate_square_crossing_human(i)
        elif rule == 'circle_crossing':
            for i in range(human_num):
                self.generate_circle_crossing_human(i)
        elif rule == 'static':
            for i in range(human_num):
                self.generate_static_human(i)
        elif rule == 'fixed':
            assert human_num == 1
            self.generate_fixed_human(0)
        elif rule == 'fixed_opposite':
            assert human_num == 1
            self.generate_fixed_opposite_human(0)
        elif rule == 'mixed':
            static = True if self.rs.random() < 0.2 else False
            if static:
                # randomly initialize static objects in a square of (width, height)
                width = 4
                height = 8
                for i in range(self.human_num):
                    if self.rs.random() > 0.5:
                        sign = -1
                    else:
                        sign = 1
                    while True:
                        px = self.rs.random() * width * 0.5 * sign
                        py = (self.rs.random() - 0.5) * height
                        collide = False
                        for agent in [self.robot] + self.humans[:i]:
                            if norm((px - agent.px, py - agent.py)) < self.humans[i].radius + agent.radius + self.discomfort_dist:
                                collide = True
                                break
                        if not collide:
                            break
                    self.humans[i].set(px, py, px, py, 0, 0, 0)
            else:
                # the first 2 two humans will be in the circle crossing scenarios
                # the rest humans will have a random starting and end position
                for i in range(self.human_num):
                    if i < 2:
                        self.generate_circle_crossing_human(i)
                    else:
                        self.generate_square_crossing_human(i)
        else:
            raise ValueError("Rule doesn't exist")

    def generate_circle_crossing_human(self, i):
        if self.randomize_attributes:
            self.humans[i].sample_random_attributes()
        while True:
            angle = self.rs.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (self.rs.random() - 0.5) * self.humans[i].v_pref
            py_noise = (self.rs.random() - 0.5) * self.humans[i].v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans[:i]:
                min_dist = self.humans[i].radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        self.humans[i].set(px, py, -px, -py, 0, 0, 0)

    def generate_square_crossing_human(self, i):
        if self.randomize_attributes:
            self.humans[i].sample_random_attributes()
        if self.rs.random() > 0.5:
            sign = -1
        else:
            sign = 1
        while True:
            px = self.rs.random() * self.square_width * 0.5 * sign
            py = (self.rs.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans[:i]:
                if norm((px - agent.px, py - agent.py)) < self.humans[i].radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        while True:
            gx = self.rs.random() * self.square_width * 0.5 * -sign
            gy = (self.rs.random() - 0.5) * self.square_width
            collide = False
            for agent in [self.robot] + self.humans[:i]:
                if norm((gx - agent.gx, gy - agent.gy)) < self.humans[i].radius + agent.radius + self.discomfort_dist:
                    collide = True
                    break
            if not collide:
                break
        self.humans[i].set(px, py, gx, gy, 0, 0, 0)

    def get_human_times(self):
        """
        Run the whole simulation to the end and compute the average time for human to reach goal.
        Once an agent reaches the goal, it stops moving and becomes an obstacle
        (doesn't need to take half responsibility to avoid collision).

        :return:
        """
        # centralized orca simulator for all humans
        if not self.robot.reached_destination():
            raise ValueError('Episode is not done yet')
        params = (10, 10, 5, 5)
        sim = rvo2.PyRVOSimulator(self.time_step, *params, 0.3, 1)
        sim.addAgent(self.robot.get_position(), *params, self.robot.radius, self.robot.v_pref,
                     self.robot.get_velocity())
        for human in self.humans:
            sim.addAgent(human.get_position(), *params, human.radius, human.v_pref, human.get_velocity())

        max_time = 1000
        while not all(self.human_times):
            for i, agent in enumerate([self.robot] + self.humans):
                vel_pref = np.array(agent.get_goal_position()) - np.array(agent.get_position())
                if norm(vel_pref) > 1:
                    vel_pref /= norm(vel_pref)
                sim.setAgentPrefVelocity(i, tuple(vel_pref))
            sim.doStep()
            self.global_time += self.time_step
            if self.global_time > max_time:
                logging.warning('Simulation cannot terminate!')
            for i, human in enumerate(self.humans):
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time

            # for visualization
            self.robot.set_position(sim.getAgentPosition(0))
            for i, human in enumerate(self.humans):
                human.set_position(sim.getAgentPosition(i + 1))
            self.state = [self.robot.get_full_state(), [human.get_full_state() for human in self.humans]]

        del sim
        return self.human_times

    def reset(self, phase='test', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        if phase == 'test':
            self.human_times = [0] * self.human_num
        else:
            self.human_times = [0] * (self.human_num if self.robot.policy.multiagent_training else 1)
        if not self.robot.policy.multiagent_training and 'fixed' not in self.train_val_sim:
            self.train_val_sim = 'circle_crossing'
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}
        self.robot.set(0, -self.circle_radius, 0, self.circle_radius, 0, 0, np.pi / 2)
        if self.case_counter[phase] >= 0:
            self.rs = np.random.RandomState(
                counter_offset[phase] + self.case_counter[phase]
            )

            if phase in ['train', 'val']:
                human_num = self.human_num if self.robot.policy.multiagent_training else 1
                self.humans = self.humans[:human_num]
                self.generate_random_human_position(human_num=human_num, rule=self.train_val_sim)
            else:
                if self.human_num != len(self.humans):
                    # human_numで初めは初期化される。Train時に#human_num=1になり、また戻すので下記の処理
                    human = self.humans[0]
                    self.humans = [
                        copy.copy(human) for _ in range(self.human_num)
                    ]
                self.generate_random_human_position(
                    human_num=self.human_num, rule=self.test_sim
                )
            # case_counter is always between 0 and case_size[phase]
            self.case_counter[phase] = \
                (self.case_counter[phase] + 1) % self.case_size[phase]
        else:
            assert phase == 'test'
            if self.case_counter[phase] == -1:
                # for debugging purposes
                self.human_num = 3
                # self.humans = [
                #   Human(self.config, 'humans') for _ in range(self.human_num)
                # ]
                self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)
            else:
                raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        self.state = list()
        if hasattr(self.robot.policy, 'action_values'):
            self.action_values = list()
        if hasattr(self.robot.policy, 'get_attention_weights'):
            self.attention_weights = list()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob = [human.get_observable_state() for human in self.humans]
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def onestep_lookahead(self, action):
        return self.step(action, update=False)

    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)

        """
        human_actions = []
        for human in self.humans:
            # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
            human_actions.append(human.act(ob))

        reward, done, info = self._generate_reward(action)

        if update:
            # store state, action value and attention weights
            self.state = [
                self.robot.get_full_state(),
                [human.get_full_state() for human in self.humans]
            ]
            if hasattr(self.robot.policy, 'action_values'):
                self.action_values.append(self.robot.policy.action_values)
            if hasattr(self.robot.policy, 'get_attention_weights'):
                self.attention_weights.append(
                    self.robot.policy.get_attention_weights()
                )

            # update all agents
            self.robot.step(action)
            for i, human_action in enumerate(human_actions):
                self.humans[i].step(human_action)
            self.global_time += self.time_step
            for i, human in enumerate(self.humans):
                # only record the first time the human reaches the goal
                if self.human_times[i] == 0 and human.reached_destination():
                    self.human_times[i] = self.global_time
            # compute the observation
            ob = [human.get_observable_state() for human in self.humans]

        else:
            if self.robot.sensor == 'coordinates':
                ob = [
                    human.get_next_observable_state(action)
                    for human, action in zip(self.humans, human_actions)
                ]
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        return ob, reward, done, info

    def render(self, mode='human'):
        humans = self.humans
        robots = [self.robot]
        self.renderer.add_humans(humans)
        self.renderer.add_robots(robots)
        self.renderer.render()

        if mode == 'human':
            pass
        else:
            raise NotImplementedError

    def on_click(self, event):
        # self.humans[0].policy.key_release(event.key)
        self.humans[0].policy.key_press(event.key)

    def _generate_reward(self, action):
        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            elif self.robot.kinematics == 'unicycle' and action.r != 0:
                vx = (action.v / action.r) * (
                        np.sin(
                            action.r * self.time_step + self.robot.theta
                        ) - np.sin(self.robot.theta))
                vy = (action.v / action.r) * (
                        np.cos(
                            action.r * self.time_step + self.robot.theta
                        ) - np.cos(self.robot.theta))
            else:
                vx = human.vx - action.v * np.cos(action.r + self.robot.theta)
                vy = human.vy - action.v * np.sin(action.r + self.robot.theta)
            ex = px + vx * self.time_step
            ey = py + vy * self.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                # logging.debug("Collision: distance between robot and p{} is {:.2E}".format(i, closest_dist))
                break
            elif closest_dist < dmin:
                dmin = closest_dist

        # collision detection between humans
        human_num = len(self.humans)
        for i in range(human_num):
            for j in range(i + 1, human_num):
                dx = self.humans[i].px - self.humans[j].px
                dy = self.humans[i].py - self.humans[j].py
                dist = (dx ** 2 + dy ** 2) ** (1 / 2) - self.humans[i].radius - self.humans[j].radius
                if dist < 0:
                    # detect collision but don't take humans' collision into account
                    logging.debug('Collision happens between humans in step()')

        # check if violating the social norm
        if type(self.robot) == SARobot:
            is_violate_snorm = self._is_in_social_norm()
        else:
            is_violate_snorm = False
        
        # check if annoying the human
        is_annoying = self._is_annoying() and self.use_annoying

        #  check if reaching the goal
        end_position = np.array(
            self.robot.compute_position(action, self.time_step)
        )
        reaching_goal = norm(
            end_position - np.array(self.robot.get_goal_position())
        ) < self.robot.radius

        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            # unit_v = (self.robot.vx**2 + self.robot.vy**2)**.5
            # 1はmax_v? pref_v, 1-unit_distは最大値と現在値の差分。補正項かな。
            # reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step * (1 - unit_v)
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor
            # self.discomfort_dist = 0.2, self.discomfort_penalty_factor=0.5で元論文と一致
            done = False
            info = Danger(dmin)
        elif is_violate_snorm:
            reward = self.violation_penalty
            done = False
            info = Violation()
        elif is_annoying:
            assert self.use_annoying is True
            reward = self.annoying_penalty
            done = False
            info = Annoying()
        else:
            reward = 0
            done = False
            info = Nothing()
        return reward, done, info

    def _transform_robot_coordinate(self, x, y):
        px, py = self.robot.px, self.robot.py
        theta_rot = self._get_robot_rotation_theta()
        rel_x, rel_y = x - px, y - py
        rot_x, rot_y = rotate_coordinate(rel_x, rel_y, theta_rot)
        return rot_x, rot_y

    def _transform_robot_angle(self, theta):
        theta_rot = self._get_robot_rotation_theta()
        return theta - theta_rot

    def _transform_robot_velocity(self, vx, vy):
        theta_rot = self._get_robot_rotation_theta()
        rot_vx, rot_vy = rotate_coordinate(vx, vy, theta_rot)
        return rot_vx, rot_vy

    def _get_robot_rotation_theta(self):
        dgx = self.robot.gx - self.robot.px
        dgy = self.robot.gy - self.robot.py
        return np.arctan2(dgy, dgx)

    def _transform_human_local_coordinate(self, human, x, y):
        """人の進行方向をX軸とする座標に変換する。原点は人の位置。

        Args:
            human ([type]): [description]
            x ([type]): [description]
            y ([type]): [description]

        Returns:
            [type]: [description]
        """
        human_vx, human_vy = human.vx, human.vy
        theta_rot = np.arctan2(human_vy, human_vx)
        rel_x, rel_y = x-human.px, y-human.py
        rot_x, rot_y = rotate_coordinate(rel_x, rel_y, theta_rot)
        return rot_x, rot_y

    def _is_in_social_norm(self):
        # Social norm; right handed rule
        # Yu Fan Chen, Michael Everett, Miao Liu, and Jonathan P. How. 2017. 
        # Socially Aware Motion Planning with Deep Reinforcement Learning.
        # In 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).
        # https://doi.org/10.1109/IROS.2017.8202312.
        local_robot_vx, local_robot_vy = self._transform_robot_velocity(
            self.robot.vx, self.robot.vy
        )
        local_robot_px, local_robot_py = self._transform_robot_coordinate(
            self.robot.px, self.robot.py
        )
        local_robot_gx, local_robot_gy = self._transform_robot_coordinate(
            self.robot.gx, self.robot.gy
        )
        dg = ((local_robot_gx - local_robot_px)**2 + (local_robot_gy - local_robot_py)**2)**.5
        v = (local_robot_vx**2 + local_robot_vy**2)**.5
        b_penalties = []
        for human in self.humans:
            local_human_vx, local_human_vy = self._transform_robot_velocity(
                human.vx, human.vy
            )
            local_human_px, local_human_py = self._transform_robot_coordinate(
                human.px, human.py
            )
            local_robot_theta = self._transform_robot_angle(self.robot.theta)
            phi = np.arctan2(local_human_vy, local_human_vx)
            diff_phi = phi - local_robot_theta
            is_s_pass = all([
                dg > 3, 1 < local_human_px and local_human_px < 4,
                abs(diff_phi) > 3/4*np.pi
            ])
            tilde_v = (local_human_vx**2 + local_human_vy**2)**.5
            is_s_outk = all([
                dg > 3, 0 < local_human_px and local_human_px < 3,
                v > tilde_v, 0 < local_human_py and local_human_py < 1,
                abs(diff_phi) < np.pi/4
            ])
            # 左側からhumanが来てcrossingするケースにペナルティ。
            phi_rot = np.arctan(
                (local_human_vy - local_robot_vy) / (local_human_vx - local_robot_vx)
            )
            da = (
                (local_robot_px - local_human_px)**2 + (local_robot_py - local_human_py)**2
                )**.5
            is_s_cross = all([
                dg > 3, da < 2, phi_rot > 0,
                -3/4*np.pi < diff_phi and diff_phi < -np.pi/4
            ])
            # print(dg, da, phi_rot, diff_phi)
            # print(is_s_cross)
            b_penalties.append(any([is_s_pass, is_s_outk, is_s_cross]))
        return any(b_penalties)

    def _is_annoying(self):
        b_annoyings = []
        for human in self.humans:
            # 1. transform human centric local coordination
            robot_px, robot_py = self._transform_human_local_coordinate(
                human, self.robot.px, self.robot.py
            )
            # 2. check if a robot is in the area where the human go to.
            b_annoying = (0 < robot_px) and (robot_px < 3)
            b_annoying &= (-1 < robot_py) and (robot_py < 1)
            b_annoyings.append(b_annoying)
        return any(b_annoyings)

    def robot_forth_human(self):
        """ robotがhumanの前を通過しているか

        """
        robot = self.robot
        humans = self.humans
        angle = 3/180*np.pi
        is_forths = []
        for human in humans:
            rel_x = robot.px - human.px
            rel_y = robot.py - human.py
            p_angle = np.arctan2(rel_y, rel_x)
            v_angle = np.arctan2(human.vy, human.vx)
            b_angle = (p_angle - angle) < v_angle
            b_angle &= (p_angle + angle) > v_angle
            is_forths.append(b_angle)
        return any(is_forths)


class Renderer(object):
    def __init__(self):
        pass

    def add_humans(self, humans):
        pass

    def add_robots(self, robots):
        pass

    def render(self):
        pass


class MatplotRenderer(Renderer):
    def __init__(self):
        self.create_fig()

    def add_humans(self, humans):
        self.reset_axis()
        for i, human in enumerate(humans):
            human_circle = plt.Circle(
                human.get_position(), human.radius, fill=False,
                color=self.cmap(i)
            )
            self.ax.add_artist(human_circle)
            goal = mlines.Line2D(
                [human.get_goal_position()[0]],
                [human.get_goal_position()[1]],
                color=self.cmap(i), marker='*',
                linestyle='None', markersize=15,
                label='Goal', fillstyle='none'
            )
            self.ax.add_artist(goal)

    def add_robots(self, robots):
        for i, robot in enumerate(robots):
            robot_circle = plt.Circle(
                robot.get_position(),
                robot.radius, fill=True, color='k', fc='orange')
            self.ax.add_artist(robot_circle)
            goal = mlines.Line2D(
                [robot.get_goal_position()[0]],
                [robot.get_goal_position()[1]],
                color='red', marker='*', linestyle='None',
                markersize=15, label='Goal'
            )
            self.ax.add_artist(goal)

    def render(self):
        plt.pause(.0001)

    def reset_axis(self):
        self.ax.clear()
        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-6, 6)

    def create_fig(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        self.fig = fig
        self.ax = ax
        self.cmap = plt.cm.get_cmap('hsv', 5)

    def connect(self, func):
        self.fig.canvas.mpl_connect('key_press_event', func)
