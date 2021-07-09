import unittest
import configparser
import gym
import torch
import logging
import numpy as np
from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.policy_factory import policy_factory as nav_policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.visualization.observer_subscriber import notify
from crowd_sim.envs.visualization.plotter import Plotter
from crowd_sim.envs.visualization.video import Video


phase = "val"
env_config_path = "configs/env.config"
policy_config = configparser.ConfigParser()
policy_config.read("configs/policy.config")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_case = None


def create_env():
    # configure policy
    policy = nav_policy_factory['cadrl']()
    policy.configure(policy_config)
    # if policy.trainable:
    #     if args.model_dir is None:
    #         parser.error('Trainable policy must be specified with a model weights directory')
    #     policy.get_model().load_state_dict(torch.load(model_weights))

    # configure environment
    env = gym.make('CrowdSim-v0')
    env.configure(env_config_path)

    robot = Robot()
    robot.configure(env_config_path, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    env.human_num = 1
    humans = [Human() for _ in range(env.human_num)]
    for i, human in enumerate(humans):
        human.configure(env_config_path, 'humans')
    orca = policy_factory['orca']()
    humans[0].set_policy(orca)
    env.set_humans(humans)
    env.test_sim = "fixed"
    # if args.square:
    #     env.test_sim = 'square_crossing'
    # if args.circle:
    #     env.test_sim = 'circle_crossing'

    policy.set_phase(phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()
    return env, robot, humans

    # ob = env.reset(phase, test_case)
    # env.set_interactive_human()
    # last_pos = np.array(robot.get_position())
    # observation_subscribers = []


class CheckTest(unittest.TestCase):
    def setUp(self):
        self.env, self.robot, self.humans = create_env()
    
    def test_is_annoying(self):
        robot_gx, robot_gy = 0, 4
        human_gx, human_gy = 4, 0
        _ = self.env.reset(phase, test_case)
        # robotがhumanの前にいる場合
        self.robot.set(
            0, 0, robot_gx, robot_gy, 0, 1, np.pi/2
        )
        self.humans[0].set(
            -2, 0, human_gx, human_gy, 1, 0, 0
        )
        is_annoying = self.env._is_annoying()
        self.assertTrue(is_annoying)
        # robotがhumanの後ろにいる場合
        self.robot.set(
            -2, 0, robot_gx, robot_gy, 0, 1, np.pi/2
        )
        self.humans[0].set(
            0.5, 0, human_gx, human_gy, 1, 0, 0
        )
        is_annoying = self.env._is_annoying()
        self.assertFalse(is_annoying)
        # robotがhumanの前にいるが、x軸方向に範囲外の場合
        self.robot.set(
            4, 0, robot_gx, robot_gy, 0, 1, np.pi/2
        )
        self.humans[0].set(
            -2, 0, human_gx, human_gy, 1, 0, 0
        )
        is_annoying = self.env._is_annoying()
        self.assertFalse(is_annoying)
        # robotがhumanの前にいるが、y軸方向に範囲外の場合
        self.robot.set(
            0, 2, robot_gx, robot_gy, 0, 1, np.pi/2
        )
        self.humans[0].set(
            -2, 0, human_gx, human_gy, 1, 0, 0
        )
        is_annoying = self.env._is_annoying()
        self.assertFalse(is_annoying)

    def test_is_in_s_pass(self):
        robot_gx, robot_gy = 4, 0
        human_gx, human_gy = -4, 0
        _ = self.env.reset(phase, test_case)
        self.robot.set(
            -1, 1, robot_gx, robot_gy, 1, 0, 0
        )
        self.humans[0].set(
            1, -1, human_gx, human_gy, -1, 0, np.pi
        )
        is_violate = self.env._is_in_social_norm()
        self.assertTrue(is_violate)
        self.robot.set(
            0, 0.5, robot_gx, robot_gy, 1, 0, 0
        )
        self.humans[0].set(
            0.5, -1, human_gx, human_gy, -1, 0, np.pi
        )
        is_violate = self.env._is_in_social_norm()
        self.assertFalse(is_violate)

    def test_is_in_s_outk(self):
        robot_gx, robot_gy = 4, 0
        human_gx, human_gy = -4, 0
        _ = self.env.reset(phase, test_case)
        self.robot.set(
            0, 0, robot_gx, robot_gy, 2, 0, 0
        )
        self.humans[0].set(
            1, 0.8, human_gx, human_gy, 1, 0, 0
        )
        is_violate = self.env._is_in_social_norm()
        self.assertTrue(is_violate)
        self.robot.set(
            0, 0, robot_gx, robot_gy, 2, 0, 0
        )
        self.humans[0].set(
            1, 2, human_gx, human_gy, 1, 0, 0
        )
        is_violate = self.env._is_in_social_norm()
        self.assertFalse(is_violate)

    def test_is_in_s_cross(self):
        robot_gx, robot_gy = 4, 0
        human_gx, human_gy = 0, -4
        _ = self.env.reset(phase, test_case)
        # robotが左から右、humanが上から下
        self.robot.set(
            0, 0, robot_gx, robot_gy, 2, 0, 0
        )
        self.humans[0].set(
            1, 0, human_gx, human_gy, 0, -1, -np.pi/2
        )
        is_violate = self.env._is_in_social_norm()
        self.assertTrue(is_violate)
        # robotが左から右、humanが上から下
        self.robot.set(
            0, 0, robot_gx, robot_gy, 2, 0, 0
        )
        self.humans[0].set(
            3, 0, human_gx, human_gy, 0, -1, -np.pi/2
        )
        is_violate = self.env._is_in_social_norm()
        self.assertFalse(is_violate)

        robot_gx, robot_gy = 4, 0
        human_gx, human_gy = 0, 4
        # robotが左から右、humanが下から上
        self.robot.set(
            0, 0, robot_gx, robot_gy, 2, 0, 0
        )
        self.humans[0].set(
            1, 0, human_gx, human_gy, 0, 1, np.pi/2
        )
        is_violate = self.env._is_in_social_norm()
        self.assertFalse(is_violate)

        robot_gx, robot_gy = 0, 4
        human_gx, human_gy = 4, 0
        # robotが下から上、humanが左から右
        self.robot.set(
            0, -1, robot_gx, robot_gy, 0, 2, np.pi/2
        )
        self.humans[0].set(
            -1, 0, human_gx, human_gy, 1, 0, -np.pi
        )
        is_violate = self.env._is_in_social_norm()
        self.assertTrue(is_violate)
        # robotが下から上、humanが右から左
        self.robot.set(
            0, 0, robot_gx, robot_gy, 0, 2, np.pi/2
        )
        self.humans[0].set(
            1, 0, human_gx, human_gy, -1, 0, -np.pi
        )
        is_violate = self.env._is_in_social_norm()
        self.assertFalse(is_violate)


class TransformationTest(unittest.TestCase):
    def setUp(self):
        self.env, self.robot, self.humans = create_env()

    def test_transform_robot_velocity(self):
        px = -3.598145756365556
        py = -1.1049101069911202
        self.robot.set(
            0, -4, 0, 4, 0, 0, np.pi/2
        )
        self.humans[0].set(
            px, py, -px, -py, 0, 0, 0
        )
        _ = self.env.reset(phase, test_case)
        x, y = self.env._transform_robot_velocity(0, 1)
        self.assertAlmostEqual(x, 1)
        self.assertAlmostEqual(y, 0)

    def test_transform_parallel(self):
        # humanとrobotの初期位置を決定する。
        px = -3.598145756365556
        py = -1.1049101069911202
        _ = self.env.reset(phase, test_case)
        self.robot.set(
            0, -4, 0, 4, 0, 0, np.pi/2
        )
        self.humans[0].set(
            px, py, -px, -py, 0, 0, 0
        )
        x, y = self.env._transform_robot_coordinate(0, 0)
        self.assertAlmostEqual(x, 4)
        self.assertAlmostEqual(y, 0)
        x, y = self.env._transform_robot_coordinate(4, 0)
        self.assertAlmostEqual(x, 4)
        self.assertAlmostEqual(y, -4)
        x, y = self.env._transform_robot_coordinate(0, 4)
        self.assertAlmostEqual(x, 8)
        self.assertAlmostEqual(y, 0)
        x, y = self.env._transform_robot_coordinate(-4, 0)
        self.assertAlmostEqual(x, 4)
        self.assertAlmostEqual(y, 4)

    def test_transform_robot_angle(self):
        # humanとrobotの初期位置を決定する。
        px = -3.598145756365556
        py = -1.1049101069911202
        _ = self.env.reset(phase, test_case)
        self.robot.set(
            0, -4, 0, 4, 0, 0, np.pi/2
        )
        self.humans[0].set(
            px, py, -px, -py, 0, 0, 0
        )
        theta = self.env._transform_robot_angle(np.pi/2)
        self.assertEqual(theta, 0)

    def test_transform_human_local_coordinate(self):
        gx = 1
        gy = 3
        _ = self.env.reset(phase, test_case)
        self.robot.set(
            0, -4, 0, 4, 0, 0, np.pi/2
        )
        self.humans[0].set(
            1, 0, gx, gy, 0, 1, np.pi/2
        )
        x, y = self.env._transform_human_local_coordinate(
            self.humans[0], self.robot.px, self.robot.py
        )
        self.assertAlmostEqual(x, -4)
        self.assertAlmostEqual(y, 1)


if __name__ == "__main__":
    unittest.main()
