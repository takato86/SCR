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


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.env, self.robot, self.humans = create_env()
        return super().setUp()

    def test_transform_velocity(self):
        _ = self.env.reset(phase, test_case)
        self.robot.set(
            1, 0, 0, 3, 0, 1, -np.pi/2
        )
        self.humans[0].set(
            0, 0, 3, 0, 1, 0, 0
        )
        b_forth = self.env.robot_forth_human()
        self.assertTrue(b_forth)

        self.robot.set(
            0, 1, 0, 3, 0, 1, -np.pi/2
        )
        self.humans[0].set(
            0, 0, 3, 0, 1, 0, 0
        )
        b_forth = self.env.robot_forth_human()
        self.assertFalse(b_forth)


if __name__ == "__main__":
    unittest.main()
