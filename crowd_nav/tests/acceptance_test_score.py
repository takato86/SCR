import logging
import argparse
import configparser
import os
import torch
import numpy as np
import gym

from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.human import Human
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.visualization.observer_subscriber import notify
from crowd_sim.envs.visualization.plotter import Plotter
from crowd_sim.envs.visualization.video import Video


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='data/output/env.config')
    parser.add_argument('--policy_config', type=str, default='data/output/policy.config')
    parser.add_argument('--policy', type=str, default='scr')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--plot_file', type=str, default=None)
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        policy_config_file = args.env_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.ConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights))

    # configure environment
    env = gym.make('CrowdSim-v0')
    env.configure(env_config_file)

    robot = Robot()
    robot.configure(env_config_file, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    humans = [Human() for _ in range(env.human_num)]
    for human in humans:
        human.configure(env_config_file, 'humans')
    env.set_humans(humans)

    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'

    explorer = Explorer(env, robot, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()

    explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)


if __name__ == '__main__':
    main()
