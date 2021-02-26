from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from shaner import SarsaRS, NaiveSRS
import logging


class Robot(Agent):
    def __init__(self):
        super().__init__()

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action


class ShapingRobot(Robot):
    def __init__(self, method, **params):
        super().__init__()
        if method == 'dta':
            self.reward_shaping = SarsaRS(**params)
        elif method == 'nrs':
            self.reward_shaping = NaiveSRS(**params)
        else:
            raise Exception(f"{method} is NOT included.")

    def start(self, ob):
        state = JointState(self.get_full_state(), ob)
        self.pre_state = state
        self.reward_shaping.start(state)

    def shape(self, ob, reward, done, info):
        state = JointState(self.get_full_state(), ob)
        sr = self.reward_shaping.perform(self.pre_state, state, reward, done, info)
        return sr

    def print4analysis(self):
        logging.info("The number of achievements for subgoals: {}".format(
            self.reward_shaping.get_counter_transit()
            )
        )

    def get_achieve_subgoals(self):
        return self.reward_shaping.get_counter_transit()
