from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from shaner import SarsaRS


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
    def __init__(self, **params):
        super().__init__()
        self.reward_shaping = SarsaRS(**params)

    def start(self, ob):
        state = JointState(self.get_full_state(), ob)
        self.pre_state = state

    def shape(self, ob, reward, done):
        state = JointState(self.get_full_state(), ob)
        sr = self.reward_shaping.perform(self.pre_state, state, reward, done)
        return sr
