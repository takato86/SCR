import numpy as np
from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import ObservableState, JointState


class Human(Agent):
    def __init__(self):
        super().__init__()

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

# TODO 修正 for SACADRL
class SAHuman(Agent):
    def __init__(self):
        super().__init__()

    def get_observable_state(self):
        phi = np.arctan2(self.vy, self.vx)
        return ObservableState(
            self.px, self.py, self.vx, self.vy, self.radius, phi
        )

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
