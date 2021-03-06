import numpy as np 


class State(object):
    pass


class FullState(State):
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.gx = gx
        self.gy = gy
        self.v_pref = v_pref
        self.theta = theta

        self.position = (self.px, self.py)
        self.goal_position = (self.gx, self.gy)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy,
                                          self.v_pref, self.theta]])

    def __eq__(self, other):
        return all([self.px == other.px, self.py == other.py, self.vx == other.vx, self.vy == other.vy, self.radius == other.radius])


class ObservableState(State):
    def __init__(self, px, py, vx, vy, radius):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (self.px, self.py, self.vx, self.vy, self.radius)

    def __str__(self):
        return ' '.join([str(x) for x in [self.px, self.py, self.vx, self.vy, self.radius]])

    def __eq__(self, other):
        return all([self.px == other.px, self.py == other.py, self.vx == other.vx, self.vy == other.vy, self.radius == other.radius])

    def is_available(self):
        return self.px is not None and self.py is not None and self.vx is not None and self.vy is not None and self.radius is not None


class SAObservableState(ObservableState):
    def __init__(self, px, py, vx, vy, radius, da, phi):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.da = da
        self.phi = phi

        self.position = (self.px, self.py)
        self.velocity = (self.vx, self.vy)

    def __add__(self, other):
        return other + (
            self.px, self.py, self.vx, self.vy, self.radius, self.da, self.phi
            )

    def __str__(self):
        return ' '.join(
            [
                str(x)
                for x in [
                    self.px, self.py, self.vx, self.vy, self.radius, self.da,
                    self.phi
                ]
            ]
        )

    def __eq__(self, other):
        return all(
            [
                self.px == other.px, self.py == other.py, self.vx == other.vx,
                self.vy == other.vy, self.radius == other.radius,
                self.da == other.da, self.phi == other.phi
            ]
        )

    def is_available(self):
        return all(
            [
                self.px is not None, self.py is not None, self.vx is not None,
                self.vy is not None, self.radius is not None,
                self.da is not None, self.phi is not None
            ]
        )


class JointState(object):
    def __init__(self, self_state, human_states):
        assert isinstance(self_state, FullState)
        for human_state in human_states:
            assert isinstance(human_state, ObservableState)

        self.self_state = self_state
        self.human_states = human_states
