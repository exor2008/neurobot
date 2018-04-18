from abc import ABCMeta, abstractmethod
import numpy as np
from primitives import Constructor
from gears import Renderable 

class BaseRobot(Constructor, Renderable):
    def __init__(self, physic):
        Constructor.__init__(self)
        self.physic = physic
        self.construct()
        self.save_state()

    @abstractmethod
    def act(self, actions):
        pass

    def save_state(self):
        self.saves = [part.pos for part in self.parts]

    def reset(self):
        for val, part in zip(self.saves, self.parts):
            part.pos = val
    

class AmoebaRobot(BaseRobot):
    def construct(self):
        self.body = self.add_box(size=(2, 1, 1), 
            pos=(0, 0, 1), mass=2, friction=1, color='blue')
        left_wing = self.add_box(size=(0.7, 0.3, 0.1), 
            pos=(1.5, -1.5, 1), mass=0.5, friction=5, color='yellow')
        right_wing = self.add_box(size=(0.7, 0.3, 0.1), 
            pos=(1.5, 1.5, 1), mass=0.5, friction=5, color='yellow')

        left_engine = self.add_hinge_joint(self.body, left_wing, (.8, 0, 0), (0, 0, 0), axis=(False, True, False))
        right_engine = self.add_hinge_joint(self.body, right_wing, (.8, 0, 0), (0, 0, 0), axis=(False, True, False))

    def act(self, actions):
        actions = np.asarray(actions) * 10
        [engine.set_impulse(impulse) for engine, impulse in zip(self._engines, actions)]

    def render(self, render):
        [rend_obj.render(render) for rend_obj in self._renderable]

__all__ = ['AmoebaRobot']