from abc import ABCMeta, abstractmethod
from physic import *

class BaseRobot(metaclass=ABCMeta):
    def __init__(self, world):
        self.world = world
        self._renderable = []
        self._engines = []
        self.construct()

    @abstractmethod
    def construct(self):
        pass

    @abstractmethod
    def render(self, render):
        pass

    @abstractmethod
    def act(self, actions):
        pass


class RobotConstructor:
    def add_box(self, *args, **kwargs):
        box = PhysicBox(*args, **kwargs)
        self.world.attachRigidBody(box.node)
        self._renderable.append(box)
        return box

    def add_hinge_joint(self, *args, **kwargs):
        joint = HingeJoint(*args, **kwargs)
        self.world.attachConstraint(joint.constr)
        self._engines.append(joint)
        return joint
    

class AmoebaRobot(BaseRobot, RobotConstructor):
    def construct(self):
        self.body = self.add_box(size=(2, 1, 1), 
            pos=(0, 0, 1), mass=2, friction=1, color='blue')
        left_wing = self.add_box(size=(0.7, 0.3, 0.1), 
            pos=(1.5, -1.5, 1), mass=0.5, friction=5, color='yellow')
        right_wing = self.add_box(size=(0.7, 0.3, 0.1), 
            pos=(1.5, 1.5, 1), mass=0.5, friction=5, color='yellow')

        left_engine = self.add_hinge_joint(self.body, left_wing, (.8, 0, 0), (0, 0, 0), axis=(False, True, False))
        left_engine.set_impulse(-10)
        right_engine = self.add_hinge_joint(self.body, right_wing, (.8, 0, 0), (0, 0, 0), axis=(False, True, False))
        right_engine.set_impulse(-10)

    def act(actions):
        pass

    def render(self, render):
        [rend_obj.render(render) for rend_obj in self._renderable]

__all__ = ['BaseRobot', 'AmoebaRobot']