from abc import ABCMeta, abstractmethod
import os
import sys
import logging

import panda3d.core as core
from panda3d.core import Vec3
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletHingeConstraint

from gears import RigidBody, Renderable
import res_loader


class Box(RigidBody, Renderable):
    def __init__(self, size=(1, 1, 1), pos=(0, 0, 0), hpr=(0, 0, 0), mass=0, friction=1, color='blue'):
        node = self.construct(size)
        super(Box, self).__init__(node, pos, hpr, mass, friction)
        self.color=color

    def construct(self, size):
        self.size = size
        node = BulletRigidBodyNode()
        shape = BulletBoxShape(Vec3(*size))
        node.addShape(shape)
        return node

    def render(self, render):
        logging.debug('box rendered')
        self.load_model(self.color)
        self.model.set_scale(self.size)
        self.np = render.attachNewNode(self.node)
        self.model.copyTo(self.np)

    def load_model(self, color):
        model_name = '{color}_box.egg'.format(color=color)
        path = res_loader.Loader().get_model_path(model_name)
        self.model = loader.loadModel(path)
        # self.model.flattenLight()


Y_AXIS = (False, True, False)
MAX_IMPULSE = 5


class BaseJoint(metaclass=ABCMeta):
    @abstractmethod
    def set_impulse(self, impulse):
        pass

    @abstractmethod
    def set_limit(self, *limits):
        pass

    @abstractmethod
    def set_axis(self, *axis):
        pass


class HingeJoint(BaseJoint):
    def __init__(self, obj1, obj2, point1, point2, axis=Y_AXIS):
        joint_point1 = core.TransformState.makePos(point1)
        joint_point2 = core.TransformState.makePos(point2)
        self.construct(obj1.node, obj2.node, joint_point1, joint_point2)
        self.set_axis(axis)
        # self.set_impulse(0)

    def set_impulse(self, impulse):
        self.constr.enableAngularMotor(True, impulse, MAX_IMPULSE)

    def construct(self, node1, node2, point1, point2):
        self.constr = BulletHingeConstraint(node1, node2, point1, point2)

    def set_limit(self, low, heigh):
        self.constr.setLimit(low, heigh)

    def set_axis(self, axis):
        self.constr.setAxis(axis)


class BaseLight(Renderable):
    pass


class PointLight(BaseLight):
    def __init__(self, color, pos):
        self.color = color
        self.pos = pos
        self.construct()

    def construct(self):
        self.light = core.PointLight('plight')
        self.light.setColor(self.color)

    def render(self, render):
        lightnp = render.attachNewNode(self.light)
        lightnp.setPos(self.pos)
        render.setLight(lightnp)


class SpotLight(BaseLight):
    def __init__(self, color, pos, target):
        self.color = color
        self.pos = pos
        self.target = target
        self.construct()

    def construct(self):
        self.light = core.Spotlight('slight')
        self.light.setColor(self.color)

    def render(self, render):
        light_np = render.attachNewNode(self.light)
        light_np.setPos(self.pos)
        light_np.lookAt(self.target)
        render.setLight(light_np)


class Constructor:
    def __init__(self):
        self._renderable = []
        self._engines = []
        self.parts = []

    def add_box(self, *args, **kwargs):
        box = Box(*args, **kwargs)
        self.physic.attachRigidBody(box.node)
        self._renderable.append(box)
        self.parts.append(box)
        return box

    def add_hinge_joint(self, *args, **kwargs):
        joint = HingeJoint(*args, **kwargs)
        self.physic.attachConstraint(joint.constr)
        self._engines.append(joint)
        return joint

    def add_point_light(self, color, pos):
        light = PointLight(color, pos)
        self._renderable.append(light)
        return light

    def add_spot_light(self, color, pos, target):
        light = SpotLight(color, pos, target)
        self._renderable.append(light)
        return light


__all__ = ['Constructor']

if __name__ == '__main__':
    box = Box()
