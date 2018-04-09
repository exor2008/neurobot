from abc import ABCMeta, abstractmethod
import panda3d.core as core


class Renderable(metaclass=ABCMeta):
    @abstractmethod
    def render(self, render):
        pass


class RigidBody:
    def __init__(self, node, pos, hpr, mass, friction):
        self.node = node
        self._transform = core.TransformState.makePosHpr(pos, hpr)
        self.pos = pos
        self.hpr = hpr
        self.mass = mass
        self.friction = friction

    def _set_pos(self, pos):
        self._transform = self._transform.setPos(pos)
        self.node.set_transform(self._transform)

    def _get_pos(self):
        pos = self.node.get_transform().getPos()
        return pos.x, pos.y, pos.z

    def _set_hpr(self, hpr):
        self._transform = self._transform.setHpr(hpr)
        self.node.set_transform(self._transform)

    def _get_hpr(self):
        hpr = self.node.get_transform().getHpr()
        return hpr.x, hpr.y, hpr.z

    def _set_mass(self, mass):
        self.node.set_mass(mass)

    def _get_mass(self):
        return self.node.get_mass()

    def _set_friction(self, friction):
        self.node.set_friction(friction)

    def _get_friction(self):
        return self.node.get_friction()

    pos = property(_get_pos, _set_pos)
    hpr = property(_get_hpr, _set_hpr)
    mass = property(_get_mass, _set_mass)
    friction = property(_get_friction, _set_friction)

__all__ = ['Renderable', 'RigidBody']