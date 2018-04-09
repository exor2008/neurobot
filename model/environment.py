from abc import ABCMeta, abstractmethod
import logging

from panda3d.bullet import BulletWorld
from direct.showbase.ShowBase import ShowBase
from gears import Renderable
from primitives import Constructor


class BaseEnvironment(Constructor, Renderable):
    def __init__(self, world, robot):
        self._renderable = [robot]
        self.robot = robot
        self.world = world.bullet_world
        self.construct()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass


class FloorEnvironment(BaseEnvironment):
    def reset(self):
        logging.debug('Environment reset')

    def step(self, actions):
        logging.debug('Environment step')

    def construct(self):
        self.construct_floor()
        self.construct_light()

    def construct_floor(self):
        self.add_box(size=(40, 10, 0.1), color='grey')

    def construct_light(self):
        COLOR = 1,1,1,1
        self.add_point_light(color=COLOR, pos=(0, 0, 3))
        self.add_point_light(color=COLOR, pos=(0, -3, 1))
        self.add_spot_light(color=COLOR, pos=(0, 5, 10), target=(0, 0, 0))

    def construct_camera(self, base):
        self.camera = Camera(base)
        self.camera.pos = 10, -30, 0
        self.camera.target = 0, 0, 0

    def render(self):
        app = ShowBase()
        self.construct_camera(base)
        [rend_obj.render(render) for rend_obj in self._renderable]
        taskMgr.add(self.update, 'update')
        app.run()

    def update(self, task):
        dt = globalClock.getDt()
        self.world.doPhysics(dt)
        return task.cont


class DefaultWorld:
    def __init__(self):
        self.bullet_world = BulletWorld()
        self.bullet_world.setGravity(0, 0, -9.81)


class Camera:
    def __init__(self, base):
        self.base = base

    def _get_pos(self):
        return self.base.cam.GetPos()

    def _set_pos(self, pos):
        self.base.cam.setPos(pos)

    def _get_target(self):
        return self.base.cam.GetTarget()

    def _set_target(self, target):
        self.base.cam.lookAt(target)
        
    pos = property(_get_pos, _set_pos)
    target = property(_get_target, _set_target)

        
__all__ = ['FloorEnvironment', 'DefaultWorld']