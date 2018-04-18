from abc import ABCMeta, abstractmethod
from collections import defaultdict
import ctypes
import logging

import numpy as np
from panda3d.bullet import BulletWorld
from panda3d.core import ConfigVariableString
from direct.showbase.ShowBase import ShowBase
from gears import Renderable
from primitives import Constructor
from robot import AmoebaRobot


class WorldInfo(metaclass=ABCMeta):
    def __init__(self, world):
        self.world = world

    @abstractmethod
    def collect(self):
        pass


class Observation(WorldInfo):
    def collect(self):
        obs = [(part.pos, np.deg2rad(part.hpr), part.ang_vel, part.lin_vel) for part in self.world.robot.parts]
        return np.asarray(obs).ravel()


class Done(WorldInfo):
    def collect(self):
        dones = [func for name, func in Done.__dict__.items() if name.startswith('_done')]
        return any([done(self) for done in dones])

    def _done_fall(self):
        return self.world.robot.body.pos[2] < -1

    def _done_time(self):
        return self.world.time >= 20


class Reward(Done):
    def __init__(self, world):
        super(Reward, self).__init__(world)
        self.max_range = 0

    def collect(self):
        rewards = [func for name, func in Reward.__dict__.items() if name.startswith('_reward')]
        return sum([reward(self) for reward in rewards])

    def _reward_advanse(self):
        old_range = self.max_range
        self.max_range = max(self.max_range, self.world.robot.body.pos[0])
        advance = self.max_range - old_range
        course = np.deg2rad(self.world.robot.body.hpr[0])
        return advance * np.cos(course) - advance * np.sin(course)

    def _reward_fall(self):
        return -1 if self._done_fall() else 0

    def reset(self):
        self.max_range = 0


class World(Constructor, Renderable):
    def __init__(self, robot, physic):
        self.physic = physic
        Constructor.__init__(self)
        self._renderable.append(robot)
        self.robot = robot
        self.construct()
        self.time = 0

    def construct(self):
        self.construct_floor()
        self.construct_light()

    def construct_floor(self):
        self.add_box(size=(100, 10, 0.1), pos=(90, 0, 0), color='grey')

    def construct_light(self):
        COLOR = 1,1,1,1
        self.add_point_light(color=COLOR, pos=(0, 0, 3))
        self.add_point_light(color=COLOR, pos=(0, -3, 1))
        self.add_spot_light(color=COLOR, pos=(0, 5, 10), target=(0, 0, 0))

    def construct_camera(self, base):
        self.camera = Camera(base)
        self.camera.pos = 10, -30, 0
        self.camera.target = 0, 0, 0

    def render(self, base):
        self.construct_camera(base)
        [rend_obj.render(render) for rend_obj in self._renderable]

    def reset(self):
        self.time = 0
        self.robot.reset()


class BaseEnvironment(metaclass=ABCMeta):
    def __init__(self, physic, robot, world, observer, reward, done):
        self.physic = physic
        self.robot = robot
        self.world = world
        self.observer = observer
        self.reward = reward
        self.done = done

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def render(self):
        pass

    def collect_reward(self):
        return self.reward.collect()

    def is_done(self):
        return self.done.collect()

    def observations(self):
        return self.observer.collect()


class Environment(BaseEnvironment):
    def step(self, actions, dt=0.1):
        self.robot.act(actions)
        self.physic.doPhysics(dt)
        self.world.time += dt
        obs = self.observations()
        reward = self.collect_reward()
        done = self.is_done()
        return obs, reward, done

    def reset(self):
        self.world.reset()
        self.reward.reset()
        return self.observer.collect()

    def render(self, agent):
        user32 = ctypes.windll.user32
        screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        resolution = ConfigVariableString('win-size')
        val = '{0} {1}'.format(*screensize)
        resolution.setValue(val)
        app = ShowBase()
        self.world.render(base)
        taskMgr.add(self.update, 'update')
        taskMgr.doMethodLater(.1, self.roboact, 'act', extraArgs=[agent], appendTask=True)
        app.run()

    def update(self, task):
        dt = globalClock.getDt()
        self.physic.doPhysics(dt)
        self.world.time += dt
        return task.again

    def roboact(self, agent, task):
        state = self.observations()
        actions = agent.predict(state)
        self.robot.act(actions)
        done = self.is_done()
        if done:
            self.reset()
        return task.again

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

        
__all__ = ['Environment', 'World', 'Observation', 'Reward', 'BulletWorld', 'Done']