import ctypes
import logging
import random
import numpy as np

from panda3d.bullet import BulletWorld
from panda3d.core import ConfigVariableString
from direct.showbase.ShowBase import ShowBase
from basemodel import Renderable, WorldInfo, BaseEnvironment
from primitives import Constructor, BaseRobot, Camera


class AmoebaObservation(WorldInfo):
    def collect(self):
        obs = [(part.pos, np.deg2rad(part.hpr), part.ang_vel, part.lin_vel) for part in self.world.robot.parts]
        return np.asarray(obs).ravel()

    def reset(self):
        pass


class AmoebaDone(WorldInfo):
    def collect(self):
        dones = [func for name, func in AmoebaDone.__dict__.items() if name.startswith('_done')]
        return any([done(self) for done in dones])

    def _done_fall(self):
        return self.world.robot.body.pos[2] < -1

    def _done_time(self):
        return self.world.time >= 120

    def reset(self):
        pass


class AmoebaReward(AmoebaDone):
    def __init__(self, world):
        super(AmoebaReward, self).__init__(world)
        self.reset()

    def collect(self):
        rewards = [func for name, func in AmoebaReward.__dict__.items() if name.startswith('_reward')]
        return sum([reward(self) for reward in rewards])

    def _reward_advanse(self):
        old_range = self.max_range
        self.max_range = max(self.max_range, self.world.robot.body.pos[0])
        advance = self.max_range - old_range
        course = np.deg2rad(self.world.robot.body.hpr[0])
        r = advance * np.cos(course) #- advance * np.abs(np.sin(course))
        # print('advance r', r)
        return r

    def _reward_dist_to_center(self):
        r = - self.size_k * np.abs(self.world.robot.body.pos[1]) / 100.0
        # print('dist to centr r', r)
        return r

    def _reward_fall(self):
        return -1 if self._done_fall() else 0

    def reset(self):
        self.max_range = 0
        self.prev_x = self.world.robot.body.pos[0]
        self.size_k = self.world.floor.size[2] / self.world.robot.body.size[2]


class AmoebaWorld(Constructor, Renderable):
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
        self.floor = self.add_box(size=(100, 10, 0.1), pos=(90, 0, 0), friction=2, color='grey')

    def construct_light(self):
        COLOR = 1,1,1,1
        self.add_point_light(color=COLOR, pos=(0, 0, 3))
        self.add_point_light(color=COLOR, pos=(0, -3, 1))
        self.add_spot_light(color=COLOR, pos=(0, 5, 10), target=(0, 0, 0))

    def construct_camera(self, base):
        self.camera = Camera(base)
        self.camera.pos = 10, -30, 0
        self.camera.target = 0, 0, 0

    def render(self, render, base):
        self.construct_camera(base)
        [rend_obj.render(render) for rend_obj in self._renderable]

    def reset(self):
        self.time = 0
        self.robot.reset()


class AmoebaEnvironment(BaseEnvironment):
    def step(self, actions, dt=0.1):
        self.robot.act(actions)
        self.physic.doPhysics(dt)
        self.world.time += dt
        obs = self.observations()
        reward = self.collect_reward()
        done = self.is_done()
        return obs, reward, done

    def reset(self, torque=True):
        self.world.reset()
        self.reward.reset()
        if torque:
            self.torque_robot(low=80, heigh=85)
        return self.observer.collect()

    def render(self, render, base):
        self.world.render(render, base)
    
    def torque_robot(self, duration=5, low=60, heigh=80):
        angle = random.uniform(low, heigh) * random.choice([-1, 1])
        for _ in np.arange(0, duration, 0.1):
            self.robot.body.node.apply_torque((0, 0, angle))
            self.physic.doPhysics(0.1)
        self.robot.body.node.clear_forces()

    def _get_state_dim(self):
        return self.observations().shape[0]

    def _get_action_dim(self):
        return len(self.robot._engines)

    state_dim = property(_get_state_dim)
    action_dim = property(_get_action_dim)


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
        self._render = render
        [rend_obj.render(render) for rend_obj in self._renderable]


class AmoebaTestField:
    def __init__(self, env, agent):
        self.agent = agent
        self.env = env

    def render(self):
        user32 = ctypes.windll.user32
        screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        resolution = ConfigVariableString('win-size')
        val = '{0} {1}'.format(*screensize)
        resolution.setValue(val)
        app = ShowBase()
        self.env.render(render, base)
        taskMgr.doMethodLater(0.01, self.update, 'update')
        app.run()

    def update(self, task):
        state = self.env.observations()
        actions = self.agent.predict(state)
        state, reward, done = self.env.step(actions)
        if done:
            self.env.reset()
        return task.again


def get_environment():
    physic = BulletWorld()
    physic.setGravity(0, 0, -9.81)
    robot = AmoebaRobot(physic)
    world = AmoebaWorld(robot, physic)
    observer = AmoebaObservation(world)
    reward = AmoebaReward(world)
    done = AmoebaDone(world)
    env = AmoebaEnvironment(physic, robot, world, observer, reward, done)
    return env


def show(env, agent):
    AmoebaTestField(env, agent).render()