from abc import ABCMeta, abstractmethod


class WorldInfo(metaclass=ABCMeta):
    def __init__(self, world):
        self.world = world

    @abstractmethod
    def collect(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class BaseEnvironment(metaclass=ABCMeta):
    def __init__(self, physic, robot, world, observer, reward, done):
        self.physic = physic
        self.robot = robot
        self.world = world
        self.observer = observer
        self.reward = reward
        self.done = done
        self._r = 0

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


class Renderable(metaclass=ABCMeta):
    @abstractmethod
    def render(self, render):
        pass


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

        
__all__ = ['WorldInfo', 'BaseEnvironment', 'Renderable', 'BaseJoint']