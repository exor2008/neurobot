from abc import ABCMeta, abstractmethod
import logging

class BaseEnvironment(metaclass=ABCMeta):
    def __init__(self, robot, floor, render=None):
        self.robot = robot
        self.render = render
        self.floor = floor

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass


class Environment(BaseEnvironment):
    def reset(self):
        logging.debug('Environment reset')

    def step(self, actions):
        logging.debug('Environment step')

        
__all__ = ['BaseEnvironment', 'Environment']