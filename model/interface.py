from abc import ABCMeta, abstractmethod


class Renderable(metaclass=ABCMeta):
    def render(self, render):
        pass


class Constructable(metaclass=ABCMeta):
    def construct(self):
        pass
