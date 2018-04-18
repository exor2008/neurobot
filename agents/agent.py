from abc import ABCMeta, abstractmethod


class Agent(metaclass=ABCMeta):
	def __init__(self, env):
		self.env = env

	@abstractmethod
	def train(self):
		pass

	@abstractmethod
	def predict(self):
		pass
