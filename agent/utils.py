import numpy as np
import scipy.signal


def orn_uhlen(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)


def discount(x, gamma=0.99):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]