import numpy as np 

def orn_uhlen(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)