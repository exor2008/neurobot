import warnings
import matplotlib.pyplot as plt

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from neurobot.model.amoeba import get_environment
    from neurobot.agent.ddpg import get_ddpg_agent


env = get_environment()

agent = get_ddpg_agent(env)
##agent.load_weights(r'model\actormodel.h5', r'model\criticmodel.h5')
rewards = agent.train(300)

plt.plot(rewards)
plt.show()
