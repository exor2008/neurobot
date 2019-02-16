import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from neurobot.model.amoeba import get_environment, show
    from neurobot.agent.ddpg import get_ddpg_agent


env = get_environment()
agent = get_ddpg_agent(env)
agent.load_weights(r'model\actormodel.h5', r'model\criticmodel.h5')

##s, r, d = env.step([-1, -1])
show(env, agent)

