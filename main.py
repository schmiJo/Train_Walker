import gym
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from ddpg_agent import Agent
from collections import deque

channel = EngineConfigurationChannel()

channel.set_configuration_parameters(width=500, height=500)
unity_env = UnityEnvironment("./walker.app", side_channels=[channel])
env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)

# Print out the state space
print(env.action_space)
# Print out the obervation space
print(env.observation_space)


def random_movement():
    max_t = 100
    for i_episode in range(8):

        rewards = []
        state = env.reset()
        for t in range(max_t):
            env.step(list(env.action_space.sample()))  # take a random action

    env.close()


agent = Agent(action_size=39, state_size=212, random_seed=0)


def ddpg(n_episodes: int = 90, max_actions: int = 1000):
    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        # The state observation at the start of each episode
        state = env.reset()

        score = 0.0

        for a in range(max_actions):
            actions = agent.act(state)
