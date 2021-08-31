import gym
import numpy as np
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from ddpg_agent import Agent
from collections import deque

channel = EngineConfigurationChannel()

channel.set_configuration_parameters(width=500, height=500, time_scale=10.0)
env = UnityEnvironment("./walker_1.app", side_channels=[channel])

# region Environment Description
env_info = env.reset()

# We will only consider the first Behavior
behavior_name = list(env.behavior_specs)[0]
print(f"Name of the behavior : {behavior_name}")
spec = env.behavior_specs[behavior_name]

# Examine the number of observations per Agent
print("Shape of observations : ", spec.observation_specs[0].shape)

# Is there a visual observation ?
# Visual observation have 3 dimensions: Height, Width and number of channels
vis_obs = any(len(spec.shape) == 3 for spec in spec.observation_specs)
print("Is there a visual observation ?", vis_obs)

# Is the Action continuous or multi-discrete ?
if spec.action_spec.continuous_size > 0:
    print(f"There are {spec.action_spec.continuous_size} continuous actions")
if spec.action_spec.is_discrete():
    print(f"There are {spec.action_spec.discrete_size} discrete actions")

# endregion

decision_steps, terminal_steps = env.get_steps(behavior_name)

agent = Agent(action_size=39, state_size=212, random_seed=0)


def ddpg(n_episodes: int = 9000, max_actions: int = 1000):
    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        tracked_agent = -1  # -1 indicates not yet tracking
        done = False  # For the tracked_agent
        episode_rewards = 0  # For the tracked_agent

        state = None

        while not done:
            # Track the first agent we see if not tracking
            # Note : len(decision_steps) = [number of agents that requested a decision]
            if tracked_agent == -1 and len(decision_steps) >= 1:
                tracked_agent = decision_steps.agent_id[0]

            if state is None:
                # Generate an action for all agents
                action_np = np.random.randn(1, 39)
                # action = spec.action_spec.random_action(len(decision_steps))
                action = ActionTuple()
                action.add_continuous(action_np)
            else:
                action_np = agent.act(state=state, add_random=False)
                action = ActionTuple()
                action.add_continuous(action_np)

            # Set the actions
            env.set_actions(behavior_name, action)

            # Move the simulation forward
            env.step()

            # Get the new simulation results
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            if tracked_agent in decision_steps:  # The agent requested a decision
                reward = decision_steps[tracked_agent].reward
                episode_rewards += reward
                next_state = decision_steps[tracked_agent].obs[0]


                if state is not None:
                    agent.step(state, action_np, next_state, reward, False)

                state = next_state

            if tracked_agent in terminal_steps:  # The agent terminated its episode
                reward = terminal_steps[tracked_agent].reward
                episode_rewards += reward
                next_state = terminal_steps[tracked_agent].obs[0]
                if state is not None:
                    agent.step(state, action_np, next_state, reward, True)

                state = None
                done = True

        scores.append(episode_rewards)
        scores_window.append(episode_rewards)

        # if episode_score % 10 == 0:
        #    agent.save_weights()

        print('\rEpisode {}\tavg Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tavg Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    env.close()
    print("Environment Closed")


ddpg()
