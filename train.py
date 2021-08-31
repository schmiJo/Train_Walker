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


def ddpg(n_episodes: int = 90, max_actions: int = 1000):
    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        episode_score = 0
        # The state observation at the start of each episode
        state = env.reset()

        score = 0.0

        for a in range(max_actions):

            # Get the new simulation results
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # Generate an action for all agents
            if state is None:
                action = np.random.randn(1, 39)  # select an action (for each agent)
            else:
                action = agent.act(state=state, add_random=i_episode < 100)

            print(action.shape, state is None)
            if state is not None:
                print(state.shape)

            print(len(decision_steps), len(terminal_steps))


            # Set the actions
            # it expects an action tuple
            action_tuple = ActionTuple()
            action_tuple.add_continuous(action)
            # Actually act in the simulated environment
            env.set_actions(behavior_name, action_tuple)

            # For all Agents with a Terminal Step
            for agent_id_terminated in terminal_steps:
                print("terminal")
                # Create its last experience (the Agent has previously terminated)
                reward = decision_steps[agent_id_terminated].reward

                episode_score += reward

            # For all Agents with a non terminal step
            for agent_id_step in decision_steps:

                reward = decision_steps[agent_id_step].reward

                episode_score += reward

                next_state = decision_steps[agent_id_step].obs[0]

                print('step')
                if state is not None:
                    agent.step(state=state, action=action, reward=reward, next_state=next_state, done=False)

                state = next_state

            env.step()  # take the action

        scores.append(episode_score)
        scores_window.append(episode_score)

        # if episode_score % 10 == 0:
        #    agent.save_weights()

        print('\rEpisode {}\tavg Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tavg Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    env.close()
    print("Environment Closed")


ddpg()
