import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
from networks import ActorNet, CriticNet
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 100  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LEARNING_RATE_ACTOR = 1.5e-4  # learning rate of the actor
LEARNING_RATE_CRITIC = 2.1e-4  # learning rate of the critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():

    def __init__(self, action_size: int = 39, state_size: int = 212, random_seed: int = 1):
        """Initializes the Agent object. Who interacts with and learns from the environment

        """
        self.action_size = action_size
        self.seed = random_seed
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Initialize the actor and the critic (both the target and the local)
        self.actor_local = ActorNet(state_size=state_size, action_size=action_size).to(device)
        self.actor_target = ActorNet(state_size=state_size, action_size=action_size).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LEARNING_RATE_ACTOR)

        self.critic_local = CriticNet(state_size=state_size, action_size=action_size).to(device)
        self.critic_target = CriticNet(state_size=state_size, action_size=action_size).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=LEARNING_RATE_ACTOR)

        Agent.hard_update(self.actor_local, self.actor_target)
        Agent.hard_update(self.critic_local, self.critic_target)

    def step(self, state: np.array, action: np.array, next_state: np.array, reward: np.array, done: np.array) -> None:
        """
        After takin a step in the environment, add it to the memory buffer
        and learn if the time is
        :param state:
        :param action:
        :param next_state:
        :param reward:
        :param done:
        :return:
        """
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > BATCH_SIZE:
            exp = self.memory.sample()
            self.learn(exp, GAMMA)

    def learn(self, experiences: tuple, gamma: float):
        """
        Take from memory and learn from it using actor critic learning
        :return:
        """
        states, actions, rewards, next_states, dones = experiences


        # --- train the critic ---
        next_actions = self.actor_target.forward(state=next_states)
        # Get expected Q values from local critic by passing in both the states and the actions
        expected_Q = self.critic_local(states, actions)
        # Get next expected Q values from local critic by passing in both the next_states and the next_actions
        next_expected_Q = self.critic_local(next_states, next_actions)
        # Compute Q targets for current states
        target_Q = rewards + (gamma * next_expected_Q * (1 - dones))
        # Caclulate the loss function using the expected return and the target

        critic_loss = F.mse_loss(expected_Q, target_Q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optim.step()

        # --- train the actor ---
        # create the action predictions by passing the states to the local network
        actions_prediction = self.actor_local.forward(states)
        # calculate the loss function of the actor
        actor_loss = -self.critic_local(states, actions_prediction).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

    def act(self, state: np.array, add_random: bool = False) -> np.array:
        """
        Return an action, given a state
        :param add_random: determines whether the agents acts a little randomly
        :param state:
        :return:
        """

        state: torch.tensor = torch.from_numpy(state).float().to(device).unsqueeze(dim=0)

        self.actor_local.eval()

        with torch.no_grad():
            # Acquire an action by passing the current state to the local actor network
            action = self.actor_local.forward(state).cpu().data.numpy()
        self.actor_local.train()

        # Add noise to the action
        if add_random:
            action += np.random.normal(0, 0.5, size=self.action_size)

        return action

    def save_weights(self, path='./weights/') -> None:
        """Saves the weights of the local networks
        (both the agent and the critic)"""
        torch.save(self.critic_local.state_dict(), path + 'critic')
        torch.save(self.actor_local.state_dict(), path + 'actor')

    def restore_weights(self, path='./weights/') -> None:
        """Restore the saved local network weights to both the target and the local network"""

        self.critic_local.load_state_dict(torch.load(path + 'critic'))
        self.critic_local.eval()

        self.critic_target.load_state_dict(torch.load(path + 'critic'))
        self.critic_target.eval()

        self.actor_local.load_state_dict(torch.load(path + 'actor'))
        self.actor_local.eval()

        self.actor_target.load_state_dict(torch.load(path + 'actor'))
        self.actor_target.eval()

    @staticmethod
    def hard_update(local_model: nn.Module, target_model: nn.Module):
        """
        Copy all weights and biases from the local network to the target network
        :param local_model:
        :param target_model:
        :return:
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    @staticmethod
    def soft_update(local_model: nn.Module, target_model: nn.Module, tau=TAU):
        """
        Bring all weights and biases of the target network in the direction of the local network given the parameter tau
        :param tau: (factor of how fast the parameters should assimilate)
        :param local_model:
        :param target_model:
        :return:
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer():
    """Fixed size replay-buffer to store experience tuples."""

    def __init__(self, buffer_size: int, batch_size: int, seed: int):
        """
        Initialize the Replay Buffer
        :param buffer_size:
        :param batch_size:
        :param seed:
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = seed

    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the replay buffer
        :param state:
        :param action:
        :param reward:
        :param next_sate:
        :return:
        """
        e = self.experience(state=state, action=action, reward=np.array([reward]), next_state=next_state, done=np.array([done]))
        self.memory.append(e)

    def sample(self) -> tuple:
        """
        Randomly sample an experience from the replay buffer (add prioritzed replay later)
        :return:
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states: torch.Tensor = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(
            device)
        actions: torch.Tensor = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(
            device).squeeze()
        rewards: torch.Tensor = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(
            device)
        next_states: torch.Tensor = torch.from_numpy(
            np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones: torch.Tensor = torch.from_numpy(np.array([e.done for e in experiences if e is not None])).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of the internal memory"""
        return len(self.memory)
