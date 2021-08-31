import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    """
    The Actor network of the walking humaniod
    """

    def __init__(self, state_size: int = 212, action_size: int = 39, fc1_size: int = 512, fc2_size: int = 265,
                 fc3_size: int = 128, seed: int = 0):
        """
        Initailize the parameters for the actor
        :param state_size:
        :param action_size:
        :param fc1_size:
        :param fc2_size:
        :param fc3_size:
        """
        super(ActorNet, self).__init__()

        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, action_size)

        self.seed = torch.manual_seed(seed)

        self.tanh = nn.Tanh()

        print("Initialized ActorNet successfully")

    def forward(self, state: torch.tensor) -> torch.tensor:
        """The straight forward feeding neural network"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        return x


class CriticNet(nn.Module):
    """
    The Critic network of the walking humanoid
    """

    def __init__(self, state_size: int = 212, action_size: int = 39, fc1_size: int = 512, fc2_size: int = 265,
                 fc3_size: int = 128, seed: int = 0):
        """
        Initailize the parameters for the actor
        :param state_size:
        :param action_size:
        :param fc1_size:
        :param fc2_size:
        :param fc3_size:
        """
        super(CriticNet, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size + action_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, 1)

        print("Initialized CriticNet successfully")

    def forward(self, state: torch.tensor, action: torch.tensor) -> torch.tensor:
        """The straight forward feeding neural network"""
        x = F.relu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
