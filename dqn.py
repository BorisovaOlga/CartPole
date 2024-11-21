import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
            x = F.relu(self.fc1(x)) # we send the state x through first layer. then pass that through to an activation function ReLu
            return self.fc2(x) # we send the result to output layer which ccalculates Q value
    
    

if __name__ == '__main__':
    state_dim = 4 # we have 4 state of our cart pole: cart position and velocity, pole angle and velocity
    action_dim = 2 # cart can make 2 action: move left and move right
    net = DQN(state_dim, action_dim)
    state = torch.randn(1, state_dim) # randn creates some random input
    output = net(state)
    print(output)