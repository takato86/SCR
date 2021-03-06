import torch
import torch.nn as nn
import torch.nn.functional as F


class Source(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400):
        super(Source, self).__init__()
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            self.fc = nn.Linear(nb_states, hidden1)
            self.mu_head = nn.Linear(hidden1, nb_actions)
            self.sigma_head = nn.Linear(hidden1, nb_actions)
        else:
            self.fc = nn.Linear(nb_states, hidden1).cuda()
            self.mu_head = nn.Linear(hidden1, nb_actions).cuda()
            self.sigma_head = nn.Linear(hidden1, nb_actions).cuda()

    def forward(self, s):
        x = F.relu(self.fc(s))
        u = torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return u, sigma
