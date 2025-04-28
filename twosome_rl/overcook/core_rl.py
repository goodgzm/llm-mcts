import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def combined_shape(length, shape = None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation = nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis = 0)[::-1]

def statistics_scalar(arr):
    arr_sum, arr_len = np.sum(arr), len(arr)
    arr_mean = arr_sum / arr_len
    arr_sum_sq = np.sum((arr - arr_mean) ** 2)
    arr_std = np.sqrt(arr_sum_sq / arr_len)
    return arr_mean, arr_std

class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act = None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp(list(obs_dim) + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits = logits)
    
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)
    
class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp(list(obs_dim) + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)

class MLPActorCritic(nn.Module):
    def __init__(self, observation_dim, action_dim,
                 hidden_sizes = (128, 128, 64), activation = nn.Tanh):
        super().__init__()
        self.pi = MLPCategoricalActor(observation_dim, action_dim, hidden_sizes, activation)

        self.v = MLPCritic(observation_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            # print(f"############action:{a}")
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()
    
    def act(self, obs):
        return self.step(obs)[0]

    
    

