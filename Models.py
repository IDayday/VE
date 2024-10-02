import torch
from torch import nn
import numpy as np


""" Actor """

class GaussianPolicy(nn.Module):
	def __init__(self, state_dim, goal_dim, action_dim, hidden_dims=[256, 256]):
		super(GaussianPolicy, self).__init__()
		fc = [nn.Linear(state_dim + goal_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc = nn.Sequential(*fc)

		self.mean_linear = nn.Linear(hidden_dims[-1], action_dim)
		self.logstd_linear = nn.Linear(hidden_dims[-1], action_dim)

		self.LOG_SIG_MIN, self.LOG_SIG_MAX = -20, 2

	def forward(self, state, goal):
		x = self.fc(torch.cat([state, goal], -1))
		mean = self.mean_linear(x)
		log_std = self.logstd_linear(x)
		std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX).exp()
		normal = torch.distributions.Normal(mean, std)
		return normal

	def sample(self, state, goal):
		normal = self.forward(state, goal)
		x_t = normal.rsample()
		action = torch.tanh(x_t)
		log_prob = normal.log_prob(x_t)
		log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
		log_prob = log_prob.sum(-1, keepdim=True)
		mean = torch.tanh(normal.mean)
		return action, log_prob, mean
	
	def get_no_gradient(self, state, goal):
		x = self.fc(torch.cat([state, goal], -1))
		mean = self.mean_linear(x)
		log_std = self.logstd_linear(x)
		std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX).exp()
		normal = torch.distributions.Normal(mean.detach(), std.detach())
		return normal
	
	def get_feature(self, state, goal):
		x = self.fc(torch.cat([state, goal], -1))
		return x



""" Critic Q"""

class CriticQ(nn.Module):
	def __init__(self, state_dim, goal_dim, action_dim, hidden_dims=[256, 256]):
		super(CriticQ, self).__init__()
		fc = [nn.Linear(state_dim + action_dim + goal_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim1, hidden_dim2 in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU()]

		fc += [nn.Linear(hidden_dims[-1], 1)]
		self.fc = nn.Sequential(*fc)

	def forward(self, state, action, goal):
		x = torch.cat([state, action, goal], -1)
		return self.fc(x)


class EnsembleCriticQ(nn.Module):
	def __init__(self, state_dim, goal_dim, action_dim, hidden_dims=[256, 256], n_Q=2):
		super(EnsembleCriticQ, self).__init__()
		ensemble_Q = [CriticQ(state_dim=state_dim, goal_dim=goal_dim, action_dim=action_dim, hidden_dims=hidden_dims) for _ in range(n_Q)]			
		self.ensemble_Q = nn.ModuleList(ensemble_Q)
		self.n_Q = n_Q

	def forward(self, state, action, goal):
		Q = [self.ensemble_Q[i](state, action, goal) for i in range(self.n_Q)]
		Q = torch.cat(Q, dim=-1)
		return Q

""" Critic V"""

class CriticV(nn.Module):
	def __init__(self, state_dim, hidden_dims=[256, 256]):
		super(CriticV, self).__init__()
		fc = [nn.Linear(2*state_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim1, hidden_dim2 in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim1, hidden_dim2), nn.ReLU()]

		fc += [nn.Linear(hidden_dims[-1], 1), nn.Tanh()]
		self.fc = nn.Sequential(*fc)

	def forward(self, state, goal):
		x = torch.cat([state, goal], -1)
		return self.fc(x) * 50 - 50


class EnsembleCriticV(nn.Module):
	def __init__(self, state_dim, hidden_dims=[64, 64], n_Q=2):
		super(EnsembleCriticV, self).__init__()
		ensemble_Q = [CriticV(state_dim=state_dim, hidden_dims=hidden_dims) for _ in range(n_Q)]			
		self.ensemble_Q = nn.ModuleList(ensemble_Q)
		self.n_Q = n_Q

	def forward(self, state, goal):
		Q = [self.ensemble_Q[i](state, goal) for i in range(self.n_Q)]
		Q = torch.cat(Q, dim=-1)
		return Q
	
""" Subgoal Planning Model """

class LaplacePolicy(nn.Module):	
	def __init__(self, state_dim, goal_dim, hidden_dims=[256, 256]):	
		super(LaplacePolicy, self).__init__()	
		fc = [nn.Linear(state_dim + goal_dim, hidden_dims[0]), nn.ReLU()]
		for hidden_dim_in, hidden_dim_out in zip(hidden_dims[:-1], hidden_dims[1:]):
			fc += [nn.Linear(hidden_dim_in, hidden_dim_out), nn.ReLU()]
		self.fc = nn.Sequential(*fc)

		self.mean = nn.Linear(hidden_dims[-1], goal_dim)	
		self.log_scale = nn.Linear(hidden_dims[-1], goal_dim)	
		self.LOG_SCALE_MIN = -20	
		self.LOG_SCALE_MAX = 2	

	def forward(self, state, goal):	
		h = self.fc( torch.cat([state, goal], -1) )	
		mean = self.mean(h)
		scale = self.log_scale(h).clamp(min=self.LOG_SCALE_MIN, max=self.LOG_SCALE_MAX).exp()
		distribution = torch.distributions.laplace.Laplace(mean, scale)
		return distribution, scale

""" Encoder """
def weights_init_encoder(m):
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class Encoder(nn.Module):
	def __init__(self, n_channels=3, state_dim=16):
		super(Encoder, self).__init__()
		self.encoder_conv = nn.Sequential(
			nn.Conv2d(n_channels, 32, 3, 2), nn.ReLU(),
			nn.Conv2d(32, 32, 3, 2), nn.ReLU(),
			nn.Conv2d(32, 32, 3, 2), nn.ReLU(),
			nn.Conv2d(32, 32, 3, 1), nn.ReLU()
		)
		self.fc = nn.Linear(32*7*7, state_dim)
		self.apply(weights_init_encoder)

	def forward(self, x):
		h = self.encoder_conv(x).view(x.size(0), -1)
		state = self.fc(h)
		return state

