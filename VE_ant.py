import numpy as np
import torch
import torch.nn.functional as F
from Models import *
from utils.data_aug import random_translate



class VE(object):
	def __init__(self, env_name, state_dim, action_dim, goal_dim, de=None, alpha=0.1, Lambda=0.1, image_env=False, n_ensemble=10, gamma=0.99, \
	      		 tau=0.005, target_update_interval=1, sg_lr=1e-4, q_lr=1e-3, pi_lr=1e-4, enc_lr=1e-4, epsilon=1e-16, \
				 device=torch.device("cuda"), writer=None):		
		
		# Utils
		self.env_name =env_name
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = device
		self.writer = writer
		self.total_it = 0


		# Actor
		self.actor = GaussianPolicy(state_dim, goal_dim, action_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=pi_lr)
		self.actor_target = GaussianPolicy(state_dim, goal_dim, action_dim).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())

		# Critic
		self.critic 		= EnsembleCriticQ(state_dim, goal_dim, action_dim).to(device)
		self.critic_target 	= EnsembleCriticQ(state_dim, goal_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=q_lr)

		# Subgoal prediction 
		self.subgoal_net = LaplacePolicy(state_dim, goal_dim).to(device)
		self.subgoal_optimizer = torch.optim.Adam(self.subgoal_net.parameters(), lr=sg_lr)

		# Encoder (for vision-based envs)
		self.image_env = image_env
		if self.image_env:
			self.encoder = Encoder(state_dim=state_dim).to(device)
			self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=enc_lr)

		# Actor-Critic Hyperparameters
		self.tau = tau
		self.target_update_interval = target_update_interval
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon

		# Subgoal Planning Model Hyperparameters
		self.Lambda = Lambda
		self.n_ensemble = n_ensemble

	def save(self, folder, t, save_optims=False):
		torch.save(self.actor.state_dict(),		 folder + "actor.pth")
		torch.save(self.critic.state_dict(),		folder + "critic.pth")
		torch.save(self.subgoal_net.state_dict(),   folder + "subgoal_net.pth")
		if self.image_env:
			torch.save(self.encoder.state_dict(), folder + "encoder.pth")
		if save_optims:
			torch.save(self.actor_optimizer.state_dict(), 	folder + "actor_opti.pth")
			torch.save(self.critic_optimizer.state_dict(), 	folder + "critic_opti.pth")
			torch.save(self.subgoal_optimizer.state_dict(), folder + "subgoal_opti.pth")
			if self.image_env:
				torch.save(self.encoder_optimizer.state_dict(), folder + "encoder_opti")

	def load(self, folder):
		self.actor.load_state_dict(torch.load(folder+"actor.pth", map_location=self.device))
		self.critic.load_state_dict(torch.load(folder+"critic.pth", map_location=self.device))
		self.subgoal_net.load_state_dict(torch.load(folder+"subgoal_net.pth", map_location=self.device))
		if self.image_env:
			self.encoder.load_state_dict(torch.load(folder+"encoder.pth", map_location=self.device))

	def value(self, state, goal):
		_, _, action = self.actor.sample(state, goal)
		V = self.critic(state, action, goal).min(-1, keepdim=True)[0].clamp(min=-100.0, max=-1.0).abs()
		return V

	def select_action(self, state, goal):
		with torch.no_grad():
			state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
			goal = torch.FloatTensor(goal).to(self.device).unsqueeze(0)
			if self.image_env:
				state = state.view(1, 3, 84, 84)
				goal = goal.view(1, 3, 84, 84)
				state = self.encoder(state)
				goal = self.encoder(goal)
			action, _, _ = self.actor.sample(state, goal)
		return action.cpu().data.numpy().flatten()

	def sample_subgoal(self, state, goal, subgoal_net, n_ensemble):
		subgoal_distribution, scale = subgoal_net(state, goal)
		subgoals = subgoal_distribution.rsample((n_ensemble,))
		subgoals = torch.transpose(subgoals, 0, 1)
		return subgoals, subgoal_distribution.loc
	
	def get_subgoal(self, state, goal):
		with torch.no_grad():
			if not torch.is_tensor(state):
				state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
			if not torch.is_tensor(goal):
				goal = torch.FloatTensor(goal).to(self.device).unsqueeze(0)
			subgoal_distribution, _ = self.subgoal_net(state, goal)
		return subgoal_distribution, subgoal_distribution.loc

	def sample_action_and_KL(self, state, goal):
		batch_size = state.size(0)
		# Sample action, subgoals and KL-divergence
		action_dist = self.actor(state, goal)
		action = action_dist.rsample()

		with torch.no_grad():
			subgoal, subgoal_   = self.sample_subgoal(state, goal, self.subgoal_net, self.n_ensemble)
		action_dists = self.actor(state.unsqueeze(1).expand(batch_size, subgoal.size(1), self.state_dim), goal.unsqueeze(1).expand(batch_size, subgoal.size(1), self.state_dim))
		prior_action_dist = self.actor_target(state.unsqueeze(1).expand(batch_size, subgoal.size(1), self.state_dim), subgoal)
		prior_actions = prior_action_dist.rsample()
		D_KL = -action_dists.log_prob(prior_actions).sum(-1, keepdim=True).mean(1)
		action = torch.tanh(action)
	
		return action, D_KL

	def train_subgoal_model(self, state, goal, subgoal):
		# Compute subgoal distribution 
		subgoal_distribution, scale = self.subgoal_net(state, goal)

		with torch.no_grad():
			# Compute target value
			new_subgoal = subgoal_distribution.loc

			policy_v_1 = self.value(state, new_subgoal)
			policy_v_2 = self.value(new_subgoal, goal)
			policy_v = torch.cat([policy_v_1, policy_v_2], -1).max(-1)[0]

			# Compute subgoal distance loss
			if state.shape != subgoal.shape:
				state_ = state.unsqueeze(1).expand(subgoal.size(0), subgoal.size(1), self.state_dim)
				goal_ = goal.unsqueeze(1).expand(subgoal.size(0), subgoal.size(1), self.state_dim)
				v_1 = self.value(state_, subgoal)
				v_2 = self.value(subgoal, goal_)

				v, idxs = torch.cat([v_1, v_2], -1).max(-1)[0].min(-1)
				subgoal_ = torch.zeros_like(state)
				for i,n in enumerate(idxs):
					subgoal_[i] = subgoal[i,n,:]
				subgoal = subgoal_
			else:
				v_1 = self.value(state, subgoal)
				v_2 = self.value(subgoal, goal)
				v = torch.cat([v_1, v_2], -1).max(-1)[0]

			adv = - (v - policy_v)

		log_prob = subgoal_distribution.log_prob(subgoal).sum(-1)

		return adv, log_prob


	def train(self, state, action, reward, next_state, done, goal, her_subgoal, bu_subgoal, t):
		""" Encode images (if vision-based environment), use data augmentation """
		if self.image_env:
			state = state.view(-1, 3, 84, 84)
			next_state = next_state.view(-1, 3, 84, 84)
			goal = goal.view(-1, 3, 84, 84)
			subgoal = subgoal.view(-1, 3, 84, 84)

			# Data augmentation
			state = random_translate(state, pad=8)
			next_state = random_translate(next_state, pad=8)
			goal = random_translate(goal, pad=8)
			subgoal = random_translate(subgoal, pad=8)

			# Stop gradient for subgoal goal and next state
			state = self.encoder(state)
			with torch.no_grad():
				goal = self.encoder(goal)
				next_state = self.encoder(next_state)
				subgoal = self.encoder(subgoal)
		
		# Compute target Q
		with torch.no_grad():
			# Q TO Q
			next_action, _, _ = self.actor.sample(next_state, goal)
			target_Q = self.critic_target(next_state, next_action, goal)
			target_Q = torch.min(target_Q, -1, keepdim=True)[0]
			target_Q = reward + (1.0-done) * self.gamma*target_Q

		# Compute critic loss
		Q = self.critic(state, action, goal)
		td_delta = Q - target_Q
		critic_loss = 0.5 * (td_delta).pow(2).sum(-1).mean()

		# Optimize the critic
		if self.image_env: self.encoder_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		if self.image_env: self.encoder_optimizer.step()
		self.critic_optimizer.step()

		# Stop backpropagation to encoder
		if self.image_env:
			state = state.detach()
			goal = goal.detach()
			subgoal = subgoal.detach()

		""" Subgoal planning model learning """
		her_batch_size = her_subgoal.size(0)
		her_adv, her_logp = self.train_subgoal_model(state[:her_batch_size,:], goal[:her_batch_size,:], her_subgoal)
		buffer_adv, buffer_logp = self.train_subgoal_model(state[her_batch_size:,:], goal[her_batch_size:,:], bu_subgoal)
		adv = torch.cat([her_adv, buffer_adv], 0)
		logp = torch.cat([her_logp, buffer_logp], 0)
		weight = F.softmax(adv/self.Lambda, dim=0)
		subgoal_loss = - (logp * weight).mean()

		self.subgoal_optimizer.zero_grad()
		subgoal_loss.backward()
		self.subgoal_optimizer.step()

		""" Actor """
		# Sample action
		action, D_KL = self.sample_action_and_KL(state, goal)

		# Compute actor loss
		Q = self.critic(state, action, goal)
		Q = torch.min(Q, -1, keepdim=True)[0]

		# Add self-imitation learning loss
		actor_loss = (self.alpha*D_KL - Q).mean()

		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update target networks
		self.total_it += 1
		if self.total_it % self.target_update_interval == 0:
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		
		# tensorboard log
		self.writer.add_scalar("critic_loss", critic_loss.item(),t)
		self.writer.add_scalar("actor_loss", actor_loss.item(),t)