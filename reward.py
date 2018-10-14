import numpy as np
import math

class Reward():
	def __init__(self, state, last_state, last_reward):
		self.state = state
		self.last_reward = last_reward
		self.last_state = last_state
		
	def compute_reward(self):
		raise NotImplementedError

class SimpleReward(Reward):
	def __init__(self, state, last_state, last_reward, keys={}):
		Reward.__init__(self, state, last_state, last_reward)

	def compute_reward(self):
		reward = 0
		nb = 0
		for lp in self.state:
			reward += 1 - math.sqrt((lp/100)**2)
			nb += 1
		reward = reward/nb
		print(np.mean(np.array(self.state)))
		
		print('Reward for this action: {}'.format(reward))
		return(reward)

class VoltageControl(Reward):
	def __init__(self, state, last_state, last_reward, keys={}):
		Reward.__init__(self, state, last_state, last_reward)

	def compute_reward(self):
		reward = 0
		nb = 0
		for vm in self.state:
			reward += 1/math.sqrt(1 + vm**2)
			nb += 1
		reward = reward/nb
		
		print('Reward for this action: {}'.format(reward))
		return(reward)

class StandardDist(Reward):
	def __init__(self, state, last_state, last_reward, keys={'mu':1, 'sigma':0.1, 'c':0.5}):
		Reward.__init__(self, state, last_state, last_reward)
		self.mu = keys['mu']
		self.sigma = keys['sigma']
		self.c = keys['c']

	def compute_reward(self):
		val = np.mean(np.array(self.state))
		
		def norm(x, mu=self.mu, sigma=self.sigma, c=self.c):
			print(self.state, mu, sigma, c)
			f = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-np.square(x-mu) / (2 * sigma + 225)) - c
			return(f)
	
		reward = norm(val)

		print('Reward for this action: {}'.format(reward))
		return(reward)

class Polynomial2(Reward):
	
	def __init__(self, state, last_state, last_reward, keys={'a':100, 'mu':1, 'c':1}):
		""" Polynominal reward function that follows
		f(x) = -a*(x - mu)**2 + c
		"""
		Reward.__init__(self, state, last_state, last_reward)
		self.a = keys['a']		
		self.mu = keys['mu']
		self.c = keys['c']

	def compute_reward(self):
		val = np.mean(np.array(self.state))

		def pol2(x, a=self.a, mu=self.mu, c=self.c):
			f = a*(x - mu)**2 + c
			return(f)

		reward = pol2(val)
		
		print('Reward for this action: {}'.format(reward))
		print('Mean state: {}'.format(self.state))
		return(reward)

class Triangle(Reward):
	def __init__(self, state, last_state, last_reward, keys={'a':5, 'b':1, 'mu':1, 'c':None}):
		"""
		Triangle reward function, that follows
		f(x) = a * (x - mu) + b | x < mu
		f'(x) = c * (x - mu) + b | x >= mu
		a, c: steepness
		mu: x-value of highest point in triangle
		b: height of triangle at mu
		if no c specified, triangle function becomes symmetric:
		g(x) = f(x)
		g'(x) = - a * (x - mu) + b
		"""
		Reward.__init__(self, state, last_state, last_reward)
		self.a = keys['a']
		self.b = keys['b']
		self.mu = keys['mu']
		if keys['c'] == None:
			self.c = -keys['a']
		else:
			self.c = keys['c']

	def compute_reward(self):
		val = np.mean(np.array(self.state))
		
		def tri(x, a=self.a, b=self.b, mu=self.mu, c=self.c):
			if x >= mu:
				f = a * (x - mu) + b
			elif x < mu:
				f = c * (x - mu) + b
			return f
	
		reward = tri(val)

		print('Reward for this action: {}'.format(reward))
		return(reward)


class Poisson(Reward):
	def __init__(self, state, last_state, last_reward, keys={'a':1, 'mu':1, 'b':0.5}):
		"""
		reward function following poisson dist:
		f(x) = a * mu^x * e^( -mu ) / (x!) - b
		warning: not appropriate for very high mu
		"""
		Reward.__init__(self, state, last_state, last_reward)
		self.a = keys['a']
		self.mu = keys['mu']
		self.b = keys['b']

	def compute_reward(self):
		val = np.mean(np.array(self.state))

		def poisson(x, a=self.a, b=self.b, mu=self.mu):
			f = -a * mu**x * math.exp(-mu) / (math.factorial(x)) - b
			return(f)

		reward = poisson(val)

		print('Reward for this action: {}'.format(reward))
		return(reward)
		
		




