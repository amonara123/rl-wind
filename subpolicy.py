from rl.policy import Policy
import rl.policy as rlp
import numpy as np # added

class SlicePolicy(Policy):

	def __init__(self, inner_policy, shape, eps, steps=100):
		super(SlicePolicy, self).__init__()

		self.eps_max = eps
		self.eps_min = 0.001
		self.inner_policy = inner_policy
		self.shape = shape
		self.steps = steps
		self.counter = 0
		
#here change?
	def select_action(self, q_values):
		
		eps, self.counter = self.anneale(self.counter)
		inner_policy = rlp.__dict__[self.inner_policy](eps)
		q_values = self.slice_qvalues(q_values, self.shape)
		action = list()
		for i in range(len(q_values)): #q_val length is 1
			q = q_values[i]
			action.append(inner_policy.select_action(q_values=q))
			#print(self.inner_policy.select_action(q))# usless due to random factor

			#print("******Debug Info******")
			#print("Q_values:",q_values)#debug
			print("MAx Q value",np.argmax(q_values))#debugg
			print("Action:",action)#debugg
			#print("Q_length",len(q_values)) #len is 1 ?
			#print("******Debug Info******")

		return(action)
			

	def slice_qvalues(self, q, shape):
		last_ind = 0
		res = list()
		shape_ind = 0
		for ind in range(0, len(q)+1):
			if ind == last_ind+shape[shape_ind] and ind > 0:
				newq = list()
				newq = q[last_ind:ind]
				last_ind = ind
				res.append(newq)
				shape_ind += 1
				
		return(res)
	
	def anneale(self, counter):
		a = -float(self.eps_max - self.eps_min) / float(self.steps)
		b = float(self.eps_max)
		eps = a*self.counter + b

		#modification becausesometime eps goes under 0. Should be investigated
		if eps < 0.00:
			eps = 0.00

		counter += 1
		print('Epsilon = {}'.format(eps))
		return((eps, counter))
