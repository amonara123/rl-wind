import keras.models as km
import keras.layers as kl
import keras.optimizers as ko
import keras

import rl.agents.dqn as rld
import rl.policy as rlp
import rl.memory as rlm

import subpolicy as sb
from callbacks import CB

class Neuralmodel():
	
	def __init__(self, log, env, model='Sequential', nn_dict={0: ('Dense', 16, 'relu'),
		     1: ('Dense', 16, 'relu')}):
		self.log = log
		self.model = km.__dict__[model]()
		self.env = env

		self.nn_dict = nn_dict

		self.model.add(kl.Flatten(input_shape=(1,) + env.observation_space.shape))

		for ind, lay_spec in nn_dict.items():
			layer, nb_neur, actv = lay_spec
			self.model.add(kl.__dict__[layer](nb_neur, activation=actv))

		self.model.add(kl.Dense(self.env.nb_actions))
		self.model.add(kl.Activation('relu'))
		self.log.info('created model')
		print(self.model.summary())

	def get_model(self):
		self.log.info('got model')
		return(self.model)

class RLAgent():
	def __init__(self, log, model, nb_actions, env, policy='EpsGreedyQPolicy',
		     eps=0.2, memory='SequentialMemory', limit=1000,
		     window_length=1, comp_func='Adam', nb_steps=1000, 
		     visualize=False, verbose=2, target_model_update=1, gamma=0.5,
		     fp_weights='./loc_data/ag_weight.txt', load=False, 
		     callbacks={'callback_epoch_end': './loc_data/epoch_log.txt', 'callback_batch_end': './loc_data/batch_log.txt', 'callback_save_result': './loc_data/save.txt'}):

		self.env = env
		self.callbacks = callbacks
		self.load_flag = load
		self.fp = fp_weights
		self.log = log
		self.policy_id = policy
		self.eps = eps
		self.limit =limit
		self.window_length = window_length
		self.comp_func = comp_func
		self.nb_steps = nb_steps
		self.visualize = visualize
		self.verbose = verbose
		self.target_model_update = target_model_update
		self.gamma = gamma		

		subpolicy = rlp.__dict__[policy](eps=self.eps)
		self.policy =sb.SlicePolicy(self.policy_id, self.env.shpe, eps, steps=nb_steps)
		
		self.memory = rlm.__dict__[memory](limit=limit, 
						   window_length=window_length)
		self.log.info('loaded custom policy/memory correctly')
		self.dqn =  rld.DQNAgent(model=model, nb_actions=nb_actions, 
					 memory=self.memory, policy=self.policy,
					 target_model_update=self.target_model_update, gamma=self.gamma)
		
		

	def get_config(self):
		dqn_dict = dict()
		dqn_dict['policy'] = self.policy_id
		dqn_dict['eps'] = self.eps
		dqn_dict['limit'] = self.limit
		dqn_dict['window_length'] = self.window_length
		dqn_dict['comp_func'] = self.comp_func
		dqn_dict['nb_steps'] = self.nb_steps
		dqn_dict['visualize'] = self.visualize
		dqn_dict['verbose'] = self.verbose
		dqn_dict['fp_weights'] = self.save_weights(filepath=self.fp)
		dqn_dict['load_agent_weights_flag'] = self.load_flag
		dqn_dict['callbacks'] = self.callbacks
		dqn_dict['target_model_update'] = self.target_model_update
		dqn_dict['gamma'] = self.gamma

		self.log.info('created config dict')

		return(dqn_dict)

	
	def compile(self):
		self.dqn.compile(ko.__dict__[self.comp_func](lr=0.01), metrics=['mae'])
		if self.load_flag:
			self.load_weights(self.fp)
	
	def fit(self, env):
		cbs = list()
		for cb, fp in self.callbacks.items():
			c = CB(self.env, fp)
			cbs.append(c.cbs[cb]())
		self.dqn.fit(env, nb_steps=self.nb_steps, visualize=self.visualize, 
			     verbose=self.verbose, callbacks=cbs)

	def load_weights(self, filepath):
		self.dqn.load_weights(filepath)

	def save_weights(self, filepath, overwrite=True):
		self.dqn.save_weights(filepath, overwrite=True)
		return(filepath)
