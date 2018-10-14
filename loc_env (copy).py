import numpy as np
from rl.core import Env
from gym.spaces import Discrete, Box
import pandapower.networks as pn
import math 
import reward as rw
import pandas as pd

import netobj as ne
from timeobj import *

csv_data_wind = pd.read_csv('data.csv', sep=',')

class Environment():
	def __init__(self, net, log, net_id='example_simple', net_fp='./loc_data/pp_net.json', grid_fp='./loc_data/gridIO.json'):
		
		self.log = log
		self.t_handle = InSimTime()
		self.net = net
		self.netobj = ne.Netz(self.net, self.log, time=self.t_handle.get_state()['time'])
		self.net_id = net_id
		self.net_fp = net_fp

		self.grid_fp = grid_fp

	def step(self, action):
		raise NotImplementedError

	def close(self):
		raise NotImplementedError
	
	def reset(self):
		raise NotImplementedError

	def update_net(self, net):
		self.net = net
		self.netobj = ne.Netz(self.net, self.log, time=self.t_handle.get_state()['time'])

	def get_net(self):
		return(self.net)
	
	def get_netobj(self):
		return(self.netobj)

	def get_logger(self):
		return(self.log)

	def get_config(self):
		arg = dict()
		arg['net_id'] = self.net_id
		arg['net_fp'] = self.net_fp
		arg['net'] = self.net
		arg['gridio_fp'] = self.grid_fp
		return(arg)

		
class LocalEnvironment(Environment):
	def __init__(self, env, agent_id, action_dict={'trafo0': 'tp_pos'}, 
		     observation_dict={'line0': 'loading_percent', 'line1':'loading_percent', 'line2': 'loading_percent', 'line3': 'loading_percent'}, 
		     reward_id=('SimpleReward', {}), max_step=100):
		
		self.env = env
		self.id = agent_id
		self.t_handle = InSimTime()

		n = self.env.get_net()
		time = self.t_handle.get_state()['time']
		self.log = self.env.get_logger()
		self.netobj = ne.Netz(n, self.log, agent=self.id, time=time)

		self.max_step = max_step
		self.step_made = 0

		self.state = list()
		self.last_state = list()

		self.dtype = dict()
		
		self.a_dict = action_dict
		self.o_dict = observation_dict

		self.list_input_keys = list()
		for keys in self.a_dict.keys():
			self.list_input_keys.append(keys)

		self.init_action_space(self.a_dict)
		self.init_observation_space(self.o_dict)
		self.last_reward = 0

		self.act = self.__get_default_pos()
	
		self.shpe = list()
		self.nb_actions = 0
		for name in self.list_input_keys:
			self.nb_actions += int(self.act_shape[name])
			self.shpe.append(self.act_shape[name])
		print('nb actions: {}'.format(self.nb_actions))
		self.reward_id, self.reward_keys = reward_id

		self.results = dict()

	def get_env(self):
		return(self.env)

	def set_env(self, env):
		self.env = env
	
	def close(self):
		return(0)
	
	def reset(self):
		self.step_made = 0
		self.state = list()
		for ctrl_id, attr in self.o_dict.items():
			self.state.append(self.netobj.get_output()[ctrl_id][attr])
		self.netobj = ne.Netz(self.env.get_net(), self.env.get_logger(), agent=self.id, time=0)
		return(self.state)

	def get_result(self):
		return(self.results)

	def get_config(self):
		config_dict = dict()
		config_dict['max_step'] = self.max_step
		config_dict['reward_id'] = (self.reward_id, self.reward_keys)
		config_dict['nb_act'] = self.nb_actions
		config_dict['agent_id'] = self.id
		iodict = dict()
		iodict['input'] = self.a_dict
		iodict['output'] = self.o_dict
		self.env.get_logger().info('grabbed config')
		return((config_dict, iodict))

	def compute_action(self, name, action, shape, dtype):

		if self.a_dict[name] == 'scaling' or self.a_dict[name] == 'df':
			self.act[name] = 1/100 * (action+1)	
		else:			
			self.act[name] = action - (shape[name]-1)/2


	def step(self, action):
		#reset net:
		current_time = self.t_handle.convert_step_to_time(self.step_made)
		net = self.env.get_net()
		self.netobj = ne.Netz(net, self.log, agent=self.id, time=current_time)

		state = list()
		state_result = dict()
		action_result = dict()	

		for index in range(len(action)):
			act = action[index]
			ctrl = self.list_input_keys[index]
			attr = self.a_dict[ctrl]
			print((ctrl, attr))
			self.compute_action(ctrl, act, self.act_shape, dtype=self.dtype[ctrl])

			self.act[ctrl] = (np.clip(np.array([self.act[ctrl]]), 
				  	  		  self.act_min[ctrl], self.act_max[ctrl]))[0]
			self.netobj.set_input(ctrl, attr, self.act[ctrl])
			action_result[ctrl+attr] = self.act[ctrl]
			print('Input-Parameter changed to: {}'.format(self.act[ctrl]))
				

		#Can be used to dynamicly change the production of Wind turbines. 
		#Skript here
		self.netobj.run_powerflow()

		
		for ctrl_id, attr in self.o_dict.items():
			state.append(self.netobj.get_output()[ctrl_id][attr])
			print('{}: {} --- {}'.format(ctrl_id, attr, self.netobj.get_output()[ctrl_id][attr]))
			state_result[ctrl_id] = self.netobj.get_output()[ctrl_id][attr]


		reward_obj = rw.__dict__[self.reward_id](state, self.last_state, self.last_reward, self.reward_keys)#state slice to control the amount of input faktors
		reward = reward_obj.compute_reward() 

		self.step_made += 1
		done = False
		if self.step_made == self.max_step:# or reward < 0:
			done = True
		print('Steps in Episode: ({}/{})'.format(self.step_made, self.max_step))

		self.results[self.step_made] = {'state': state_result, 'action': action_result}
		self.last_reward = reward
		self.last_state = state

		self.t_handle.incr_step()

		self.env.get_logger().info('executed step; see exp data')
		self.env.update_net(self.netobj.net)
		return np.array(state), reward, done, {}


	def init_action_space(self, action_dict):
		self.act_max = dict()
		self.act_min = dict()
		self.act_shape = dict()
		high = list()
		low = list()
		for name, attr in action_dict.items():
			controller = self.netobj.get_controller()[name]
			l, h, datatype = controller.get_space(attr)
			low.append(l)
			high.append(h)
			self.act_max[name] = h
			self.act_min[name] = l
			self.dtype[name] = datatype
			if datatype == 'float' and (attr == 'scaling' or attr == 'df'):
				nb = 100
			elif datatype == 'float':
				nb = h/100 - l/100 + 1
			elif datatype == 'int':
				nb = h - l + 1
			self.act_shape[name] = nb
			
		#print(self.nb_actions)
			

		self.action_space = Box(np.array(low), np.array(high))#, dtype=datatype)

	def init_observation_space(self, observation_dict):
		high = list()
		low = list()
		for name, attr in observation_dict.items():
			controller = self.netobj.get_controller()[name]
			l, h, datatype = controller.get_space(attr)      
			low.append(l)
			high.append(h)
                                
		self.observation_space = Box(np.array(low), np.array(high), dtype=datatype)

	def __get_default_pos(self):
		act = dict()
		for ctrl, attr in self.a_dict.items():
			act[ctrl] = (self.netobj.get_input()[ctrl][attr])
		return(act)

