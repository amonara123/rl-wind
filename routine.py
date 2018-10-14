import numpy as np
import config_handle as ch
import loc_env as env
import neuralmodel as nm
import pandapower.networks as pn
import os
import random

import logging

class Routine():
	def __init__(self, raw=True):
		print(raw)
		
		self.cnfg = ch.ConfigHandler()
		#self.logger.info('created config handler instance')
	
		if raw:
			self._init_raw_config(raw)
			self.logger.info('created raw config')
		
		self.config_dict = self.cnfg.grab_config(raw)
		self.logObj = Log(self.config_dict['log']['log_path'])
		self.logger = self.logObj.get_logger()
		self.logger.info('created config dict instance')
		self.net = self._init_network(raw, self.config_dict)
		self.logger.info('created net instance')

		self.main_env = env.Environment(self.net, self.logger, net_id=self.config_dict['env']['pp_name'], net_fp=self.config_dict['env']['file_path'])
		self.logger.info('created env instance')

		self.agents = dict()
		for item in self.config_dict['ag']:
			ag_id = item
			ag_dict = self.config_dict['ag'][ag_id]['agent']
			env_dict = self.config_dict['ag'][ag_id]['env']
			mod_dict = self.config_dict['ag'][ag_id]['model']
			io_dict = self.config_dict['ag'][ag_id]['iomap']

			loc_env = env.LocalEnvironment(self.main_env, ag_id, action_dict=io_dict['input'], observation_dict=io_dict['output'], reward_id=env_dict['reward_id'], max_step=env_dict['max_step'])
			self.logger.info('created local environment for {}'.format(ag_id))
			print(io_dict)
			loc_model = nm.Neuralmodel(self.logger, loc_env, nn_dict=mod_dict)
			self.logger.info('created neural model for {}'.format(ag_id))

			agent = self._init_agent(loc_env.nb_actions, loc_model, loc_env, ag_dict)
			self.logger.info('created agent {}'.format(ag_id))
			
			self.agents[ag_id] = (loc_env, loc_model, agent)

	def run(self, nb_runs=1):
		print('Training the following agents:')
		for i in range(nb_runs):
			for ag_id in self.agents:
				print(ag_id)
				loc_env, model, agent = self.agents[ag_id]
				loc_env.set_env(self.main_env)	
				agent.compile()
				self.logger.info('compiled agent, running exp')
				agent.fit(loc_env)
				self.logger.info('ran exp')
				self.main_env = loc_env.get_env()
			print('Run {} of {}: done'.format(i+1, nb_runs))
			self.logger.info('run {} of {} done'.format(i+1, nb_runs))
			
	
	def save_config(self):
		self.cnfg.gen_config(self.main_env, self.agents, log=self.logObj, raw=False)
		self.logger.info('saved config')	

	def _init_agent(self, nb_actions, model, env, ag_dict):
		dqn_dict = ag_dict
		agent = nm.RLAgent(self.logger, 
				   model.get_model(), nb_actions, env,
				   policy=dqn_dict['policy'], 
				   eps=dqn_dict['eps'], 
				   limit=dqn_dict['limit'], 
				   window_length=dqn_dict['window_length'],
				   comp_func=dqn_dict['comp_func'],
				   nb_steps=dqn_dict['nb_steps'], 
				   visualize=dqn_dict['visualize'],
				   verbose=dqn_dict['verbose'],
				   fp_weights=dqn_dict['fp_weights'],
				   load=dqn_dict['load_agent_weights_flag'],
				   callbacks=dqn_dict['callbacks'],
				   target_model_update = dqn_dict['target_model_update'],
				   gamma = dqn_dict['gamma'])
		return(agent)
	
	def _init_network(self, raw=True, config=dict()):
		if raw:
			net = pn.example_simple()
		else:
			net = config['net']
		return(net)

	def _init_raw_config(self, raw):
		self.logObj = Log()
		self.logger = self.logObj.get_logger()
		self.net = self._init_network(raw)

		self.main_env = env.Environment(self.net, self.logger)
		loc_env = env.LocalEnvironment(self.main_env, 'agent01')
		model = nm.Neuralmodel(self.logger, loc_env)
		nb_actions = loc_env.nb_actions
		agent = nm.RLAgent(self.logger, model.get_model(), nb_actions, loc_env)
		self.agents = {'agent01': (loc_env, model, agent)}
		self.cnfg.gen_config(self.main_env, self.agents,
				     log=self.logObj, raw=raw)
		print('Raw Configuration generated ...')
			

class Log():
	def __init__(self, path='./loc_data/exp.log'):
		self.path = path
		if not os.path.isdir('./loc_data'):
			os.system('mkdir loc_data')
		if not os.path.isfile('./loc_data/exp.log'):
			os.system('touch ./loc_data/exp.log')
		self.logger = logging.getLogger('exp')
		fh = logging.FileHandler(self.path)
		fh.setLevel(logging.DEBUG)
		self.logger.setLevel(logging.DEBUG)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		fh.setFormatter(formatter)
		self.logger.addHandler(fh)
		self.logger.info('created formatter instance')
	
	def get_logger(self):
		return(self.logger)
		
	def get_path(self):	
		return(self.path)

#a = routine(raw=True)
#a.run()
#a.save_config()

