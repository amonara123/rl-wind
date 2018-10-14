import loc_env as en
import json
import pandapower.networks as pn
import pandapower as pp
import os

class ConfigHandler():

	def grab_config(self, raw=False):
		if not os.path.isfile('./config'):
			raw=True
		fobj = open('./config', 'r')
		config = json.load(fobj)
		load_dict = dict()
		load_dict['ag'] = dict()
		for item in config['AGENTS']:
			load_dict['ag'][item] = dict()
			load_dict['ag'][item]['env'] = config['AGENTS'][item]['env']
			load_dict['ag'][item]['model'] = config['AGENTS'][item]['model']
			load_dict['ag'][item]['agent'] = config['AGENTS'][item]['agent']
			load_dict['ag'][item]['iomap'] = config['AGENTS'][item]['io_map']
		
		net = self.__get_net_from_config(config)
		load_dict['net'] = net
		load_dict['env'] = config['PP_NET']
		load_dict['gridIO'] = config['GRID_IO']['gridIO_fp']
		load_dict['log'] = config['LOG']
		#self.log.info('loaded dict from config and sliced it')

		return load_dict
		
	def gen_config(self, main_env, agents, log, raw=False):
		agent_ids = list()
		for item in agents:
			agent_ids.append(item)
		sorted(agent_ids)

		config = dict()
		config['GRID_IO'] = self.w_config_IO(main_env)
		#config['IO_MAP'] = self.w_config_iomap(env, raw=raw)
		#config['NN'] = self.w_config_model(model)
		config['PP_NET'] = self.w_pp_net(main_env)
		#config['ENV'] = self.w_environment(main_env)
		config['AGENTS'] = self.w_config_agent(agents)
		config['LOG'] = self.w_logger(log)
		if not os.path.isfile('config'):
			os.system('touch config')
		fobj = open('./config', 'w')
		json.dump(config, fobj, indent=4, sort_keys=True)
		#self.log.info('saved config')
		fobj.close()

	def w_logger(self, log):
		log_dict = dict()
		log_dict['log_path'] = log.get_path()
		#log_dict['agent_weights'] = agent.save_weights()
		return(log_dict)

	def w_config_IO(self, env):
		grid_IO = dict()
		for ctrl_id, controller in env.get_netobj().get_controller().items():
			grid_IO[ctrl_id] = dict()
			grid_IO[ctrl_id]['input'] = dict()
			inp_id = controller.get_config()['ctrl_in']
			for item in inp_id:
				inp_type = controller.get_space(item)[2]
				grid_IO[ctrl_id]['input'][item] = str(inp_type)
			grid_IO[ctrl_id]['output'] = dict()
			out_id = controller.get_config()['ctrl_out']
			for item in out_id:
				out_type = controller.get_space(item)[2]
				grid_IO[ctrl_id]['output'][item] = str(out_type)
		fp = env.get_config()['gridio_fp']
		fobj = open(fp, 'w')
		json.dump(grid_IO, fobj, indent=4, sort_keys=True)
		fobj.close()

		return({'gridIO_fp': fp})
	
	def w_config_iomap(self, env, raw=False):
		io_map = dict()
		if raw:	
			#default map
			io_map['input'] = dict()
			io_map['input']['trafo0'] = 'tp_pos'
			io_map['output'] = dict()

			for i in range(0, 4):
				io_map['output']['line{}'.format(i)] = 'loading_percent' 
		else:
			io_map['input'] = env.action_dict
			io_map['output'] =env.observation_dict

		return(io_map)
	
	def w_pp_net(self, env):
		pp_network = dict()
		arg = env.get_config()
		#use if pandapower net obj is exported:
		pp_network['file_path'] = arg['net_fp']
		#use if you want to use in-built network
		pp_network['pp_name'] = arg['net_id']
		
		if not os.path.isdir('./loc_data'):
			os.system('mkdir ./loc_data')
		if not os.path.isfile('./loc_data/pp_net.json'):
			os.system('touch ./loc_data/pp_net.json')
		pp.to_json(arg['net'], arg['net_fp'])
		
		return(pp_network)

	def w_config_agent(self, agents):
		ag_dict = dict()
		for item in agents:
			env, mod, ag = agents[item]
			ag_dict[item] = dict()
			ag_dict[item]['env'] = env.get_config()[0]
			ag_dict[item]['io_map'] = env.get_config()[1]
			ag_dict[item]['model'] = mod.nn_dict
			ag_dict[item]['agent'] = ag.get_config()
		return(ag_dict)

		
	def __get_net_from_config(self, config):
		if config['PP_NET']['pp_name'] == '':
			print('Grab net from file ...')
			net = pp.from_json(config['PP_NET']['file_path'])
		else: 
			print('Grab net from pandapower lib ...')
			net = pn.__dict__[config['PP_NET']['pp_name']]()
		return net

