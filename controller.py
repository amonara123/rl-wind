import pandapower as pp
import numpy as np

class Controller():
	def __init__(self, net):
		self.net = net
		self.max_float = 10000
		self.min_float = -10000
		self.ctrl_in = list()
		self.ctrl_out = list()
		self.space_dict = dict()
		self.init_controllable_input_list()
		self.init_controlled_output_list()
		self.init_space_dict()

	def chg_in(self, input_attr, input_value):
		self.pp_element.at[self.index, input_attr] = input_value
		return(self.net)

	def get_out(self):
		output = dict()
		for item in self.ctrl_out:
			output[item] = self.res_element.at[self.index, item]
		return(output)
	
	def get_in(self):
		inpt = dict()
		for item in self.ctrl_in:
			inpt[item] = self.pp_element.at[self.index, item]
		return(inpt)

	def get_config(self):
		config = dict()
		config['name'] = '{}{}'.format(self.name, self.index)
		config['ctrl_in'] = self.ctrl_in
		config['ctrl_out'] = self.ctrl_out
		config['output_states'] = self.get_out()
		return(config)

	def get_space(self, attr):
		#print(self.space_dict)
		return(self.space_dict[attr])
	 
	def init_controllable_input_list(self):
		raise NotImplementedError

	def init_controlled_output_list(self):
		raise NotImplementedError

	def init_space_dict(self):
		raise NotImplementedError

class Load(Controller):
	def __init__(self, net, index):
		self.name = 'load'
		self.index = index
		Controller.__init__(self, net)
		self.pp_element = getattr(self.net, '{}'.format(self.name))
		self.res_element = getattr(self.net, 'res_{}'.format(self.name))

	def init_controllable_input_list(self):
		self.ctrl_in.append('scaling')

	def init_controlled_output_list(self):
		self.ctrl_out.append('p_kw')
		self.ctrl_out.append('q_kvar')

	def init_space_dict(self):
		self.space_dict = dict()
		self.space_dict['p_kw'] = (0., self.max_float, 'float')
		self.space_dict['q_kvar'] = (0.001, self.max_float, 'float')
		self.space_dict['scaling'] = (0., 1., 'float')
		

class Line(Controller):
	def __init__(self, net, index):
		self.name = 'line'
		self.index = index
		Controller.__init__(self, net)
		self.pp_element = getattr(self.net, '{}'.format(self.name))
		self.res_element = getattr(self.net, 'res_{}'.format(self.name))

	def init_controllable_input_list(self):
		self.ctrl_in.append('df')
		
	def init_controlled_output_list(self):
		self.ctrl_out.append('pl_kw')
		self.ctrl_out.append('i_ka')
		self.ctrl_out.append('loading_percent')
		self.ctrl_out.append('ql_kvar')

	def init_space_dict(self):
		self.space_dict['df'] = (0., 1., 'float')
		self.space_dict['pl_kw'] = (self.min_float, 
					    self.max_float, 'float')
		self.space_dict['i_ka'] = (self.min_float,
					   self.max_float, 'float')
		self.space_dict['ql_kvar'] = (self.min_float,
					      self.max_float, 'float')
		self.space_dict['loading_percent'] = (0., 100., 'float')

class Trafo(Controller):
	def __init__(self, net, index):
		self.name = 'trafo'
		self.index = index
		Controller.__init__(self, net)
		self.pp_element = getattr(self.net, '{}'.format(self.name))
		self.res_element = getattr(self.net, 'res_{}'.format(self.name))
		
	def init_controllable_input_list(self):
		self.ctrl_in.append('tp_pos')
	
	def init_controlled_output_list(self):
		self.ctrl_out.append('pl_kw')
		self.ctrl_out.append('ql_kvar')

	def init_space_dict(self):
		self.space_dict['tp_pos'] = (-9, +9, 'int')
		self.space_dict['pl_kw'] = (self.min_float,
					    self.max_float, 'float')
		self.space_dict['ql_kvar'] = (self.min_float,
					      self.max_float, 'float')

class Gen(Controller):
	def __init__(self, net, index):
		self.name = 'gen'
		self.index = index
		Controller.__init__(self, net)
		self.pp_element = getattr(self.net, '{}'.format(self.name))
		self.res_element = getattr(self.net, 'res_{}'.format(self.name))

	def init_controllable_input_list(self):
		self.ctrl_in.append('scaling')

	def init_controlled_output_list(self):
		self.ctrl_out.append('p_kw')
		self.ctrl_out.append('q_kvar')
		self.ctrl_out.append('vm_pu')

	def init_space_dict(self):
		self.space_dict['vm_pu'] = (self.min_float,
					    self.max_float, 'float')
		self.space_dict['scaling'] = (0, 1, 'float')
		self.space_dict['p_kw'] = (self.min_float,
					   self.max_float, 'float')
		self.space_dict['q_kvar'] = (self.min_float,
					     self.max_float, 'float')

class Switch(Controller):
	def __init__(self, net, index):
		self.name = 'switch'
		self.index = index
		Controller.__init__(self, net)
		self.pp_element = getattr(self.net, '{}'.format(self.name))
		#self.res_element existiert nicht; 

	def init_controllable_input_list(self):
		return(0)

	def init_controlled_output_list(self):
		return(1)

	def init_space_dict(self):
		self.space_dict['closed'] = (0, 1, 'bool')

class SGen(Controller):
	def __init__(self, net, index):
		self.name = 'sgen'
		self.index = index
		Controller.__init__(self, net)
		self.pp_element = getattr(self.net, '{}'.format(self.name))
		self.res_element = getattr(self.net, 'res_{}'.format(self.name))

	def init_controllable_input_list(self):
		self.ctrl_in.append('scaling')

	def init_controlled_output_list(self):
		self.ctrl_out.append('p_kw')
		self.ctrl_out.append('q_kvar')

	def init_space_dict(self):
		self.space_dict['p_kw'] = (self.min_float, 0, 
					   'float')
		self.space_dict['q_kvar'] = (self.min_float, 
					     self.max_float, 'float')
		self.space_dict['scaling'] = (0, 1, 'float')

class Bus(Controller):
	def __init__(self, net, index):
		self.name = 'bus'
		self.index = index
		Controller.__init__(self, net)
		self.pp_element = getattr(self.net, '{}'.format(self.name))
		self.res_element = getattr(self.net, 'res_{}'.format(self.name))

	def init_controllable_input_list(self):
		return(0)

	def init_controlled_output_list(self):
		self.ctrl_out.append('p_kw')
		self.ctrl_out.append('q_kvar')
		self.ctrl_out.append('vm_pu')

	def init_space_dict(self):
		self.space_dict['p_kw'] = (self.min_float, 0, 
					   'float')
		self.space_dict['q_kvar'] = (self.min_float, 
					     self.max_float, 'float')
		self.space_dict['vm_pu'] = (0, self.max_float, 'float')
