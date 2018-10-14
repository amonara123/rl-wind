import numpy as np
import pandapower as pp
import pandapower.networks as pn
from controller import Load, Line, Trafo, Gen, SGen, Switch, Bus

class Netz():

	def __init__(self, net, log, agent=None, time=0):
		self.log = log
		self.net = net
		pp.runpp(self.net)
		self.id = agent
		self.time = time
		self.net_controller = dict()
		self.__create_net_controller_dict()
		self.log.info('set up net obj')

	def run_powerflow(self):
		pp.runpp(self.net)
		self.log.info('ran powerflow')

	def get_output(self):
		output = dict()
		self.__create_net_controller_dict()
		for name, controller in self.net_controller.items():
			output[name] = controller.get_out()
		#self.log.info('got net output') 
		return(output)
		

	def get_input(self):
		inpt = dict()
		for name, controller in self.net_controller.items():
			inpt[name] = controller.get_in() 
		self.log.info('got net input') 
		return(inpt)

	def set_input(self, ctrl_name, input_attr, input_value):
		for name, controller in self.net_controller.items():
			if name == ctrl_name:
				self.net = controller.chg_in(input_attr, input_value)
		self.log.info('set net input') 

	def get_controller(self):
		return(self.net_controller)
				

	def __create_net_controller_dict(self):
		#load:
		net_element = getattr(self.net, 'load')
		for index in range(0, len(net_element)):
			self.net_controller[str('load')+str(index)] = (Load(self.net, index))
		#line:
		net_element = getattr(self.net, 'line')
		for index in net_element.index:
			self.net_controller[str('line')+str(index)] = Line(self.net, index)
		#trafo:
		net_element = getattr(self.net, 'trafo')
		for index in net_element.index:
			self.net_controller['trafo' + str(index)] = Trafo(self.net, index)
		#gen:
		net_element = getattr(self.net, 'gen')
		for index in net_element.index:
			self.net_controller['gen'+str(index)] = Gen(self.net, index)
		#sgen:
		net_element = getattr(self.net, 'sgen')
		for index in net_element.index:
			self.net_controller['sgen'+str(index)] = SGen(self.net, index)
		#switch:
		net_element = getattr(self.net, 'switch')
		for index in net_element.index:
			self.net_controller['switch'+str(index)] = Switch(self.net, index)
		#bus:
		net_element = getattr(self.net, 'bus')
		for index in net_element.index:
			self.net_controller['bus'+str(index)] = Bus(self.net, index)
