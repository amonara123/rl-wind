import keras.callbacks as kc
import json
import time
import pandapower as pp
import os

class CB():
	def __init__(self, env, filepath='./loc_data/exp_log.json'):
		self.fp = filepath
		self.log = open(filepath, mode='w')
		self.cbs = {'callback_epoch_end': self.callback_epoch_end,
			    'callback_batch_end': self.callback_batch_end,
			    'callback_epoch_begin': self.callback_epoch_begin,
			    'callback_batch_begin': self.callback_batch_begin,
			    'callback_save_result': self.callback_save_result,
			    'callback_save_net': self.callback_save_net}
	
		self.start_time = time.time()
		self.env = env

		self.flag = True

	def callback_epoch_end(self):
		self.start_time = time.time()
		return(kc.LambdaCallback(on_epoch_begin = lambda epoch, logs: self.__set_start_time(epoch, logs), on_epoch_end= lambda epoch, logs: self.__epoch_logging(epoch, logs)))		
	
	def callback_batch_end(self):
		return(kc.LambdaCallback(on_batch_begin = lambda epoch, logs: self.__set_start_time(epoch, logs), on_batch_end= lambda batch, logs: self.__batch_logging(batch, logs)))

	def callback_epoch_begin(self):
		return(kc.LambdaCallback(on_epoch_begin= lambda epoch, logs: self.__epoch_logging(epoch, logs)))

	def callback_batch_begin(self):
		return(kc.LambdaCallback(on_batch_begin= lambda batch, logs: self.__batch_logging(batch, logs)))

	def callback_save_result(self):
		return(kc.LambdaCallback(on_episode_end= lambda epoch, logs:
self.__epoch_save(epoch, logs)))

	def callback_save_net(self):
		return(kc.LambdaCallback(on_batch_end= lambda batch, logs:
self.__batch_save_net(batch, logs)))

	def __batch_save_net(self, batch, logs):
		if not os.path.isdir('./nets'):
			os.system('mkdir nets')
		save_id = './nets/net{}.json'.format(batch)
		pp.to_json(self.env.env.net.net, save_id)

	def __epoch_save(self, epoch, logs):
		result = self.env.get_result()
		action_names, state_names = self.__create_key_list(result)
		fobj = open(self.fp, 'a')
		first_line = '# epoch: {} \n'.format(epoch)
		fobj.write(first_line)
		i = False
		for batch in result:
			header = '#' + str(batch) + '\t'
			line = str()
			for name_id in action_names:
				header += str(name_id) + '\t'
				line += str(result[batch]['action'][name_id]) + '\t'
			for name_id in state_names:
				header += str(name_id) + '\t'
				line += str(result[batch]['state'][name_id]) + '\t'
			header += '\n'
			line += '\n'
			if not i:
				fobj.write(header)
				i = True
			fobj.write(line)
		fobj.close()

	def __create_key_list(self, dic):
		a_nameid = list()
		s_nameid = list()
		for batch in dic:
			for name_id in dic[batch]['action']:
				if not name_id in a_nameid:
					a_nameid.append(name_id)
			for name_id in dic[batch]['state']:
				if not name_id in s_nameid:
					s_nameid.append(name_id)
		a_nameid.sort()
		s_nameid.sort()
		return((a_nameid, s_nameid))		


	def __set_start_time(self, epoch, logs):
		self.start_time = time.time()

	def __epoch_logging(self, epoch, logs):
		fobj = open(self.fp, 'a')
		log_dict = dict()
		keylist = list()
		log_dict['epoch'] = epoch
		for item, value in logs.items():
			log_dict[item] = value
			keylist.append(item)
		log_dict['time'] = time.time() - self.start_time 
		keylist.append('time')
		keylist.sort()
		header = '#'
		line = ''
		for item in keylist:
			header += item + '\t'
			line += str(log_dict[item]) + '\t'
		if self.flag:
			fobj.write(header + '\n')
			self.flag = False
		fobj.write(line + '\n')

	def __batch_logging(self, batch, logs):
		log_dict = dict()
		log_dict['batch'] = str(batch)
		for item, value in logs.items():
			log_dict[item] = str(value)
		self.log.write(json.dumps(log_dict) + '\n')

