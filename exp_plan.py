import experiment as ex
import os
import json
import numpy as np

def step(reward, fp):
	if not os.path.isdir('./results'):
		os.system('mkdir results')
	fobj = open(fp, 'r')
	cnfg = json.load(fobj)
	fobj.close()

	reward_id = reward[0]
	rkeys = reward[1]

	key_list = list()
	for keys in rkeys.keys():
		key_list.append(keys)
	key_list.sort()		
	save_id = str(reward_id)
	for key in key_list:
		save_id += '_{}{}'.format(key, rkeys[key])

	cnfg['AGENT']['callbacks']['callback_epoch_end'] = './results/epoch_{}.txt'.format(save_id)
	cnfg['AGENT']['callbacks']['callback_save_result'] = './results/save_{}.txt'.format(save_id)

	cnfg['AGENT']['nb_steps'] = 4000
	cnfg['ENV']['max_step'] = 1000
	cnfg['ENV']['reward_id'] = reward

	cnfg['IO_MAP']['input'] = {'trafo0': 'tp_pos', 'trafo1': 'tp_pos', 'trafo2':'tp_pos', 'trafo3':'tp_pos', 'trafo4':'tp_pos', 'trafo5':'tp_pos', 'load0': 'scaling', 'load1':'scaling', 'load2':'scaling', 'load3':'scaling', 'load4':'scaling', 'gen0': 'scaling', 'gen1':'scaling', 'gen2':'scaling'}
	cnfg['IO_MAP']['output'] = {'bus1': 'vm_pu', 'bus2': 'vm_pu', 'bus3': 'vm_pu','bus4': 'vm_pu','bus5': 'vm_pu','bus6': 'vm_pu','bus7': 'vm_pu', 'bus8': 'vm_pu', 'bus9': 'vm_pu', 'bus10': 'vm_pu', 'bus11': 'vm_pu', 'bus0': 'vm_pu'}

	cnfg['NN']['0'] = ['Dense', 256, 'relu']
	cnfg['NN']['1'] = ['Dense', 256, 'relu']

	fobj = open(fp, 'w')
	json.dump(cnfg, fobj, indent=4, sort_keys=True)
	fobj.close()

	os.system('python3 experiment.py -r')

def get_rewards():
	#StdDist:
	rewards = list()
	#c:
	for i in np.arange(-0.9, 0, 0.1):
		#sigma:
		for j in np.arange(0.02, 0.22, 0.02):
			entry = ['StandardDist']
			entry.append({'c': float(i), 'mu': 1, 'sigma':float(j)})
			rewards.append(entry)
	
	#Polynom2:
	#a:
	for a in np.arange(10, 110, 10):
		#c:
		for c in np.arange(-5, -0.5, 0.5):
			entry = ['Polynomial2']
			entry.append({'a': float(a), 'mu': 1, 'c':float(c)})
			rewards.append(entry)
	
	#Dreieck:
	#b:
	for b in np.arange(-5, -0.5, 0.5):
		#a:
		for a in np.arange(5, 10.5, 0.5):
			#c:
			for c in np.arange(5, 7.5, 0.5):
				entry = ['Triangle']
				entry.append({'a': float(a), 'b':float(b), 'mu': 1, 'c':float(c)})
				rewards.append(entry)
	
	return(rewards)
	
	

def main():
	rewards = get_rewards()
	
	os.system('python3 experiment.py --getconfig')
	fp = './config'
	
	ind = 0
	for item in rewards[632:]:
		step(item, fp)
		ind += 1
		print('Step ({}/{})'.format(ind, len(rewards[632:])))

main()
	


