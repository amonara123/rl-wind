#!/usr/bin/python

from routine import Routine 
from optparse import OptionParser

def main():
	
	parser = OptionParser('experiment.py [Optionen]')

	parser.add_option('-c', '--init', action='store_true', dest='init', default=False, help='Init raw config')
	parser.add_option('-g', '--getconfig', action='store_true', dest='get_config', default=False, help='Generate config for edited pandapower grid')
	parser.add_option('-r', '--run', action='store_true', dest='run_flag', default=False, help='Run experiment; run once - edit # runs with (--steps, -s)')
	parser.add_option('-s', '--steps', action='store', dest='nb_steps', default=1, type='int', help='Int; number of steps to run the experiment; dont mix with other commands than (--run, -r)')
	
	(optionen, args) = parser.parse_args()
 
	assert type(optionen.nb_steps) == int, 'Error: number of runs must be int'

	if optionen.init:
		a = Routine(optionen.init)
	
	if optionen.get_config:
		a = Routine(raw=False)
		a.save_config()
		
	if optionen.run_flag:
		a = Routine(raw=False)
		a.run(optionen.nb_steps)
		a.save_config()

if __name__ == '__main__':
	main()
