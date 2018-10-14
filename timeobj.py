class InSimTime():
	def __init__(self, step=0, time=0, alpha=1):
		'''
		alpha: time = alpha * step ---> alpha == 1 time unit
		time: for late repercussion: time --- 1 min
		'''
		self.step = step
		self.time = time
		self.alpha = alpha

	def convert_time_to_step(self, time):
		step = int(time / self.alpha)
		return(step)

	def convert_step_to_time(self, step):
		time = step * self.alpha
		return(time)

	def get_state(self):
		return({'time': self.time, 'step': self.step, 'alpha': self.alpha})

	def incr_time(self, interval=1):
		self.time += interval * self.alpha
		self.step = int(self.time / self.alpha)

	def incr_step(self, interval=1):
		self.step += interval
		self.time = (self.step * self.alpha)
