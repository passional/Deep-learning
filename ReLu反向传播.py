import numpy as np
class Relu(object):
	def __init__(self):
		self.x = None

	def __call__(self, x):
		self.x = x
		return forward(self.X)

	def forward(self, x):
		return np.maxinum(0, x)

	def backward(self, grad_output):
		"""
        grad_output: loss对relu激活输出的梯度
        return: relu对输入input_z的梯度
        """
        grad_relu = self.x > 0
        return grad_relu * grad_output
