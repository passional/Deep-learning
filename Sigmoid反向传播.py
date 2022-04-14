class sigmoid():

	def __init__(self):
		self.x = None

	def __call__(self, x):
		self.x = x
		return self.forward(self.x)

	def forward(self, x):
		return self._sigmoid(x)

	def backwrd(self, grad_output):
		sigmoid_grad = self._sigmoid(self.x) * (1 - self._sigmoid(self.x))
		return grad_output * sigmoid_grad

	def _sigmoid(self, x):
		return 1.0 / (1 + np.exp(-x))