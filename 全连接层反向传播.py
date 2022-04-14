import numpy as numpy

class linear():
	def __init__(self, dim_in, dim_out):
	"""
	参数：
		dim_in:输入维度
		dim_out:输出维度
	"""	

	#初始化参数
	scale = np.sqrt(dim_in / 2)
	#np.random.standard_normal返回的是标准正态分布
	self.weight = np.random.standard_normal((dim_in, dim_out)) / scale
	self.bias = np.random.standard_normal(dim_out) / scale
	self.params = [self.weight, self.bias]

	def __call__(self, x):
		 """
        参数：
            X：这一层的输入，shape=(batch_size, dim_in)
        return：
            xw + b
        """
        self.x = x
        return self.forward()

    def forward(self):
    	return np.dot(self.x, self.weight) + self.bias

    def backward(self, d_out):
    	"""
        参数：
            d_out：输出的梯度, shape=(batch_size, dim_out)
        return：
            返回loss对输入 X 的梯度（前一层（l-1）的激活值的梯度）
        """
        d_x = np.dot(d_out, self.weight.T)
        d_w = np.dot(self.X.T, d_out)
        d_b = np.mean(d_out, axis = 0)
        return d_x, [d_w, d_b]


