import numpy as np
import math

class ASGD(object):
	"""docstring for ASGD"""
	def __init__(self, reg, r0=1e-2, a=1e-2, c=2.0/3, maxiter = 1000000, X=None, Y=None, check = False):
		super(ASGD, self).__init__()
		self.reg = reg
		self.r0 = r0
		self.a = a
		self.c = c

		self.maxiter = maxiter
		self.X = X
		self.Y = Y

		self.W = None
		self.SW = None 

		self.iteration = 1
		self.check = check

	def update(self, X, Y):
		if self.iteration > self.maxiter:
			return
		x_shape = X.shape
		y_shape = Y.shape

		assert len(y_shape) == 1, "This is a binary svm. Do ova or ovo method instead of multiclass."


		if self.W is None or self.SW is None:
			self.SW = 0.1 / math.sqrt(self.reg) * np.random.randn(x_shape[1] , 1) 
			self.W = np.copy(self.SW)
		

		eta = self.r0 * (1 + self.r0 * self.a * self.iteration) ** (-self.c)
		rho = 1.0 / (1 + self.iteration)


		#Evaluation
		p = np.multiply(np.dot(X, self.W), Y)

		#Get the support
		support = np.multiply(X, Y)
		if p[0][0] < 1:
			support = 0 * X

		#Stochastic Gradient Descent Step
		self.SW = (1.0 - eta * self.reg) * self.W + eta * support.T

		self.W = (1.0 - rho) * self.W + rho * self.SW


		if self.check:
			#Sanity Check on Loss Function
			if self.iteration % 1000 == 0:
				print "loss", self.loss(self.X, self.Y)
		

		self.iteration = self.iteration + 1

	def solve(self):
		pass

	def score(self, X, Y):
		
		prediction = np.multiply(np.dot(X, self.W), Y.reshape(Y.shape[0], 1))
		return len(prediction[prediction > 0]) / float(len(prediction))

	def loss(self, X, Y):
		loss = 1 - np.multiply(np.dot(X, self.W), Y.reshape(Y.shape[0], 1))
		temp = np.zeros(loss.shape)
		return np.sum(np.maximum(loss, temp))


	def infer(self, X):
		return np.dot(X, self.W)