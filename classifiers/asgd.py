import numpy as np
import math

class ASGD(object):
	"""docstring for ASGD"""
	def __init__(self, reg, r0=1e-2, a=1e-2, c=2.0/3, t0=0, maxiter = 1000000, X=None, Y=None, check = False):
		super(ASGD, self).__init__()
		self.reg = reg
		self.r0 = r0
		self.a = a
		self.c = c

		self.t0 = t0

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

		if self.W is None or self.SW is None:
			self.SW = 0.01 / math.sqrt(self.reg) * np.random.randn(x_shape[1] , y_shape[1]) 
			self.W = np.copy(self.SW)
		
		eta = self.r0 / (1 + self.r0 * self.a * self.iteration) ** (self.c)
		rho = 1.0 / max(1, self.iteration - self.t0)

		#Evaluation
		z = np.multiply(np.dot(X, self.W), Y)

		#Stochastic Gradient Descent Step
		self.SW = (1.0 - eta * self.reg) * self.SW - self.vectorize_dloss(z) * eta * np.multiply(X, Y.T).T

		self.W = (1.0 - rho) * self.W + rho * self.SW


		if self.check:
			#Sanity Check on Loss Function
			if self.iteration % 1000 == 0:
				print "loss", self.loss(self.X, self.Y)
		

		self.iteration = self.iteration + 1

	def solve(self):
		pass

	def vectorize_dloss(self, z):
		#L1-loss vectorize
		dloss = np.zeros(z.shape)
		dloss[z<1] = -1
		return dloss

	def score(self, X, Y):
		#binary svm
		if Y.shape[1] == 1:
			prediction = np.multiply(np.dot(X, self.W), Y)
			return len(prediction[prediction > 0]) / float(len(prediction))
		#ova
		else:
			#prediction
			prediction = np.argmax(np.dot(X, self.W), axis = 1) - np.argmax(Y, axis = 1)
			return len(prediction[prediction == 0]) / float(len(prediction))

	def loss(self, X, Y):
		loss = 1 - np.multiply(np.dot(X, self.W), Y)
		return np.sum(loss[loss > 0])


	def infer(self, X):
		return np.dot(X, self.W)