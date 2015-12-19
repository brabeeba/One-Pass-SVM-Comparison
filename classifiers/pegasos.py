import numpy as np
import math

class Pegasos(object):
	"""docstring for Pegasos"""
	def __init__(self, reg, k, maxiter = 1000000, X=None, Y=None, check = False ):
		super(Pegasos, self).__init__()
		self.reg = reg
		self.k = k
		self.W = None
		self.iteration = 2
		self.maxiter = maxiter
		self.X = X
		self.Y = Y

		self.check = check

	def update(self, X, Y):
		if self.iteration > self.maxiter:
			return
		x_shape = X.shape
		y_shape = Y.shape


		if self.W is None:
			self.W = 1/math.sqrt(self.reg) * np.random.randn(x_shape[1] , y_shape[1])

		#Learning Rate
		eta = 1.0 / (self.reg * self.iteration)

		#Evaluation
		z = np.multiply(np.dot(X, self.W), Y)


		#Stochastic Gradient Descent Step
		self.W = (1.0 - eta * self.reg) * self.W - eta / self.k * self.vectorize_dloss(z) * np.multiply(X, Y.T).T

		#Projection Step
		projection = 1.0 / math.sqrt(self.reg) / (np.linalg.norm(self.W))
		if projection < 1.0:
			self.W = projection * self.W

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
		#ova svm
		else:
			#prediction
			prediction = np.argmax(np.dot(X, self.W), axis = 1) - np.argmax(Y, axis = 1)
			return len(prediction[prediction == 0]) / float(len(prediction))
		

	def loss(self, X, Y):
		loss = 1 - np.multiply(np.dot(X, self.W), Y)
		return np.sum(loss[loss > 0])


	def infer(self, X):
		return np.dot(X, self.W)
